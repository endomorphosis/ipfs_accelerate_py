"""Admit one complete atomic transitive repair plan (RPR-039).

``ChangePropagationPlanner@1`` consumes a complete impact closure, consumer
obligations, value-mapping proofs, analytical transforms, and support-behavior
placement decisions.  It returns either:

* one content-addressed :class:`AtomicPropagationPlan` with
  ``PlanDisposition.ADMITTED`` that covers every mandatory consumer under a
  deterministic step DAG and SCC transaction grouping, or
* an explicit abstention (``PlanDisposition.ABSTAINED``) that grants no write
  authority.

Construction never edits code, never executes providers, and never expands
paths beyond admitted evidence.  The canonical RPR-022
``AtomicPropagationPlan@1`` / ``PropagationPlanStep@1`` records are imported
and returned; this module does not redefine them.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationPlanStep,
    PropagationSCCGroup,
    TransformDisposition,
    obligation_set_identity,
)
from ..analysis.contract_repair_contracts import (
    DecisionDisposition,
    RepairTargetDecision,
)
from ..proof.formal_verification_contracts import content_identity
from ..proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from .support_behavior_placement import (
    SupportPlacementDecision,
    SupportPlacementDisposition,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

CHANGE_PROPAGATION_PLANNER_INTERFACE: Final[str] = "ChangePropagationPlanner@1"
PROPAGATION_PLAN_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-admission@1"
)
PLAN_EVIDENCE_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-evidence-bundle@1"
)
PLAN_PATH_SPAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-path-span@1"
)
PLAN_RESOURCE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-resource-bounds@1"
)
PLAN_VALIDATION_COMMAND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-validation-command@1"
)
PRODUCER_ID: Final[str] = "change-propagation-plan@1"
CONTRACT_VERSION: Final[int] = 1

MAX_PATHS: Final[int] = 1_024
MAX_SPANS: Final[int] = 2_048
MAX_COMMANDS: Final[int] = 256
MAX_PROOFS: Final[int] = 1_024
MAX_TRANSFORMS: Final[int] = 512
MAX_PLACEMENTS: Final[int] = 512
MAX_OBLIGATIONS: Final[int] = 1_024
MAX_STEPS: Final[int] = 512
MAX_SCC_GROUPS: Final[int] = 256
MAX_VALIDATION_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
DEFAULT_FIXED_POINT_ITERATIONS: Final[int] = 8
DEFAULT_CHECKPOINT_STRATEGY: Final[str] = "checkpoint:content-addressed"
DEFAULT_ROLLBACK_STRATEGY: Final[str] = "rollback:restore-checkpoint"
DEFAULT_FIXED_POINT_OBLIGATION: Final[str] = "fixed-point:reprove-closure"

_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "archive",
        "archives",
        "build",
        "dist",
        "node_modules",
        "third_party",
        "vendor",
        "vendors",
        ".git",
    }
)

# Dispositions that discharge a consumer without a migration step.
_NO_STEP_DISPOSITIONS: Final[frozenset[ConsumerDisposition]] = frozenset(
    {
        ConsumerDisposition.COMPATIBLE,
        ConsumerDisposition.EXCLUDED,
        ConsumerDisposition.UPSTREAM,
    }
)

# Dispositions that block automated admission when present on mandatory consumers.
_BLOCKING_DISPOSITIONS: Final[frozenset[ConsumerDisposition]] = frozenset(
    {
        ConsumerDisposition.ABSTAIN,
        ConsumerDisposition.FRONTIER,
        ConsumerDisposition.REVIEW_ONLY,
    }
)

# Dispositions that require covering write steps.
_MIGRATE_DISPOSITIONS: Final[frozenset[ConsumerDisposition]] = frozenset(
    {
        ConsumerDisposition.MIGRATE,
        ConsumerDisposition.ADAPTER,
    }
)


# ---------------------------------------------------------------------------
# Errors and closed rejection vocabulary
# ---------------------------------------------------------------------------


class ChangePropagationPlanError(ValueError):
    """Malformed plan evidence or an attempt to weaken a fail-closed boundary."""


class ChangePropagationPlanAuthorityError(ChangePropagationPlanError):
    """Root, path, proof, or write-authority mismatch."""


class PlanRejectionReason(str, Enum):
    """Closed, audit-stable abstention codes (never free-form model text)."""

    OMISSION = "omission"
    DUPLICATE = "duplicate"
    COMPETING_MAPPING = "competing_mapping"
    COMPETING_SITE = "competing_site"
    FAILED_PROOF = "failed_proof"
    UNRESOLVED_REQUIRED_FRONTIER = "unresolved_required_frontier"
    STALE_ROOTS = "stale_roots"
    FORBIDDEN_PATH = "forbidden_path"
    CROSS_ROOT_PATH = "cross_root_path"
    CYCLE_OUTSIDE_SCC = "cycle_outside_scc"
    INVALID_VALIDATION = "invalid_validation"
    EQUALLY_VALID_PLANS = "equally_valid_plans"
    INCOMPLETE_CLOSURE = "incomplete_closure"
    MISSING_TRANSFORM = "missing_transform"
    MISSING_PLACEMENT = "missing_placement"
    MISSING_WRITE_AUTHORITY = "missing_write_authority"
    ROOT_MISMATCH = "root_mismatch"
    UNCOVERED_MIGRATE = "uncovered_migrate"
    DUPLICATE_DISPOSITION = "duplicate_disposition"
    EMPTY_OBLIGATIONS = "empty_obligations"
    RESOURCE_BOUND = "resource_bound"
    MISSING_CHECKPOINT = "missing_checkpoint"
    MISSING_FIXED_POINT = "missing_fixed_point"
    NON_ADMITTED_TRANSFORM = "non_admitted_transform"
    NON_ADMITTED_PLACEMENT = "non_admitted_placement"
    STEP_DAG_INVALID = "step_dag_invalid"
    UNKNOWN_CONSUMER = "unknown_consumer"
    UNKNOWN_OBLIGATION = "unknown_obligation"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_VALIDATION_BYTES) -> str:
    if not isinstance(value, str):
        raise ChangePropagationPlanError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ChangePropagationPlanError(f"{name} is required")
    if len(result.encode("utf-8")) > limit:
        raise ChangePropagationPlanError(f"{name} exceeds its byte bound")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True, limit=MAX_VALIDATION_BYTES)
    if any(char.isspace() for char in result):
        raise ChangePropagationPlanError(f"{name} must be a compact identifier")
    return result


def _path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw in {".", ""}:
        raise ChangePropagationPlanAuthorityError(
            f"{name} must be a normalized repository-relative path"
        )
    if raw != candidate.as_posix():
        raise ChangePropagationPlanAuthorityError(
            f"{name} must be a normalized repository-relative path"
        )
    return raw


def _paths(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_PATHS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationPlanError(f"{name} must be a sequence of paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ChangePropagationPlanError(f"{name} must not be empty")
    if len(ordered) > limit:
        raise ChangePropagationPlanError(f"{name} exceeds path bound")
    return tuple(ordered)


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_PROOFS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationPlanError(f"{name} must be a sequence of identifiers")
    if preserve_order:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in values:
            ident = _identifier(item, name)
            if ident not in seen:
                seen.add(ident)
                ordered.append(ident)
        result = tuple(ordered)
    else:
        result = tuple(sorted({_identifier(item, name) for item in values}))
    if required and not result:
        raise ChangePropagationPlanError(f"{name} must not be empty")
    if len(result) > limit:
        raise ChangePropagationPlanError(f"{name} exceeds identifier bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ChangePropagationPlanError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChangePropagationPlanError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ChangePropagationPlanError(f"{name} exceeds its bound")
    return value


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ChangePropagationPlanError(f"unknown {name}: {value}") from exc
    raise ChangePropagationPlanError(f"{name} must be a {enum_cls.__name__}")


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**value)
        )
    raise ChangePropagationPlanError("roots must be PropagationAuthorityRoots")


def _path_forbidden(path: str) -> bool:
    parts = {part.casefold() for part in PurePosixPath(path).parts}
    return bool(parts & _FORBIDDEN_PATH_PARTS)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({item for item in values if item}))


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class PlanPathSpan:
    """Exact half-open byte span used as read or write authority."""

    path: str
    start: int
    end: int
    artifact_id: str = ""
    before_hash: str = ""
    schema: str = PLAN_PATH_SPAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "start", _nonneg_int(self.start, "start"))
        object.__setattr__(self, "end", _nonneg_int(self.end, "end"))
        if self.end < self.start:
            raise ChangePropagationPlanError("span end must be at or after span start")
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, "artifact_id", required=False)
        )
        object.__setattr__(
            self, "before_hash", _text(self.before_hash, "before_hash", required=False)
        )
        if self.schema != PLAN_PATH_SPAN_SCHEMA:
            raise ChangePropagationPlanError("unsupported plan path span schema")

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "path": self.path,
                "start": self.start,
                "end": self.end,
                "artifact_id": self.artifact_id,
                "before_hash": self.before_hash,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "artifact_id": self.artifact_id,
            "before_hash": self.before_hash,
            "content_id": self.content_id,
        }


@dataclass(frozen=True)
class PlanResourceBounds:
    """Hard bounds that an admitted plan must satisfy."""

    max_steps: int = MAX_STEPS
    max_scc_groups: int = MAX_SCC_GROUPS
    max_write_paths: int = MAX_PATHS
    max_read_paths: int = MAX_PATHS
    max_validation_commands: int = MAX_COMMANDS
    fixed_point_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS
    schema: str = PLAN_RESOURCE_BOUNDS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_steps", _nonneg_int(self.max_steps, "max_steps", maximum=MAX_STEPS)
        )
        object.__setattr__(
            self,
            "max_scc_groups",
            _nonneg_int(self.max_scc_groups, "max_scc_groups", maximum=MAX_SCC_GROUPS),
        )
        object.__setattr__(
            self,
            "max_write_paths",
            _nonneg_int(self.max_write_paths, "max_write_paths", maximum=MAX_PATHS),
        )
        object.__setattr__(
            self,
            "max_read_paths",
            _nonneg_int(self.max_read_paths, "max_read_paths", maximum=MAX_PATHS),
        )
        object.__setattr__(
            self,
            "max_validation_commands",
            _nonneg_int(
                self.max_validation_commands,
                "max_validation_commands",
                maximum=MAX_COMMANDS,
            ),
        )
        object.__setattr__(
            self,
            "fixed_point_iterations",
            _nonneg_int(self.fixed_point_iterations, "fixed_point_iterations", maximum=10_000),
        )
        if self.max_steps < 1 or self.fixed_point_iterations < 1:
            raise ChangePropagationPlanError(
                "max_steps and fixed_point_iterations must be at least 1"
            )
        if self.schema != PLAN_RESOURCE_BOUNDS_SCHEMA:
            raise ChangePropagationPlanError("unsupported plan resource bounds schema")

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "max_steps": self.max_steps,
                "max_scc_groups": self.max_scc_groups,
                "max_write_paths": self.max_write_paths,
                "max_read_paths": self.max_read_paths,
                "max_validation_commands": self.max_validation_commands,
                "fixed_point_iterations": self.fixed_point_iterations,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "max_steps": self.max_steps,
            "max_scc_groups": self.max_scc_groups,
            "max_write_paths": self.max_write_paths,
            "max_read_paths": self.max_read_paths,
            "max_validation_commands": self.max_validation_commands,
            "fixed_point_iterations": self.fixed_point_iterations,
            "content_id": self.content_id,
        }


@dataclass(frozen=True, order=True)
class PlanValidationCommand:
    """One focused validation command bound into the plan."""

    command_id: str
    argv: tuple[str, ...]
    required: bool = True
    schema: str = PLAN_VALIDATION_COMMAND_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _identifier(self.command_id, "command_id"))
        if (
            isinstance(self.argv, (str, bytes, bytearray))
            or not isinstance(self.argv, Sequence)
            or not self.argv
        ):
            raise ChangePropagationPlanError("argv must be a non-empty string sequence")
        argv = tuple(_text(item, "argv", required=True) for item in self.argv)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "required", _bool(self.required, "required"))
        if self.schema != PLAN_VALIDATION_COMMAND_SCHEMA:
            raise ChangePropagationPlanError("unsupported plan validation command schema")
        # Reject empty tokens and shell metacharacters that would make validation invalid.
        joined = " ".join(argv)
        if any(token == "" for token in argv):
            raise ChangePropagationPlanError("validation command tokens must be non-empty")
        if any(ch in joined for ch in (";", "|", "&", "`", "\n", "\r")):
            raise ChangePropagationPlanError("validation command contains forbidden shell metacharacters")

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "command_id": self.command_id,
                "argv": list(self.argv),
                "required": self.required,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "command_id": self.command_id,
            "argv": list(self.argv),
            "required": self.required,
            "content_id": self.content_id,
        }


@dataclass(frozen=True)
class PlanEvidenceBundle:
    """Complete admitted evidence for constructing one atomic plan.

    Paths and spans are the only sources of write authority.  Construction
    cannot invent paths, execute providers, or expand beyond this evidence.
    """

    roots: PropagationAuthorityRoots
    change_set_id: str
    delta_id: str
    impact_closure: ImpactClosureReceipt
    obligations: tuple[ConsumerMigrationObligation, ...]
    value_mapping_proofs: tuple[ValueMappingProof, ...] = ()
    analytical_transforms: tuple[AnalyticalTransform, ...] = ()
    placement_decisions: tuple[SupportPlacementDecision, ...] = ()
    repair_target_decisions: tuple[RepairTargetDecision, ...] = ()
    read_spans: tuple[PlanPathSpan, ...] = ()
    write_spans: tuple[PlanPathSpan, ...] = ()
    validation_commands: tuple[PlanValidationCommand, ...] = ()
    resource_bounds: PlanResourceBounds = PlanResourceBounds()
    checkpoint_strategy_ref: str = DEFAULT_CHECKPOINT_STRATEGY
    rollback_strategy_ref: str = DEFAULT_ROLLBACK_STRATEGY
    fixed_point_obligation_ref: str = DEFAULT_FIXED_POINT_OBLIGATION
    proof_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    candidate_set_id: str = ""
    # When set, must exactly match evidence roots (staleness check).
    expected_roots: PropagationAuthorityRoots | None = None
    schema: str = PLAN_EVIDENCE_BUNDLE_SCHEMA
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        if not isinstance(self.impact_closure, ImpactClosureReceipt):
            raise ChangePropagationPlanError(
                "impact_closure must be ImpactClosureReceipt@1"
            )
        if not isinstance(self.obligations, Sequence) or isinstance(
            self.obligations, (str, bytes, bytearray)
        ):
            raise ChangePropagationPlanError(
                "obligations must be a ConsumerMigrationObligation sequence"
            )
        obligations = tuple(self.obligations)
        if len(obligations) > MAX_OBLIGATIONS:
            raise ChangePropagationPlanError("obligations exceed bound")
        if not all(isinstance(item, ConsumerMigrationObligation) for item in obligations):
            raise ChangePropagationPlanError(
                "obligations must contain ConsumerMigrationObligation values"
            )
        object.__setattr__(self, "obligations", obligations)

        proofs = _typed_tuple(
            self.value_mapping_proofs,
            ValueMappingProof,
            "value_mapping_proofs",
            limit=MAX_PROOFS,
        )
        object.__setattr__(self, "value_mapping_proofs", proofs)

        transforms = _typed_tuple(
            self.analytical_transforms,
            AnalyticalTransform,
            "analytical_transforms",
            limit=MAX_TRANSFORMS,
        )
        object.__setattr__(self, "analytical_transforms", transforms)

        placements = _typed_tuple(
            self.placement_decisions,
            SupportPlacementDecision,
            "placement_decisions",
            limit=MAX_PLACEMENTS,
        )
        object.__setattr__(self, "placement_decisions", placements)

        decisions = _typed_tuple(
            self.repair_target_decisions,
            RepairTargetDecision,
            "repair_target_decisions",
            limit=MAX_PLACEMENTS,
        )
        object.__setattr__(self, "repair_target_decisions", decisions)

        read_spans = _typed_tuple(
            self.read_spans, PlanPathSpan, "read_spans", limit=MAX_SPANS
        )
        write_spans = _typed_tuple(
            self.write_spans, PlanPathSpan, "write_spans", limit=MAX_SPANS
        )
        object.__setattr__(self, "read_spans", read_spans)
        object.__setattr__(self, "write_spans", write_spans)

        commands = _typed_tuple(
            self.validation_commands,
            PlanValidationCommand,
            "validation_commands",
            limit=MAX_COMMANDS,
        )
        object.__setattr__(self, "validation_commands", commands)

        if not isinstance(self.resource_bounds, PlanResourceBounds):
            raise ChangePropagationPlanError(
                "resource_bounds must be PlanResourceBounds"
            )
        object.__setattr__(
            self,
            "checkpoint_strategy_ref",
            _identifier(self.checkpoint_strategy_ref, "checkpoint_strategy_ref"),
        )
        object.__setattr__(
            self,
            "rollback_strategy_ref",
            _identifier(self.rollback_strategy_ref, "rollback_strategy_ref"),
        )
        object.__setattr__(
            self,
            "fixed_point_obligation_ref",
            _identifier(self.fixed_point_obligation_ref, "fixed_point_obligation_ref"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs"),
        )
        object.__setattr__(
            self,
            "candidate_set_id",
            _text(self.candidate_set_id, "candidate_set_id", required=False),
        )
        if self.expected_roots is not None:
            object.__setattr__(self, "expected_roots", _roots(self.expected_roots))
        if self.schema != PLAN_EVIDENCE_BUNDLE_SCHEMA:
            raise ChangePropagationPlanError("unsupported plan evidence bundle schema")
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "roots": self.roots.content_id,
            "change_set_id": self.change_set_id,
            "delta_id": self.delta_id,
            "impact_closure_id": self.impact_closure.content_id,
            "obligation_ids": [
                item.content_id
                for item in sorted(self.obligations, key=lambda o: o.obligation_id)
            ],
            "value_mapping_proof_ids": sorted(
                item.proof_id for item in self.value_mapping_proofs
            ),
            "analytical_transform_ids": sorted(
                item.content_id for item in self.analytical_transforms
            ),
            "placement_decision_ids": sorted(
                item.content_id for item in self.placement_decisions
            ),
            "repair_target_decision_ids": sorted(
                item.content_id for item in self.repair_target_decisions
            ),
            "read_span_ids": sorted(item.content_id for item in self.read_spans),
            "write_span_ids": sorted(item.content_id for item in self.write_spans),
            "validation_command_ids": sorted(
                item.content_id for item in self.validation_commands
            ),
            "resource_bounds": self.resource_bounds.content_id,
            "checkpoint_strategy_ref": self.checkpoint_strategy_ref,
            "rollback_strategy_ref": self.rollback_strategy_ref,
            "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            "proof_refs": list(self.proof_refs),
            "invalidation_refs": list(self.invalidation_refs),
            "candidate_set_id": self.candidate_set_id,
            "expected_roots": (
                None if self.expected_roots is None else self.expected_roots.content_id
            ),
            "producer_id": self.producer_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "content_id": self.content_id,
        }


def _typed_tuple(
    values: Sequence[Any],
    cls: type,
    name: str,
    *,
    limit: int,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationPlanError(f"{name} must be a sequence")
    if len(values) > limit:
        raise ChangePropagationPlanError(f"{name} exceeds bound")
    if not all(isinstance(item, cls) for item in values):
        raise ChangePropagationPlanError(f"{name} must contain {cls.__name__} values")
    return tuple(values)


# ---------------------------------------------------------------------------
# Admission result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationPlanAdmission:
    """Deterministic plan admission result (always carries a plan record)."""

    disposition: PlanDisposition
    plan: AtomicPropagationPlan
    evidence_bundle_id: str
    reason_codes: tuple[str, ...] = ()
    alternative_plan_ids: tuple[str, ...] = ()
    step_order: tuple[str, ...] = ()
    scc_group_ids: tuple[str, ...] = ()
    permitted_read_spans: tuple[PlanPathSpan, ...] = ()
    permitted_write_spans: tuple[PlanPathSpan, ...] = ()
    validation_command_ids: tuple[str, ...] = ()
    resource_bounds_ref: str = ""
    producer_id: str = PRODUCER_ID
    schema: str = PROPAGATION_PLAN_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PlanDisposition, "disposition"),
        )
        if not isinstance(self.plan, AtomicPropagationPlan):
            raise ChangePropagationPlanError(
                "plan must be the canonical AtomicPropagationPlan@1"
            )
        object.__setattr__(
            self,
            "evidence_bundle_id",
            _identifier(self.evidence_bundle_id, "evidence_bundle_id"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", preserve_order=False),
        )
        object.__setattr__(
            self,
            "alternative_plan_ids",
            _ids(self.alternative_plan_ids, "alternative_plan_ids"),
        )
        object.__setattr__(
            self,
            "step_order",
            _ids(self.step_order, "step_order", preserve_order=True, limit=MAX_STEPS),
        )
        object.__setattr__(
            self,
            "scc_group_ids",
            _ids(self.scc_group_ids, "scc_group_ids", preserve_order=True, limit=MAX_SCC_GROUPS),
        )
        read_spans = _typed_tuple(
            self.permitted_read_spans, PlanPathSpan, "permitted_read_spans", limit=MAX_SPANS
        )
        write_spans = _typed_tuple(
            self.permitted_write_spans,
            PlanPathSpan,
            "permitted_write_spans",
            limit=MAX_SPANS,
        )
        object.__setattr__(self, "permitted_read_spans", read_spans)
        object.__setattr__(self, "permitted_write_spans", write_spans)
        object.__setattr__(
            self,
            "validation_command_ids",
            _ids(self.validation_command_ids, "validation_command_ids"),
        )
        object.__setattr__(
            self,
            "resource_bounds_ref",
            _text(self.resource_bounds_ref, "resource_bounds_ref", required=False),
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        if self.schema != PROPAGATION_PLAN_ADMISSION_SCHEMA:
            raise ChangePropagationPlanError("unsupported plan admission schema")

        if self.disposition is PlanDisposition.ADMITTED:
            if self.plan.disposition is not PlanDisposition.ADMITTED:
                raise ChangePropagationPlanError(
                    "admitted admission requires an admitted AtomicPropagationPlan"
                )
            if self.reason_codes:
                raise ChangePropagationPlanError(
                    "admitted admissions cannot carry rejection reason codes"
                )
            if self.alternative_plan_ids:
                raise ChangePropagationPlanError(
                    "admitted admissions cannot retain alternative plan identities"
                )
            if not self.plan.steps:
                raise ChangePropagationPlanError("admitted admissions require plan steps")
        else:
            if self.plan.disposition is PlanDisposition.ADMITTED:
                raise ChangePropagationPlanError(
                    "non-admitted admissions cannot carry an admitted plan"
                )
            if self.plan.permitted_write_paths:
                raise ChangePropagationPlanAuthorityError(
                    "non-admitted admissions cannot grant write path authority"
                )
            if self.permitted_write_spans:
                raise ChangePropagationPlanAuthorityError(
                    "non-admitted admissions cannot grant write spans"
                )
            if not self.reason_codes:
                raise ChangePropagationPlanError(
                    "non-admitted admissions require reason codes"
                )

    @property
    def admitted(self) -> bool:
        return self.disposition is PlanDisposition.ADMITTED

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "disposition": self.disposition.value,
                "plan_id": self.plan.content_id,
                "evidence_bundle_id": self.evidence_bundle_id,
                "reason_codes": list(self.reason_codes),
                "alternative_plan_ids": list(self.alternative_plan_ids),
                "step_order": list(self.step_order),
                "scc_group_ids": list(self.scc_group_ids),
                "permitted_read_span_ids": [item.content_id for item in self.permitted_read_spans],
                "permitted_write_span_ids": [
                    item.content_id for item in self.permitted_write_spans
                ],
                "validation_command_ids": list(self.validation_command_ids),
                "resource_bounds_ref": self.resource_bounds_ref,
                "producer_id": self.producer_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "disposition": self.disposition.value,
            "plan": self.plan.to_record(),
            "evidence_bundle_id": self.evidence_bundle_id,
            "reason_codes": list(self.reason_codes),
            "alternative_plan_ids": list(self.alternative_plan_ids),
            "step_order": list(self.step_order),
            "scc_group_ids": list(self.scc_group_ids),
            "permitted_read_spans": [item.to_dict() for item in self.permitted_read_spans],
            "permitted_write_spans": [item.to_dict() for item in self.permitted_write_spans],
            "validation_command_ids": list(self.validation_command_ids),
            "resource_bounds_ref": self.resource_bounds_ref,
            "producer_id": self.producer_id,
            "content_id": self.content_id,
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class ChangePropagationPlanner:
    """Build and admit one deterministic atomic transitive repair plan."""

    INTERFACE: ClassVar[str] = CHANGE_PROPAGATION_PLANNER_INTERFACE

    def admit(self, evidence: PlanEvidenceBundle) -> PropagationPlanAdmission:
        """Admit one plan from complete evidence, or abstain fail-closed."""

        if not isinstance(evidence, PlanEvidenceBundle):
            raise ChangePropagationPlanError("evidence must be PlanEvidenceBundle")

        reasons: list[str] = []
        alternative_ids: list[str] = []

        # --- root / staleness -------------------------------------------------
        roots = evidence.roots
        if evidence.expected_roots is not None and evidence.expected_roots != roots:
            reasons.append(PlanRejectionReason.STALE_ROOTS.value)
        if evidence.impact_closure.roots != roots:
            reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
        for obligation in evidence.obligations:
            if obligation.roots != roots:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
                break
        for transform in evidence.analytical_transforms:
            if transform.roots != roots:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
                break
        for placement in evidence.placement_decisions:
            if placement.roots != roots:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
                break

        # --- impact closure completeness -------------------------------------
        closure = evidence.impact_closure
        if closure.delta_id != evidence.delta_id:
            reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
        if closure.completeness is ImpactCompleteness.ABSTAINED:
            reasons.append(PlanRejectionReason.INCOMPLETE_CLOSURE.value)
        if closure.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            # Required frontier remaining → abstain (review-only is not automated).
            if closure.frontier_node_ids or closure.frontier_edge_ids:
                reasons.append(PlanRejectionReason.UNRESOLVED_REQUIRED_FRONTIER.value)
            else:
                reasons.append(PlanRejectionReason.INCOMPLETE_CLOSURE.value)
        if closure.completeness is not ImpactCompleteness.COMPLETE:
            if PlanRejectionReason.INCOMPLETE_CLOSURE.value not in reasons and (
                PlanRejectionReason.UNRESOLVED_REQUIRED_FRONTIER.value not in reasons
            ):
                reasons.append(PlanRejectionReason.INCOMPLETE_CLOSURE.value)

        # --- obligations: complete, unique per consumer ----------------------
        if not evidence.obligations:
            reasons.append(PlanRejectionReason.EMPTY_OBLIGATIONS.value)

        consumer_to_obligations: dict[str, list[ConsumerMigrationObligation]] = defaultdict(
            list
        )
        obligation_by_id: dict[str, ConsumerMigrationObligation] = {}
        for obligation in evidence.obligations:
            if obligation.delta_id != evidence.delta_id:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
            if obligation.obligation_id in obligation_by_id:
                reasons.append(PlanRejectionReason.DUPLICATE.value)
            obligation_by_id[obligation.obligation_id] = obligation
            consumer_to_obligations[obligation.consumer_id].append(obligation)

        for consumer_id, rows in consumer_to_obligations.items():
            if len(rows) > 1:
                reasons.append(PlanRejectionReason.DUPLICATE_DISPOSITION.value)

        closure_consumers = {item.consumer_id: item for item in closure.consumers}
        obligation_consumers = set(consumer_to_obligations)
        # Omissions: every closure consumer must appear exactly once.
        missing_consumers = set(closure_consumers) - obligation_consumers
        extra_consumers = obligation_consumers - set(closure_consumers)
        if missing_consumers:
            reasons.append(PlanRejectionReason.OMISSION.value)
        if extra_consumers:
            reasons.append(PlanRejectionReason.UNKNOWN_CONSUMER.value)

        for consumer_id, obligation_list in consumer_to_obligations.items():
            obligation = obligation_list[0]
            impact = closure_consumers.get(consumer_id)
            if impact is None:
                continue
            if impact.mandatory and obligation.disposition in _BLOCKING_DISPOSITIONS:
                if obligation.disposition is ConsumerDisposition.FRONTIER:
                    reasons.append(PlanRejectionReason.UNRESOLVED_REQUIRED_FRONTIER.value)
                else:
                    reasons.append(PlanRejectionReason.FAILED_PROOF.value)

        # --- value mapping proofs --------------------------------------------
        proofs_by_requirement: dict[str, list[ValueMappingProof]] = defaultdict(list)
        for proof in evidence.value_mapping_proofs:
            if proof.repository_id and proof.repository_id != roots.repository_id:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
            if proof.tree_id and proof.tree_id not in {
                roots.base_tree_id,
                roots.candidate_tree_id,
            }:
                reasons.append(PlanRejectionReason.STALE_ROOTS.value)
            if proof.toolchain_id and proof.toolchain_id != roots.toolchain_id:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
            if proof.policy_id and proof.policy_id != roots.policy_id:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
            proofs_by_requirement[proof.requirement_id].append(proof)

        # Group proofs by (requirement_id, consumer_id) so two consumers that
        # share a requirement name do not look like a competing mapping set.
        proofs_by_req_consumer: dict[tuple[str, str], list[ValueMappingProof]] = (
            defaultdict(list)
        )
        for proof in evidence.value_mapping_proofs:
            proofs_by_req_consumer[(proof.requirement_id, proof.consumer_id)].append(
                proof
            )

        for (_requirement_id, _consumer_id), rows in proofs_by_req_consumer.items():
            if len(rows) > 1:
                proved_ids = {
                    cid for item in rows for cid in item.proved_candidate_ids
                }
                if len(proved_ids) > 1:
                    reasons.append(PlanRejectionReason.COMPETING_MAPPING.value)
                else:
                    reasons.append(PlanRejectionReason.DUPLICATE.value)
            for proof in rows:
                if proof.disposition is SynthesisDisposition.AMBIGUOUS:
                    reasons.append(PlanRejectionReason.COMPETING_MAPPING.value)
                elif proof.disposition is SynthesisDisposition.REFUTED:
                    reasons.append(PlanRejectionReason.FAILED_PROOF.value)
                elif proof.disposition in {
                    SynthesisDisposition.UNKNOWN,
                    SynthesisDisposition.TIMEOUT,
                    SynthesisDisposition.UNSUPPORTED,
                }:
                    # Only fail when a migrate obligation depends on this requirement.
                    pass

        for obligation in evidence.obligations:
            if obligation.disposition not in _MIGRATE_DISPOSITIONS:
                continue
            for missing_id in obligation.missing_input_ids:
                rows = proofs_by_requirement.get(missing_id, [])
                if not rows:
                    reasons.append(PlanRejectionReason.FAILED_PROOF.value)
                    continue
                if not any(
                    item.disposition is SynthesisDisposition.UNIQUE_PROVED
                    and len(item.proved_candidate_ids) == 1
                    for item in rows
                ):
                    if any(
                        item.disposition is SynthesisDisposition.AMBIGUOUS for item in rows
                    ):
                        reasons.append(PlanRejectionReason.COMPETING_MAPPING.value)
                    else:
                        reasons.append(PlanRejectionReason.FAILED_PROOF.value)

        # --- analytical transforms -------------------------------------------
        transforms_by_obligation: dict[str, list[AnalyticalTransform]] = defaultdict(list)
        admitted_transforms: list[AnalyticalTransform] = []
        transform_ids: set[str] = set()
        for transform in evidence.analytical_transforms:
            if transform.transform_id in transform_ids:
                reasons.append(PlanRejectionReason.DUPLICATE.value)
            transform_ids.add(transform.transform_id)
            if transform.disposition is TransformDisposition.ADMITTED:
                admitted_transforms.append(transform)
                for oid in transform.obligation_ids:
                    transforms_by_obligation[oid].append(transform)
            elif transform.disposition is TransformDisposition.REJECTED:
                # Rejected transforms must not cover migrate obligations.
                for oid in transform.obligation_ids:
                    if oid in obligation_by_id and obligation_by_id[
                        oid
                    ].disposition in _MIGRATE_DISPOSITIONS:
                        reasons.append(PlanRejectionReason.NON_ADMITTED_TRANSFORM.value)

        # Competing admitted transforms for the same obligation (distinct content).
        for oid, rows in transforms_by_obligation.items():
            if len(rows) > 1:
                identities = {item.content_id for item in rows}
                path_sets = {tuple(item.target_paths) for item in rows}
                if len(identities) > 1 and len(path_sets) > 1:
                    reasons.append(PlanRejectionReason.EQUALLY_VALID_PLANS.value)
                elif len(identities) > 1:
                    # Same paths but different transforms → competing mapping.
                    reasons.append(PlanRejectionReason.COMPETING_MAPPING.value)

        # --- placement decisions ---------------------------------------------
        placements_by_behavior: dict[str, list[SupportPlacementDecision]] = defaultdict(
            list
        )
        admitted_placements: list[SupportPlacementDecision] = []
        for placement in evidence.placement_decisions:
            if placement.disposition is SupportPlacementDisposition.ADMITTED:
                admitted_placements.append(placement)
                placements_by_behavior[placement.behavior_id].append(placement)
            elif placement.disposition in {
                SupportPlacementDisposition.AMBIGUOUS,
            }:
                reasons.append(PlanRejectionReason.COMPETING_SITE.value)

        for behavior_id, rows in placements_by_behavior.items():
            if len(rows) > 1:
                paths = {item.target_path for item in rows}
                if len(paths) > 1:
                    reasons.append(PlanRejectionReason.COMPETING_SITE.value)
                elif len({item.content_id for item in rows}) > 1:
                    reasons.append(PlanRejectionReason.EQUALLY_VALID_PLANS.value)

        for obligation in evidence.obligations:
            if obligation.disposition not in _MIGRATE_DISPOSITIONS:
                continue
            for behavior_id in obligation.behavior_contract_ids:
                rows = placements_by_behavior.get(behavior_id, [])
                if not rows:
                    # Behavior may be covered by an analytical transform only.
                    covered_by_transform = any(
                        t.disposition is TransformDisposition.ADMITTED
                        and obligation.obligation_id in t.obligation_ids
                        for t in evidence.analytical_transforms
                    )
                    if not covered_by_transform:
                        reasons.append(PlanRejectionReason.MISSING_PLACEMENT.value)

        # --- repair target decisions (optional path authority) ---------------
        decision_write_paths: set[str] = set()
        for decision in evidence.repair_target_decisions:
            if decision.disposition is DecisionDisposition.ADMITTED:
                decision_write_paths.update(decision.permitted_write_paths)
            # Stale tree binding: AuthorityRoots.tree_id must match candidate tree.
            if decision.roots.repository_id != roots.repository_id:
                reasons.append(PlanRejectionReason.ROOT_MISMATCH.value)
            if decision.roots.tree_id not in {
                roots.base_tree_id,
                roots.candidate_tree_id,
            }:
                reasons.append(PlanRejectionReason.STALE_ROOTS.value)

        # --- write / read authority ------------------------------------------
        # Writes derive only from admitted transforms, placements, and repair
        # target decisions.  Explicit write_spans further constrain authority
        # when present (derived paths must be a subset of span paths).
        transform_write_paths = {
            path
            for transform in admitted_transforms
            for path in transform.target_paths
        }
        placement_write_paths = {
            path
            for placement in admitted_placements
            for path in placement.placement_paths
        }
        span_write_paths = {span.path for span in evidence.write_spans}
        span_read_paths = {span.path for span in evidence.read_spans}
        derived_write_paths = (
            transform_write_paths | placement_write_paths | decision_write_paths
        )
        if span_write_paths:
            if not derived_write_paths.issubset(span_write_paths):
                reasons.append(PlanRejectionReason.MISSING_WRITE_AUTHORITY.value)
            authorized_write_paths = frozenset(
                path for path in derived_write_paths if path in span_write_paths
            )
        else:
            authorized_write_paths = frozenset(derived_write_paths)

        authorized_read_paths = frozenset(
            span_read_paths
            | authorized_write_paths
            | transform_write_paths
            | {obl.node.path for obl in evidence.obligations if obl.node.path}
        )

        for path in authorized_write_paths | authorized_read_paths:
            if _path_forbidden(path):
                reasons.append(PlanRejectionReason.FORBIDDEN_PATH.value)
            # Cross-root / escape already blocked by _path normalization; also ban
            # absolute-like or parent traversal residues.
            if path.startswith("/") or ".." in PurePosixPath(path).parts:
                reasons.append(PlanRejectionReason.CROSS_ROOT_PATH.value)

        # --- validation commands ---------------------------------------------
        if not evidence.validation_commands:
            reasons.append(PlanRejectionReason.INVALID_VALIDATION.value)
        command_ids = [item.command_id for item in evidence.validation_commands]
        if len(command_ids) != len(set(command_ids)):
            reasons.append(PlanRejectionReason.DUPLICATE.value)
        validation_refs = tuple(
            sorted({item.content_id for item in evidence.validation_commands})
        )
        if len(evidence.validation_commands) > evidence.resource_bounds.max_validation_commands:
            reasons.append(PlanRejectionReason.RESOURCE_BOUND.value)

        # --- checkpoint / fixed-point ----------------------------------------
        if not evidence.checkpoint_strategy_ref or not evidence.rollback_strategy_ref:
            reasons.append(PlanRejectionReason.MISSING_CHECKPOINT.value)
        if not evidence.fixed_point_obligation_ref:
            reasons.append(PlanRejectionReason.MISSING_FIXED_POINT.value)

        # --- build steps (even when abstaining, for diagnostics without writes) --
        steps, scc_groups, step_reasons, step_alternatives = self._build_steps(
            evidence=evidence,
            obligation_by_id=obligation_by_id,
            admitted_transforms=admitted_transforms,
            admitted_placements=admitted_placements,
            authorized_write_paths=authorized_write_paths,
            authorized_read_paths=authorized_read_paths,
            validation_refs=validation_refs,
            closure=closure,
        )
        reasons.extend(step_reasons)
        alternative_ids.extend(step_alternatives)

        # Resource bounds on constructed steps.
        if len(steps) > evidence.resource_bounds.max_steps:
            reasons.append(PlanRejectionReason.RESOURCE_BOUND.value)
        if len(scc_groups) > evidence.resource_bounds.max_scc_groups:
            reasons.append(PlanRejectionReason.RESOURCE_BOUND.value)
        if len(authorized_write_paths) > evidence.resource_bounds.max_write_paths:
            reasons.append(PlanRejectionReason.RESOURCE_BOUND.value)
        if len(authorized_read_paths) > evidence.resource_bounds.max_read_paths:
            reasons.append(PlanRejectionReason.RESOURCE_BOUND.value)

        # Migrate obligations must be covered when we would admit.
        migrate_ids = {
            item.obligation_id
            for item in evidence.obligations
            if item.disposition in _MIGRATE_DISPOSITIONS
        }
        covered = {oid for step in steps for oid in step.obligation_ids}
        if migrate_ids - covered:
            reasons.append(PlanRejectionReason.UNCOVERED_MIGRATE.value)
            # Also missing transform when no transform/placement produced a step.
            for oid in migrate_ids - covered:
                if oid not in transforms_by_obligation:
                    reasons.append(PlanRejectionReason.MISSING_TRANSFORM.value)

        # AtomicPropagationPlan@1 requires steps and write authority on admit.
        if not steps:
            reasons.append(PlanRejectionReason.OMISSION.value)
        if not authorized_write_paths:
            reasons.append(PlanRejectionReason.MISSING_WRITE_AUTHORITY.value)

        # Proof refs for the plan.
        plan_proof_refs = _sorted_unique(
            list(evidence.proof_refs)
            + [
                ref
                for transform in admitted_transforms
                for ref in transform.proof_refs
            ]
            + [
                ref
                for placement in admitted_placements
                for ref in placement.proof_receipt_ids
            ]
            + [
                item.proof_id
                for item in evidence.value_mapping_proofs
                if item.disposition is SynthesisDisposition.UNIQUE_PROVED
            ]
            + [
                ref
                for obligation in evidence.obligations
                for ref in obligation.proof_refs
            ]
        )
        if not plan_proof_refs and migrate_ids:
            reasons.append(PlanRejectionReason.FAILED_PROOF.value)

        invalidation_refs = _sorted_unique(
            list(evidence.invalidation_refs)
            + [roots.candidate_tree_id, roots.candidate_overlay_id, closure.content_id]
        )

        # Deduplicate reasons while preserving stable sorted order for identity.
        unique_reasons = tuple(sorted(set(reasons)))
        unique_alternatives = tuple(sorted(set(alternative_ids)))

        # Obligation set identity (requires nonempty unique obligations).
        try:
            set_id = (
                obligation_set_identity(evidence.obligations)
                if evidence.obligations
                else content_identity(
                    {
                        "schema": "empty-obligation-set@1",
                        "delta_id": evidence.delta_id,
                    }
                )
            )
        except Exception:
            unique_reasons = tuple(
                sorted(set(unique_reasons) | {PlanRejectionReason.DUPLICATE.value})
            )
            set_id = content_identity(
                {
                    "schema": "invalid-obligation-set@1",
                    "delta_id": evidence.delta_id,
                }
            )

        plan_id = content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-identity@1",
                "roots": roots.content_id,
                "change_set_id": evidence.change_set_id,
                "delta_id": evidence.delta_id,
                "impact_closure_id": closure.content_id,
                "obligation_set_id": set_id,
                "transform_ids": sorted(t.content_id for t in admitted_transforms),
                "placement_ids": sorted(p.content_id for p in admitted_placements),
                "proof_ids": list(plan_proof_refs),
                "candidate_set_id": evidence.candidate_set_id,
                "graph_id": roots.graph_id,
                "index_id": roots.index_id,
                "model_id": roots.model_id,
                "translator_id": roots.translator_id,
                "toolchain_id": roots.toolchain_id,
                "policy_id": roots.policy_id,
                "validation_refs": list(validation_refs),
                "resource_bounds": evidence.resource_bounds.content_id,
                "reasons": list(unique_reasons),
            }
        )

        if unique_reasons:
            # Prefer a contract-valid obligation set.  Duplicate consumers or an
            # empty set cannot be embedded in AtomicPropagationPlan@1, so fall
            # back to a single diagnostic obligation for the abstention record.
            try:
                abstain_obligations = tuple(evidence.obligations)
                abstain_set_id = (
                    obligation_set_identity(abstain_obligations)
                    if abstain_obligations
                    else ""
                )
                # Probe uniqueness invariant (one obligation per consumer).
                if len({item.consumer_id for item in abstain_obligations}) != len(
                    abstain_obligations
                ):
                    raise ChangePropagationPlanError("duplicate consumers")
                if not abstain_obligations:
                    raise ChangePropagationPlanError("empty obligations")
            except Exception:
                diagnostic = _diagnostic_obligation(roots, evidence.delta_id)
                abstain_obligations = (diagnostic,)
                abstain_set_id = obligation_set_identity(abstain_obligations)

            abstained_plan = AtomicPropagationPlan(
                roots=roots,
                plan_id=plan_id,
                change_set_id=evidence.change_set_id,
                delta_id=evidence.delta_id,
                impact_closure_id=closure.content_id,
                disposition=PlanDisposition.ABSTAINED,
                obligations=abstain_obligations,
                obligation_set_id=abstain_set_id,
                steps=(),
                scc_groups=(),
                permitted_read_paths=(),
                permitted_write_paths=(),
                checkpoint_strategy_ref="",
                rollback_strategy_ref="",
                fixed_point_obligation_ref="",
                proof_refs=(),
                invalidation_refs=invalidation_refs or (roots.candidate_tree_id,),
            )
            return PropagationPlanAdmission(
                disposition=PlanDisposition.ABSTAINED,
                plan=abstained_plan,
                evidence_bundle_id=evidence.content_id,
                reason_codes=unique_reasons,
                alternative_plan_ids=unique_alternatives,
                step_order=(),
                scc_group_ids=(),
                permitted_read_spans=(),
                permitted_write_spans=(),
                validation_command_ids=(),
                resource_bounds_ref=evidence.resource_bounds.content_id,
            )

        # Admitted plan.
        ordered_steps = tuple(steps)
        step_order = tuple(step.step_id for step in ordered_steps)
        scc_group_ids = tuple(group.group_id for group in scc_groups)

        read_paths = _paths(sorted(authorized_read_paths), "permitted_read_paths")
        write_paths = _paths(sorted(authorized_write_paths), "permitted_write_paths", required=True)

        admitted_plan = AtomicPropagationPlan(
            roots=roots,
            plan_id=plan_id,
            change_set_id=evidence.change_set_id,
            delta_id=evidence.delta_id,
            impact_closure_id=closure.content_id,
            disposition=PlanDisposition.ADMITTED,
            obligations=tuple(
                sorted(evidence.obligations, key=lambda item: item.obligation_id)
            ),
            obligation_set_id=set_id,
            steps=ordered_steps,
            scc_groups=tuple(scc_groups),
            permitted_read_paths=read_paths,
            permitted_write_paths=write_paths,
            checkpoint_strategy_ref=evidence.checkpoint_strategy_ref,
            rollback_strategy_ref=evidence.rollback_strategy_ref,
            fixed_point_obligation_ref=evidence.fixed_point_obligation_ref,
            proof_refs=plan_proof_refs,
            invalidation_refs=invalidation_refs,
        )

        # Exact spans for admission audit: prefer evidence spans, else path-wide.
        read_spans = evidence.read_spans or tuple(
            PlanPathSpan(path=path, start=0, end=0, artifact_id=f"path:{path}")
            for path in read_paths
        )
        write_spans = evidence.write_spans or tuple(
            PlanPathSpan(path=path, start=0, end=0, artifact_id=f"path:{path}")
            for path in write_paths
        )
        # Filter write spans to authorized paths only.
        write_spans = tuple(
            span for span in write_spans if span.path in authorized_write_paths
        )
        read_spans = tuple(
            span for span in read_spans if span.path in authorized_read_paths
        )

        return PropagationPlanAdmission(
            disposition=PlanDisposition.ADMITTED,
            plan=admitted_plan,
            evidence_bundle_id=evidence.content_id,
            reason_codes=(),
            alternative_plan_ids=(),
            step_order=step_order,
            scc_group_ids=scc_group_ids,
            permitted_read_spans=read_spans,
            permitted_write_spans=write_spans,
            validation_command_ids=tuple(
                sorted(item.command_id for item in evidence.validation_commands)
            ),
            resource_bounds_ref=evidence.resource_bounds.content_id,
        )

    # ------------------------------------------------------------------
    # Step / SCC construction
    # ------------------------------------------------------------------

    def _build_steps(
        self,
        *,
        evidence: PlanEvidenceBundle,
        obligation_by_id: Mapping[str, ConsumerMigrationObligation],
        admitted_transforms: Sequence[AnalyticalTransform],
        admitted_placements: Sequence[SupportPlacementDecision],
        authorized_write_paths: frozenset[str],
        authorized_read_paths: frozenset[str],
        validation_refs: tuple[str, ...],
        closure: ImpactClosureReceipt,
    ) -> tuple[
        list[PropagationPlanStep],
        list[PropagationSCCGroup],
        list[str],
        list[str],
    ]:
        reasons: list[str] = []
        alternatives: list[str] = []

        # Map consumer → scc_id from closure.
        consumer_to_scc: dict[str, str] = {}
        for scc in closure.sccs:
            for member in scc.member_consumer_ids:
                consumer_to_scc[member] = scc.scc_id

        # Build analytical steps from admitted transforms (deterministic order).
        transforms_sorted = sorted(
            admitted_transforms,
            key=lambda item: (item.transform_id, item.content_id),
        )
        transform_id_to_step: dict[str, str] = {}
        pending_steps: dict[str, dict[str, Any]] = {}

        for transform in transforms_sorted:
            unknown = set(transform.obligation_ids) - set(obligation_by_id)
            if unknown:
                reasons.append(PlanRejectionReason.UNKNOWN_OBLIGATION.value)
                continue
            write_paths = tuple(
                path for path in transform.target_paths if path in authorized_write_paths
            )
            if transform.target_paths and not write_paths:
                reasons.append(PlanRejectionReason.MISSING_WRITE_AUTHORITY.value)
            # Determine SCC group from obligation consumers.
            scc_ids = {
                consumer_to_scc.get(obligation_by_id[oid].consumer_id, "")
                for oid in transform.obligation_ids
                if oid in obligation_by_id
            }
            scc_ids.discard("")
            if len(scc_ids) > 1:
                # One step cannot straddle multiple SCCs without an explicit group.
                # Treat multi-SCC transform as still valid if all members share one
                # multi-member group; otherwise flag.
                reasons.append(PlanRejectionReason.CYCLE_OUTSIDE_SCC.value)
            scc_id = next(iter(scc_ids), "")
            group_id = f"group:{scc_id}" if scc_id else ""
            step_id = f"step:analytical:{transform.transform_id}"
            transform_id_to_step[transform.transform_id] = step_id
            read_paths = _paths(
                sorted(
                    set(transform.target_paths)
                    | {
                        obligation_by_id[oid].node.path
                        for oid in transform.obligation_ids
                        if oid in obligation_by_id
                    }
                    & set(authorized_read_paths)
                ),
                "read_paths",
            )
            pending_steps[step_id] = {
                "step_id": step_id,
                "kind": PlanStepKind.ANALYTICAL,
                "obligation_ids": tuple(sorted(transform.obligation_ids)),
                "dependency_transform_ids": tuple(
                    sorted(transform.dependency_transform_ids)
                ),
                "transform_id": transform.transform_id,
                "read_paths": read_paths,
                "write_paths": _paths(sorted(write_paths), "write_paths") if write_paths else (),
                "precondition_refs": tuple(
                    sorted(
                        {
                            f"pre:obligation:{oid}"
                            for oid in transform.obligation_ids
                        }
                        | {f"pre:transform:{transform.transform_id}"}
                    )
                ),
                "postcondition_refs": tuple(
                    sorted(
                        {
                            f"post:transform:{transform.transform_id}",
                            f"post:kind:{transform.kind.value}",
                        }
                    )
                ),
                "validation_refs": validation_refs,
                "scc_group_id": group_id,
                "scc_id": scc_id,
                "consumer_ids": tuple(
                    sorted(
                        {
                            obligation_by_id[oid].consumer_id
                            for oid in transform.obligation_ids
                            if oid in obligation_by_id
                        }
                    )
                ),
            }

        # LLM-bounded steps for placements not covered by analytical transforms.
        covered_obligations = {
            oid
            for step in pending_steps.values()
            for oid in step["obligation_ids"]
        }
        placements_sorted = sorted(
            admitted_placements,
            key=lambda item: (item.behavior_id, item.selected_candidate_id, item.content_id),
        )
        for placement in placements_sorted:
            # Find migrate obligations that list this behavior and lack a step.
            target_obligations = [
                obl
                for obl in evidence.obligations
                if placement.behavior_id in obl.behavior_contract_ids
                and obl.disposition in _MIGRATE_DISPOSITIONS
                and obl.obligation_id not in covered_obligations
            ]
            if not target_obligations:
                continue
            write_paths = tuple(
                path
                for path in placement.placement_paths
                if path in authorized_write_paths
            )
            if not write_paths:
                reasons.append(PlanRejectionReason.MISSING_WRITE_AUTHORITY.value)
                continue
            scc_ids = {
                consumer_to_scc.get(obl.consumer_id, "") for obl in target_obligations
            }
            scc_ids.discard("")
            scc_id = next(iter(scc_ids), "")
            group_id = f"group:{scc_id}" if scc_id else ""
            step_id = f"step:llm:{placement.behavior_id}:{placement.selected_candidate_id}"
            obligation_ids = tuple(sorted(obl.obligation_id for obl in target_obligations))
            pending_steps[step_id] = {
                "step_id": step_id,
                "kind": PlanStepKind.LLM_BOUNDED,
                "obligation_ids": obligation_ids,
                "dependency_transform_ids": (),
                "transform_id": "",
                "read_paths": _paths(
                    sorted(
                        set(write_paths)
                        | {obl.node.path for obl in target_obligations}
                    ),
                    "read_paths",
                ),
                "write_paths": _paths(sorted(write_paths), "write_paths", required=True),
                "precondition_refs": tuple(
                    sorted(
                        {
                            f"pre:placement:{placement.behavior_id}",
                            f"pre:candidate:{placement.selected_candidate_id}",
                        }
                    )
                ),
                "postcondition_refs": (
                    f"post:placement:{placement.behavior_id}",
                ),
                "validation_refs": validation_refs,
                "scc_group_id": group_id,
                "scc_id": scc_id,
                "consumer_ids": tuple(
                    sorted(obl.consumer_id for obl in target_obligations)
                ),
            }
            covered_obligations.update(obligation_ids)

        # When every consumer is already compatible/excluded, still emit one
        # write-free validation step so an admitted plan has a concrete DAG.
        if not pending_steps and evidence.obligations:
            only_passive = all(
                obl.disposition in _NO_STEP_DISPOSITIONS for obl in evidence.obligations
            )
            if only_passive:
                obl_ids = tuple(sorted(obl.obligation_id for obl in evidence.obligations))
                pending_steps["step:validation:closure"] = {
                    "step_id": "step:validation:closure",
                    "kind": PlanStepKind.VALIDATION,
                    "obligation_ids": obl_ids,
                    "dependency_transform_ids": (),
                    "transform_id": "",
                    "read_paths": _paths(
                        sorted(
                            {
                                obl.node.path
                                for obl in evidence.obligations
                                if obl.node.path in authorized_read_paths
                            }
                        ),
                        "read_paths",
                    ),
                    "write_paths": (),
                    "precondition_refs": ("pre:closure:complete",),
                    "postcondition_refs": ("post:validation:compatible",),
                    "validation_refs": validation_refs,
                    "scc_group_id": "",
                    "scc_id": "",
                    "consumer_ids": tuple(
                        sorted(obl.consumer_id for obl in evidence.obligations)
                    ),
                }

        # Resolve dependency_step_ids from transform dependency edges.
        for step_id, payload in pending_steps.items():
            dep_step_ids: list[str] = []
            for dep_tid in payload["dependency_transform_ids"]:
                dep_step = transform_id_to_step.get(dep_tid)
                if dep_step is None:
                    reasons.append(PlanRejectionReason.STEP_DAG_INVALID.value)
                elif dep_step != step_id:
                    dep_step_ids.append(dep_step)
            payload["dependency_step_ids"] = tuple(sorted(set(dep_step_ids)))

        # Detect dependency cycles outside a shared SCC group.
        cycle_reasons = self._detect_cycles_outside_scc(pending_steps)
        reasons.extend(cycle_reasons)

        # Topological order (deterministic Kahn with step_id tie-break).
        ordered_ids, topo_ok = self._topo_sort(pending_steps)
        if not topo_ok:
            reasons.append(PlanRejectionReason.CYCLE_OUTSIDE_SCC.value)

        steps: list[PropagationPlanStep] = []
        for step_id in ordered_ids:
            payload = pending_steps[step_id]
            steps.append(
                PropagationPlanStep(
                    step_id=payload["step_id"],
                    kind=payload["kind"],
                    obligation_ids=payload["obligation_ids"],
                    dependency_step_ids=payload["dependency_step_ids"],
                    transform_id=payload["transform_id"],
                    read_paths=payload["read_paths"],
                    write_paths=payload["write_paths"],
                    precondition_refs=payload["precondition_refs"],
                    postcondition_refs=payload["postcondition_refs"],
                    validation_refs=payload["validation_refs"],
                    scc_group_id=payload["scc_group_id"],
                )
            )

        # Build SCC groups for multi-member SCCs that have steps.
        scc_groups: list[PropagationSCCGroup] = []
        steps_by_group: dict[str, list[str]] = defaultdict(list)
        consumers_by_group: dict[str, set[str]] = defaultdict(set)
        scc_id_by_group: dict[str, str] = {}
        for step_id, payload in pending_steps.items():
            group_id = payload["scc_group_id"]
            if not group_id:
                continue
            # Only materialize groups for multi-member SCCs from the closure.
            scc_id = payload["scc_id"]
            member_count = 0
            for scc in closure.sccs:
                if scc.scc_id == scc_id:
                    member_count = len(scc.member_consumer_ids)
                    break
            if member_count < 2:
                continue
            steps_by_group[group_id].append(step_id)
            consumers_by_group[group_id].update(payload["consumer_ids"])
            scc_id_by_group[group_id] = scc_id

        for group_id in sorted(steps_by_group):
            scc_groups.append(
                PropagationSCCGroup(
                    group_id=group_id,
                    scc_id=scc_id_by_group[group_id],
                    step_ids=tuple(sorted(steps_by_group[group_id])),
                    consumer_ids=tuple(sorted(consumers_by_group[group_id])),
                )
            )

        return steps, scc_groups, reasons, alternatives

    @staticmethod
    def _detect_cycles_outside_scc(
        pending_steps: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        """Return rejection reasons for dependency cycles not confined to one SCC group."""

        reasons: list[str] = []
        # Build adjacency from dependency_transform resolution already done.
        graph: dict[str, set[str]] = {sid: set() for sid in pending_steps}
        for sid, payload in pending_steps.items():
            for dep in payload.get("dependency_step_ids", ()):
                if dep in graph:
                    graph[dep].add(sid)

        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        components: list[list[str]] = []

        def strongconnect(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlink[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            for succ in sorted(graph[node]):
                if succ not in indices:
                    strongconnect(succ)
                    lowlink[node] = min(lowlink[node], lowlink[succ])
                elif succ in on_stack:
                    lowlink[node] = min(lowlink[node], indices[succ])
            if lowlink[node] == indices[node]:
                component: list[str] = []
                while True:
                    item = stack.pop()
                    on_stack.discard(item)
                    component.append(item)
                    if item == node:
                        break
                components.append(component)

        for node in sorted(graph):
            if node not in indices:
                strongconnect(node)

        for component in components:
            if len(component) < 2:
                # Self-loop check.
                node = component[0]
                if node in graph.get(node, ()):
                    reasons.append(PlanRejectionReason.CYCLE_OUTSIDE_SCC.value)
                continue
            groups = {
                pending_steps[sid].get("scc_group_id", "")
                for sid in component
            }
            groups.discard("")
            if len(groups) != 1:
                reasons.append(PlanRejectionReason.CYCLE_OUTSIDE_SCC.value)
        return reasons

    @staticmethod
    def _topo_sort(
        pending_steps: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[str], bool]:
        """Deterministic topological sort; returns (order, ok)."""

        indegree: dict[str, int] = {sid: 0 for sid in pending_steps}
        successors: dict[str, list[str]] = defaultdict(list)
        for sid, payload in pending_steps.items():
            for dep in payload.get("dependency_step_ids", ()):
                if dep not in pending_steps:
                    continue
                successors[dep].append(sid)
                indegree[sid] += 1
        for sid in successors:
            successors[sid] = sorted(set(successors[sid]))

        ready = deque(sorted(sid for sid, deg in indegree.items() if deg == 0))
        order: list[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for succ in successors[node]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    # Insert sorted to keep deterministic ready queue.
                    ready_list = sorted(list(ready) + [succ])
                    ready = deque(ready_list)
        if len(order) != len(pending_steps):
            # Cyclic remainder: append remaining by step_id for stable diagnostics.
            remaining = sorted(set(pending_steps) - set(order))
            order.extend(remaining)
            return order, False
        return order, True


def _diagnostic_obligation(
    roots: PropagationAuthorityRoots,
    delta_id: str,
) -> ConsumerMigrationObligation:
    """Minimal obligation used only when evidence omitted all obligations."""

    from ..analysis.change_propagation_contracts import GraphNodeRef, GraphProvenance

    node = GraphNodeRef(
        node_id="node:diagnostic",
        kind="function",
        path="pkg/diagnostic.py",
        symbol_id="symbol:diagnostic",
        artifact_id="blob:diagnostic",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:diagnostic",
    )
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id="obligation:diagnostic",
        consumer_id="consumer:diagnostic",
        delta_id=delta_id,
        disposition=ConsumerDisposition.ABSTAIN,
        clause_ids=("clause:diagnostic",),
        node=node,
        invalidation_refs=(roots.candidate_tree_id,),
    )


def admit_change_propagation_plan(
    evidence: PlanEvidenceBundle,
) -> PropagationPlanAdmission:
    """Module-level entry point matching the planner interface."""

    return ChangePropagationPlanner().admit(evidence)


def plan_set_identity(plans: Sequence[AtomicPropagationPlan]) -> str:
    """Identity over a deterministic set of atomic plans (for equality checks)."""

    if (
        isinstance(plans, (str, bytes, bytearray))
        or not isinstance(plans, Sequence)
        or not plans
    ):
        raise ChangePropagationPlanError("plans must be a non-empty sequence")
    if not all(isinstance(item, AtomicPropagationPlan) for item in plans):
        raise ChangePropagationPlanError("plans must contain AtomicPropagationPlan values")
    ids = tuple(sorted(item.content_id for item in plans))
    if len(set(ids)) != len(ids):
        raise ChangePropagationPlanError("plan set contains duplicate plans")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-set@1",
            "plan_ids": list(ids),
        }
    )


__all__ = (
    "CHANGE_PROPAGATION_PLANNER_INTERFACE",
    "PROPAGATION_PLAN_ADMISSION_SCHEMA",
    "PLAN_EVIDENCE_BUNDLE_SCHEMA",
    "PLAN_PATH_SPAN_SCHEMA",
    "PLAN_RESOURCE_BOUNDS_SCHEMA",
    "PLAN_VALIDATION_COMMAND_SCHEMA",
    "PRODUCER_ID",
    "ChangePropagationPlanAuthorityError",
    "ChangePropagationPlanError",
    "ChangePropagationPlanner",
    "PlanEvidenceBundle",
    "PlanPathSpan",
    "PlanRejectionReason",
    "PlanResourceBounds",
    "PlanValidationCommand",
    "PropagationPlanAdmission",
    "admit_change_propagation_plan",
    "plan_set_identity",
)
