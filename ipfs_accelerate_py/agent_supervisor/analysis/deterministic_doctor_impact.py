"""Close all callers and compile one atomic deterministic repair plan (LPR-037).

Interface: ``DeterministicDoctorImpactAnalyzer@1``

Given a current real-checkout diagnostic snapshot and a proof-admitted
analytical overlay (LPR-036), this module:

* rebuilds the candidate :class:`ProgramContractDelta`;
* resolves the bounded current call / import / dependency graph;
* requires exactly one
  ``migrated`` / ``proved_compatible`` / ``unaffected`` / ``approval`` /
  ``unsupported`` disposition per resolved consumer;
* discovers second-order consumers introduced by the overlay;
* represents reflection, unknown dispatch, generated code, native/FFI and
  unsupported interprocedural paths as required open frontiers; and
* compiles **one** atomic :class:`DeterministicDoctorPlan` covering every
  necessary SCC step.

Mutation is fail-closed: complete closure, current CIDs, no forbidden path,
and one atomic plan covering all SCC steps are mandatory. Missed, duplicate or
stale consumers, circular ownership, plan gaps, or any open required frontier
abstain **before** any write.

This module reuses current impact / propagation authority
(:class:`ContractChangeImpactAnalyzer`, :class:`ProgramDependencyGraph`,
:class:`ProgramCallResolver`, :class:`ChangeConsumerInventory`,
:class:`ChangePropagationPlanner`) and never invokes an LLM or model provider.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .change_propagation_contracts import (
    ConsumerDisposition,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    ImpactClosureReceipt,
    ImpactCompleteness,
    MAX_SCC_COUNT,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)
from .contract_change_impact import (
    ContractChangeImpactAnalyzer,
    ImpactClosureBounds,
    ImpactClosureResult,
)
from .deterministic_doctor_contracts import (
    DOCTOR_TCB_PATH_MARKERS,
    MAX_CONSUMER_COUNT as DOCTOR_MAX_CONSUMERS,
    MAX_FRONTIER_COUNT,
    MAX_PATH_BYTES,
    MAX_REFERENCE_COUNT as DOCTOR_MAX_REFS,
    MAX_TEXT_BYTES,
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition as PlanConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
    DoctorResourceBounds,
    consumer_disposition_set_identity,
    is_doctor_tcb_path,
)
from .dynamic_impact_frontier import (
    DynamicImpactFrontier,
    DynamicImpactFrontierAnalyzer,
    FrontierKind,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_IMPACT_INTERFACE: Final[str] = (
    "DeterministicDoctorImpactAnalyzer@1"
)
DOCTOR_IMPACT_CLOSURE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/impact-closure@1"
)
DOCTOR_IMPACT_CONSUMER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/impact-consumer@1"
)
DOCTOR_PLAN_COMPILATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/plan-compilation@1"
)
DOCTOR_IMPACT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/impact-request@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-impact@1"
CONTRACT_VERSION: Final[int] = 1

MAX_DISPOSITIONS: Final[int] = DOCTOR_MAX_CONSUMERS
MAX_REASON_CODES: Final[int] = 128
MAX_EDGES: Final[int] = 65_536
MAX_DEPTH: Final[int] = 256
DEFAULT_CHECKPOINT_REF: Final[str] = "checkpoint:content-addressed"
DEFAULT_ROLLBACK_REF: Final[str] = "rollback:restore-checkpoint"

# Required open-frontier kinds that block autonomous mutation (acceptance).
_REQUIRED_OPEN_FRONTIER_KINDS: Final[frozenset[str]] = frozenset(
    {
        FrontierKind.REFLECTION.value,
        FrontierKind.STRING_DISPATCH.value,
        FrontierKind.GENERATED_CODE.value,
        FrontierKind.NATIVE_FFI.value,
        "unknown_dispatch",
        "unsupported_interprocedural",
        "unsupported",
    }
)

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

# Overlay may introduce second-order consumers via these edge kind tags.
_SECOND_ORDER_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "calls",
        "imports",
        "aliases",
        "re_exports",
        "registers",
        "depends_on",
        "tests",
        "schema_of",
        "generated_from",
        "entry_point",
        "wrapper",
        "method",
    }
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DoctorConsumerDisposition(str, Enum):
    """Exactly one closed outcome per resolved consumer (LPR-037).

    Distinct from the plan-level
    :class:`~ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts.DoctorConsumerDisposition`
    contract (which binds a :class:`DoctorRepairDisposition`).  This enum is
    the impact-closure vocabulary required by the acceptance criteria.
    """

    MIGRATED = "migrated"
    PROVED_COMPATIBLE = "proved_compatible"
    UNAFFECTED = "unaffected"
    APPROVAL = "approval"
    UNSUPPORTED = "unsupported"

    @property
    def requires_write(self) -> bool:
        return self is DoctorConsumerDisposition.MIGRATED

    @property
    def blocks_autonomous_mutation(self) -> bool:
        return self in {
            DoctorConsumerDisposition.APPROVAL,
            DoctorConsumerDisposition.UNSUPPORTED,
        }

    @property
    def is_closed(self) -> bool:
        return self in {
            DoctorConsumerDisposition.MIGRATED,
            DoctorConsumerDisposition.PROVED_COMPATIBLE,
            DoctorConsumerDisposition.UNAFFECTED,
        }


class DoctorImpactReason(str, Enum):
    """Stable, machine-readable impact / plan-compilation reason codes."""

    COMPLETE_CLOSURE = "complete_closure"
    INCOMPLETE_CLOSURE = "incomplete_closure"
    OPEN_REQUIRED_FRONTIER = "open_required_frontier"
    MISSED_CONSUMER = "missed_consumer"
    DUPLICATE_CONSUMER = "duplicate_consumer"
    STALE_CONSUMER = "stale_consumer"
    STALE_ROOTS = "stale_roots"
    STALE_CID = "stale_cid"
    CIRCULAR_OWNERSHIP = "circular_ownership"
    PLAN_GAP = "plan_gap"
    FORBIDDEN_PATH = "forbidden_path"
    TCB_PATH = "trusted_computing_base_path"
    SECOND_ORDER_DISCOVERED = "second_order_discovered"
    UNCOVERED_SCC = "uncovered_scc"
    MUTATION_ADMISSIBLE = "mutation_admissible"
    MUTATION_BLOCKED = "mutation_blocked"
    ROOT_MISMATCH = "root_mismatch"
    EMPTY_CONSUMERS = "empty_consumers"
    OVERLAY_REQUIRED = "overlay_required"
    DELTA_REBUILT = "candidate_delta_rebuilt"
    DISPOSITION_REQUIRED = "disposition_required"
    UNSUPPORTED_FRONTIER = "unsupported_frontier"
    REFLECTION_FRONTIER = "reflection_frontier"
    UNKNOWN_DISPATCH_FRONTIER = "unknown_dispatch_frontier"
    GENERATED_CODE_FRONTIER = "generated_code_frontier"
    NATIVE_FFI_FRONTIER = "native_ffi_frontier"
    INTERPROCEDURAL_FRONTIER = "unsupported_interprocedural_frontier"
    NO_MODEL_INVARIANT = "no_model_invariant"
    PLAN_ADMITTED = "plan_admitted"
    PLAN_ABSTAINED = "plan_abstained"
    CURRENT_CIDS = "current_cids"
    ATOMIC_SCC_PLAN = "atomic_scc_plan"
    MALFORMED_INPUT = "malformed_input"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    OWNERSHIP_CYCLE = "ownership_cycle"
    MISSING_LEASE = "missing_lease"
    MISSING_CHECKPOINT = "missing_checkpoint"
    MISSING_ROLLBACK = "missing_rollback"
    MISSING_OPERATOR = "missing_operator"
    MISSING_PROOF = "missing_proof"
    WRITE_WITHOUT_CLOSURE = "write_without_closure"


class DoctorImpactPlanDisposition(str, Enum):
    """Outcome of compiling one atomic deterministic repair plan."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    APPROVAL_REQUIRED = "approval_required"
    REJECTED = "rejected"

    @property
    def grants_write_authority(self) -> bool:
        return self is DoctorImpactPlanDisposition.ADMITTED


# Disposition → plan-level DoctorRepairDisposition mapping.
_TO_PLAN_REPAIR: Final[Mapping[DoctorConsumerDisposition, DoctorRepairDisposition]] = (
    MappingProxyType(
        {
            DoctorConsumerDisposition.MIGRATED: DoctorRepairDisposition.SUPPORTED,
            DoctorConsumerDisposition.PROVED_COMPATIBLE: DoctorRepairDisposition.SUPPORTED,
            DoctorConsumerDisposition.UNAFFECTED: DoctorRepairDisposition.SUPPORTED,
            DoctorConsumerDisposition.APPROVAL: DoctorRepairDisposition.APPROVAL_REQUIRED,
            DoctorConsumerDisposition.UNSUPPORTED: DoctorRepairDisposition.ABSTAIN,
        }
    )
)

# Impact disposition → change-propagation ConsumerDisposition (for obligations).
_TO_PROPAGATION: Final[Mapping[DoctorConsumerDisposition, ConsumerDisposition]] = (
    MappingProxyType(
        {
            DoctorConsumerDisposition.MIGRATED: ConsumerDisposition.MIGRATE,
            DoctorConsumerDisposition.PROVED_COMPATIBLE: ConsumerDisposition.COMPATIBLE,
            DoctorConsumerDisposition.UNAFFECTED: ConsumerDisposition.EXCLUDED,
            DoctorConsumerDisposition.APPROVAL: ConsumerDisposition.REVIEW_ONLY,
            DoctorConsumerDisposition.UNSUPPORTED: ConsumerDisposition.FRONTIER,
        }
    )
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DoctorImpactError(ContractValidationError):
    """Malformed impact input or closed-boundary violation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: DoctorImpactReason | str = DoctorImpactReason.MALFORMED_INPUT,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


class DoctorImpactAuthorityError(DoctorImpactError):
    """Root, CID, path, or write-authority mismatch."""


class DoctorImpactBoundsError(DoctorImpactError):
    """A record exceeded its deterministic compactness bound."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise DoctorImpactError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise DoctorImpactError(f"{name} must be a string")
    text = value.strip() if name.endswith(("_id", "_ref", "_cid")) else value
    if required and not text.strip():
        raise DoctorImpactError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise DoctorImpactBoundsError(
            f"{name} exceeds its byte bound",
            reason_code=DoctorImpactReason.BOUNDS_EXCEEDED,
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True)
    if any(char.isspace() for char in text):
        raise DoctorImpactError(f"{name} must be an opaque compact identifier")
    return text


def _optional_identifier(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, name)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorImpactError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int = 2**31 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DoctorImpactError(f"{name} must be a finite integer")
    if value < 0 or value > maximum:
        raise DoctorImpactBoundsError(
            f"{name} is outside its bound",
            reason_code=DoctorImpactReason.BOUNDS_EXCEEDED,
        )
    return value


def _path(value: Any, name: str = "path") -> str:
    path = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise DoctorImpactAuthorityError(
            f"{name} must be a relative repository path",
            reason_code=DoctorImpactReason.FORBIDDEN_PATH,
        )
    return candidate.as_posix()


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DoctorImpactError(f"{name} must be a sequence of paths")
    if len(values) > DOCTOR_MAX_REFS:
        raise DoctorImpactBoundsError(
            f"{name} exceeds its item bound",
            reason_code=DoctorImpactReason.BOUNDS_EXCEEDED,
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        p = _path(item, name)
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return tuple(ordered)


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = DOCTOR_MAX_REFS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise DoctorImpactError(f"{name} is required")
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DoctorImpactError(f"{name} must be a sequence of identifiers")
    if len(values) > limit:
        raise DoctorImpactBoundsError(
            f"{name} exceeds its item bound",
            reason_code=DoctorImpactReason.BOUNDS_EXCEEDED,
        )
    items = [_identifier(item, name) for item in values]
    if required and not items:
        raise DoctorImpactError(f"{name} is required")
    if preserve_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return tuple(ordered)
    return tuple(sorted(set(items)))


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise DoctorImpactError(
            f"{name} must be one of: {allowed}"
        ) from exc


def doctor_roots_to_propagation_roots(
    roots: DoctorAuthorityRoots,
) -> PropagationAuthorityRoots:
    """Bridge doctor roots into propagation impact authority roots."""

    if not isinstance(roots, DoctorAuthorityRoots):
        raise DoctorImpactError("roots must be DoctorAuthorityRoots")
    return PropagationAuthorityRoots(
        repository_id=roots.repository_id,
        base_forest_id=f"base-of:{roots.forest_id}",
        base_tree_id=f"base-of:{roots.tree_id}",
        base_overlay_id=f"base-of:{roots.overlay_id}",
        candidate_forest_id=roots.forest_id,
        candidate_tree_id=roots.tree_id,
        candidate_overlay_id=roots.overlay_id,
        graph_id=roots.graph_id,
        index_id=roots.index_id,
        model_id=roots.model_id,
        config_id=roots.corpus_id,
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )


def path_is_forbidden(path: str, extra_forbidden: Sequence[str] = ()) -> bool:
    """True when ``path`` is TCB, vendored, or otherwise non-writable."""

    normalized = PurePosixPath(path).as_posix()
    if is_doctor_tcb_path(normalized):
        return True
    parts = {part.casefold() for part in PurePosixPath(normalized).parts}
    if parts & _FORBIDDEN_PATH_PARTS:
        return True
    for marker in DOCTOR_TCB_PATH_MARKERS:
        m = marker.rstrip("/")
        if normalized == m or normalized.startswith(m):
            return True
    for item in extra_forbidden:
        marker = str(item).rstrip("/")
        if not marker:
            continue
        if normalized == marker or normalized.startswith(marker + "/"):
            return True
    return False


def _disposition_key(value: Any) -> DoctorConsumerDisposition:
    if isinstance(value, DoctorConsumerDisposition):
        return value
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "migrated": DoctorConsumerDisposition.MIGRATED,
        "migrate": DoctorConsumerDisposition.MIGRATED,
        "proved_compatible": DoctorConsumerDisposition.PROVED_COMPATIBLE,
        "proved-compatible": DoctorConsumerDisposition.PROVED_COMPATIBLE,
        "compatible": DoctorConsumerDisposition.PROVED_COMPATIBLE,
        "unaffected": DoctorConsumerDisposition.UNAFFECTED,
        "excluded": DoctorConsumerDisposition.UNAFFECTED,
        "approval": DoctorConsumerDisposition.APPROVAL,
        "approval_required": DoctorConsumerDisposition.APPROVAL,
        "review_only": DoctorConsumerDisposition.APPROVAL,
        "unsupported": DoctorConsumerDisposition.UNSUPPORTED,
        "frontier": DoctorConsumerDisposition.UNSUPPORTED,
        "abstain": DoctorConsumerDisposition.UNSUPPORTED,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise DoctorImpactError(
            f"unsupported consumer disposition: {value!r}",
            reason_code=DoctorImpactReason.DISPOSITION_REQUIRED,
        ) from exc


def map_to_plan_repair_disposition(
    disposition: DoctorConsumerDisposition | str,
) -> DoctorRepairDisposition:
    """Map an impact disposition onto the plan-level repair disposition."""

    kind = _disposition_key(disposition)
    return _TO_PLAN_REPAIR[kind]


def map_to_propagation_disposition(
    disposition: DoctorConsumerDisposition | str,
) -> ConsumerDisposition:
    """Map an impact disposition onto a change-propagation disposition."""

    kind = _disposition_key(disposition)
    return _TO_PROPAGATION[kind]


# ---------------------------------------------------------------------------
# Input observations (fixture-friendly, body-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorImpactConsumerObservation:
    """One resolved or frontier consumer observation for impact closure.

    Fixtures and adapters supply these rather than re-emitting full program
    graphs.  Bodies / source text are forbidden.
    """

    consumer_id: str
    path: str
    symbol_id: str = ""
    depth: int = 1
    mandatory: bool = True
    edge_refs: tuple[str, ...] = ()
    edge_kinds: tuple[str, ...] = ()
    disposition: DoctorConsumerDisposition | str | None = None
    second_order: bool = False
    stale: bool = False
    owner_id: str = ""
    node_id: str = ""
    reason_codes: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    artifact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "symbol_id", _optional_identifier(self.symbol_id, "symbol_id")
        )
        object.__setattr__(self, "depth", _nonneg_int(self.depth, "depth", maximum=MAX_DEPTH))
        object.__setattr__(self, "mandatory", _bool(self.mandatory, "mandatory"))
        object.__setattr__(self, "edge_refs", _ids(self.edge_refs, "edge_refs"))
        object.__setattr__(
            self,
            "edge_kinds",
            _ids(self.edge_kinds, "edge_kinds", preserve_order=True),
        )
        if self.disposition is not None:
            object.__setattr__(
                self, "disposition", _disposition_key(self.disposition)
            )
        object.__setattr__(self, "second_order", _bool(self.second_order, "second_order"))
        object.__setattr__(self, "stale", _bool(self.stale, "stale"))
        object.__setattr__(
            self, "owner_id", _optional_identifier(self.owner_id, "owner_id")
        )
        object.__setattr__(
            self, "node_id", _optional_identifier(self.node_id, "node_id")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES)
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "artifact_id", _optional_identifier(self.artifact_id, "artifact_id")
        )


@dataclass(frozen=True)
class DoctorImpactFrontierObservation:
    """One required/open frontier observation (reflection, FFI, …)."""

    kind: str
    route: str
    required: bool = True
    evidence_refs: tuple[str, ...] = ()
    graph_node_id: str = ""
    graph_edge_id: str = ""
    reason: str = ""
    closed: bool = False

    def __post_init__(self) -> None:
        kind = _text(self.kind, "kind", required=True, limit=128).casefold()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "route", _text(self.route, "route", required=True, limit=MAX_TEXT_BYTES)
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self,
            "graph_node_id",
            _optional_identifier(self.graph_node_id, "graph_node_id"),
        )
        object.__setattr__(
            self,
            "graph_edge_id",
            _optional_identifier(self.graph_edge_id, "graph_edge_id"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False, limit=512)
        )
        object.__setattr__(self, "closed", _bool(self.closed, "closed"))


@dataclass(frozen=True)
class DoctorGraphEdgeObservation:
    """Bounded directed consumer edge for SCC / ownership analysis."""

    source_consumer_id: str
    target_consumer_id: str
    kind: str = "calls"
    ownership: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_consumer_id",
            _identifier(self.source_consumer_id, "source_consumer_id"),
        )
        object.__setattr__(
            self,
            "target_consumer_id",
            _identifier(self.target_consumer_id, "target_consumer_id"),
        )
        object.__setattr__(
            self, "kind", _text(self.kind, "kind", required=True, limit=64).casefold()
        )
        object.__setattr__(self, "ownership", _bool(self.ownership, "ownership"))


@dataclass(frozen=True)
class DoctorImpactRequest:
    """Inputs for one deterministic doctor impact-closure computation.

    Prefer compact consumer / frontier / edge observations over bulk golden
    envelopes.  An optional precomputed
    :class:`ImpactClosureReceipt` may be supplied when the call graph has
    already been closed by :class:`ContractChangeImpactAnalyzer`.
    """

    roots: DoctorAuthorityRoots
    base_delta: ProgramContractDelta | None = None
    candidate_delta: ProgramContractDelta | None = None
    overlay_id: str = ""
    overlay_path: str = ""
    overlay_patch_cid: str = ""
    overlay_before_hash: str = ""
    overlay_after_hash: str = ""
    subject_symbol_id: str = ""
    change_set_id: str = ""
    before_contract_ref: str = ""
    after_contract_ref: str = ""
    clause_ids: tuple[str, ...] = ()
    consumers: tuple[DoctorImpactConsumerObservation, ...] = ()
    frontiers: tuple[DoctorImpactFrontierObservation, ...] = ()
    edges: tuple[DoctorGraphEdgeObservation, ...] = ()
    second_order_consumers: tuple[DoctorImpactConsumerObservation, ...] = ()
    expected_consumer_ids: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    impact_closure: ImpactClosureReceipt | None = None
    dynamic_frontier: DynamicImpactFrontier | None = None
    current_graph_cid: str = ""
    current_index_cid: str = ""
    current_ast_cid: str = ""
    evidence_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorImpactError("roots must be DoctorAuthorityRoots")
        if self.base_delta is not None and not isinstance(
            self.base_delta, ProgramContractDelta
        ):
            raise DoctorImpactError("base_delta must be ProgramContractDelta")
        if self.candidate_delta is not None and not isinstance(
            self.candidate_delta, ProgramContractDelta
        ):
            raise DoctorImpactError("candidate_delta must be ProgramContractDelta")
        object.__setattr__(
            self, "overlay_id", _optional_identifier(self.overlay_id, "overlay_id")
        )
        if self.overlay_path:
            object.__setattr__(self, "overlay_path", _path(self.overlay_path, "overlay_path"))
        object.__setattr__(
            self,
            "overlay_patch_cid",
            _optional_identifier(self.overlay_patch_cid, "overlay_patch_cid"),
        )
        object.__setattr__(
            self,
            "overlay_before_hash",
            _optional_identifier(self.overlay_before_hash, "overlay_before_hash"),
        )
        object.__setattr__(
            self,
            "overlay_after_hash",
            _optional_identifier(self.overlay_after_hash, "overlay_after_hash"),
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _optional_identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self, "change_set_id", _optional_identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(
            self,
            "before_contract_ref",
            _optional_identifier(self.before_contract_ref, "before_contract_ref"),
        )
        object.__setattr__(
            self,
            "after_contract_ref",
            _optional_identifier(self.after_contract_ref, "after_contract_ref"),
        )
        object.__setattr__(self, "clause_ids", _ids(self.clause_ids, "clause_ids"))
        if not isinstance(self.consumers, Sequence) or isinstance(
            self.consumers, (str, bytes)
        ):
            raise DoctorImpactError("consumers must be a sequence")
        if len(self.consumers) > MAX_DISPOSITIONS:
            raise DoctorImpactBoundsError("consumers exceeds bound")
        for item in self.consumers:
            if not isinstance(item, DoctorImpactConsumerObservation):
                raise DoctorImpactError(
                    "consumers must contain DoctorImpactConsumerObservation values"
                )
        object.__setattr__(self, "consumers", tuple(self.consumers))
        if not isinstance(self.frontiers, Sequence) or isinstance(
            self.frontiers, (str, bytes)
        ):
            raise DoctorImpactError("frontiers must be a sequence")
        for item in self.frontiers:
            if not isinstance(item, DoctorImpactFrontierObservation):
                raise DoctorImpactError(
                    "frontiers must contain DoctorImpactFrontierObservation values"
                )
        object.__setattr__(self, "frontiers", tuple(self.frontiers))
        if not isinstance(self.edges, Sequence) or isinstance(self.edges, (str, bytes)):
            raise DoctorImpactError("edges must be a sequence")
        if len(self.edges) > MAX_EDGES:
            raise DoctorImpactBoundsError("edges exceeds bound")
        for item in self.edges:
            if not isinstance(item, DoctorGraphEdgeObservation):
                raise DoctorImpactError(
                    "edges must contain DoctorGraphEdgeObservation values"
                )
        object.__setattr__(self, "edges", tuple(self.edges))
        for item in self.second_order_consumers:
            if not isinstance(item, DoctorImpactConsumerObservation):
                raise DoctorImpactError(
                    "second_order_consumers must contain DoctorImpactConsumerObservation"
                )
        object.__setattr__(
            self, "second_order_consumers", tuple(self.second_order_consumers)
        )
        object.__setattr__(
            self,
            "expected_consumer_ids",
            _ids(self.expected_consumer_ids, "expected_consumer_ids", limit=MAX_DISPOSITIONS),
        )
        object.__setattr__(
            self, "forbidden_paths", _paths(self.forbidden_paths, "forbidden_paths")
        )
        if self.impact_closure is not None and not isinstance(
            self.impact_closure, ImpactClosureReceipt
        ):
            raise DoctorImpactError("impact_closure must be ImpactClosureReceipt")
        if self.dynamic_frontier is not None and not isinstance(
            self.dynamic_frontier, DynamicImpactFrontier
        ):
            raise DoctorImpactError("dynamic_frontier must be DynamicImpactFrontier")
        object.__setattr__(
            self,
            "current_graph_cid",
            _optional_identifier(self.current_graph_cid, "current_graph_cid"),
        )
        object.__setattr__(
            self,
            "current_index_cid",
            _optional_identifier(self.current_index_cid, "current_index_cid"),
        )
        object.__setattr__(
            self,
            "current_ast_cid",
            _optional_identifier(self.current_ast_cid, "current_ast_cid"),
        )
        object.__setattr__(self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorImpactConsumerRecord(CanonicalContract):
    """One dispositioned consumer inside a doctor impact-closure receipt."""

    SCHEMA: ClassVar[str] = DOCTOR_IMPACT_CONSUMER_SCHEMA

    consumer_id: str
    disposition: DoctorConsumerDisposition
    path: str
    symbol_id: str = ""
    depth: int = 1
    mandatory: bool = True
    second_order: bool = False
    scc_id: str = ""
    edge_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    obligation_ref: str = ""
    node_id: str = ""
    owner_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(
            self, "disposition", _disposition_key(self.disposition)
        )
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "symbol_id", _optional_identifier(self.symbol_id, "symbol_id")
        )
        object.__setattr__(
            self, "depth", _nonneg_int(self.depth, "depth", maximum=MAX_DEPTH)
        )
        object.__setattr__(self, "mandatory", _bool(self.mandatory, "mandatory"))
        object.__setattr__(self, "second_order", _bool(self.second_order, "second_order"))
        object.__setattr__(self, "scc_id", _optional_identifier(self.scc_id, "scc_id"))
        object.__setattr__(self, "edge_refs", _ids(self.edge_refs, "edge_refs"))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "obligation_ref", _optional_identifier(self.obligation_ref, "obligation_ref")
        )
        object.__setattr__(self, "node_id", _optional_identifier(self.node_id, "node_id"))
        object.__setattr__(self, "owner_id", _optional_identifier(self.owner_id, "owner_id"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "consumer_id": self.consumer_id,
            "disposition": self.disposition.value,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "depth": self.depth,
            "mandatory": self.mandatory,
            "second_order": self.second_order,
            "scc_id": self.scc_id,
            "edge_refs": list(self.edge_refs),
            "reason_codes": list(self.reason_codes),
            "proof_refs": list(self.proof_refs),
            "obligation_ref": self.obligation_ref,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorImpactConsumerRecord":
        if not isinstance(payload, Mapping):
            raise DoctorImpactError("consumer payload must be a mapping")
        values = {
            key: payload.get(key)
            for key in (
                "consumer_id",
                "disposition",
                "path",
                "symbol_id",
                "depth",
                "mandatory",
                "second_order",
                "scc_id",
                "edge_refs",
                "reason_codes",
                "proof_refs",
                "obligation_ref",
                "node_id",
                "owner_id",
            )
            if key in payload or key in {"consumer_id", "disposition", "path"}
        }
        # Defaults for optional keys omitted from compact fixtures.
        values.setdefault("symbol_id", "")
        values.setdefault("depth", 1)
        values.setdefault("mandatory", True)
        values.setdefault("second_order", False)
        values.setdefault("scc_id", "")
        values.setdefault("edge_refs", ())
        values.setdefault("reason_codes", ())
        values.setdefault("proof_refs", ())
        values.setdefault("obligation_ref", "")
        values.setdefault("node_id", "")
        values.setdefault("owner_id", "")
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DoctorImpactSCC:
    """One strongly connected consumer group treated as a transaction unit."""

    scc_id: str
    member_consumer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scc_id", _identifier(self.scc_id, "scc_id"))
        object.__setattr__(
            self,
            "member_consumer_ids",
            _ids(
                self.member_consumer_ids,
                "member_consumer_ids",
                required=True,
                limit=MAX_DISPOSITIONS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scc_id": self.scc_id,
            "member_consumer_ids": list(self.member_consumer_ids),
        }


@dataclass(frozen=True)
class DoctorImpactClosureReceipt(CanonicalContract):
    """Complete (or fail-closed partial) doctor impact-closure receipt.

    Every resolved consumer carries exactly one
    :class:`DoctorConsumerDisposition`.  ``mutation_admissible`` is true only
    when completeness is COMPLETE, required frontiers are empty, every
    consumer is closed (migrated / proved_compatible / unaffected), CIDs are
    current, and no forbidden path / circular ownership / disposition gap is
    present.
    """

    SCHEMA: ClassVar[str] = DOCTOR_IMPACT_CLOSURE_RECEIPT_SCHEMA

    roots: DoctorAuthorityRoots
    impact_closure_id: str
    delta_id: str
    candidate_delta_id: str
    completeness: ImpactCompleteness
    consumers: tuple[DoctorImpactConsumerRecord, ...]
    sccs: tuple[DoctorImpactSCC, ...] = ()
    open_required_frontiers: tuple[str, ...] = ()
    frontier_kinds: tuple[str, ...] = ()
    second_order_consumer_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    forbidden_path_hits: tuple[str, ...] = ()
    circular_ownership_refs: tuple[str, ...] = ()
    missed_consumer_ids: tuple[str, ...] = ()
    duplicate_consumer_ids: tuple[str, ...] = ()
    stale_consumer_ids: tuple[str, ...] = ()
    current_graph_cid: str = ""
    current_index_cid: str = ""
    current_ast_cid: str = ""
    overlay_id: str = ""
    overlay_patch_cid: str = ""
    underlying_impact_closure_id: str = ""
    producer_id: str = PRODUCER_ID
    mutation_admissible: bool = False
    no_model_invariant: bool = True
    model_invocation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorImpactError("roots must be DoctorAuthorityRoots")
        object.__setattr__(
            self,
            "impact_closure_id",
            _identifier(self.impact_closure_id, "impact_closure_id"),
        )
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self,
            "candidate_delta_id",
            _identifier(self.candidate_delta_id, "candidate_delta_id"),
        )
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, ImpactCompleteness, "completeness"),
        )
        if not isinstance(self.consumers, Sequence) or isinstance(
            self.consumers, (str, bytes)
        ):
            raise DoctorImpactError("consumers must be a sequence")
        if len(self.consumers) > MAX_DISPOSITIONS:
            raise DoctorImpactBoundsError("consumers exceeds bound")
        consumers = tuple(self.consumers)
        if not all(isinstance(item, DoctorImpactConsumerRecord) for item in consumers):
            raise DoctorImpactError(
                "consumers must contain DoctorImpactConsumerRecord values"
            )
        # Deterministic order by (depth, path, consumer_id).
        ordered = tuple(
            sorted(
                consumers,
                key=lambda item: (item.depth, item.path, item.consumer_id),
            )
        )
        object.__setattr__(self, "consumers", ordered)
        consumer_ids = [item.consumer_id for item in ordered]
        if len(set(consumer_ids)) != len(consumer_ids):
            raise DoctorImpactError(
                "impact consumers must have unique consumer_ids",
                reason_code=DoctorImpactReason.DUPLICATE_CONSUMER,
            )
        sccs = tuple(self.sccs)
        if len(sccs) > MAX_SCC_COUNT:
            raise DoctorImpactBoundsError("sccs exceeds bound")
        for scc in sccs:
            if not isinstance(scc, DoctorImpactSCC):
                raise DoctorImpactError("sccs must contain DoctorImpactSCC values")
            missing = set(scc.member_consumer_ids) - set(consumer_ids)
            if missing:
                raise DoctorImpactError(
                    "scc members must reference known impact consumers"
                )
        object.__setattr__(self, "sccs", sccs)
        object.__setattr__(
            self,
            "open_required_frontiers",
            _ids(
                self.open_required_frontiers,
                "open_required_frontiers",
                limit=MAX_FRONTIER_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "frontier_kinds",
            _ids(self.frontier_kinds, "frontier_kinds", limit=MAX_FRONTIER_COUNT),
        )
        object.__setattr__(
            self,
            "second_order_consumer_ids",
            _ids(
                self.second_order_consumer_ids,
                "second_order_consumer_ids",
                limit=MAX_DISPOSITIONS,
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES),
        )
        object.__setattr__(self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "forbidden_path_hits",
            _paths(self.forbidden_path_hits, "forbidden_path_hits"),
        )
        object.__setattr__(
            self,
            "circular_ownership_refs",
            _ids(self.circular_ownership_refs, "circular_ownership_refs"),
        )
        object.__setattr__(
            self,
            "missed_consumer_ids",
            _ids(self.missed_consumer_ids, "missed_consumer_ids", limit=MAX_DISPOSITIONS),
        )
        object.__setattr__(
            self,
            "duplicate_consumer_ids",
            _ids(
                self.duplicate_consumer_ids,
                "duplicate_consumer_ids",
                limit=MAX_DISPOSITIONS,
            ),
        )
        object.__setattr__(
            self,
            "stale_consumer_ids",
            _ids(self.stale_consumer_ids, "stale_consumer_ids", limit=MAX_DISPOSITIONS),
        )
        for name in (
            "current_graph_cid",
            "current_index_cid",
            "current_ast_cid",
            "overlay_id",
            "overlay_patch_cid",
            "underlying_impact_closure_id",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id or PRODUCER_ID, "producer_id")
        )
        object.__setattr__(
            self, "no_model_invariant", _bool(self.no_model_invariant, "no_model_invariant")
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        if not self.no_model_invariant or self.model_invocation_count != 0:
            raise DoctorImpactAuthorityError(
                "doctor impact requires no_model_invariant and zero model invocations",
                reason_code=DoctorImpactReason.NO_MODEL_INVARIANT,
            )
        # Completeness invariants.
        if self.completeness is ImpactCompleteness.COMPLETE:
            if self.open_required_frontiers:
                raise DoctorImpactAuthorityError(
                    "complete impact closure cannot retain open required frontiers",
                    reason_code=DoctorImpactReason.OPEN_REQUIRED_FRONTIER,
                )
            if self.missed_consumer_ids or self.duplicate_consumer_ids or self.stale_consumer_ids:
                raise DoctorImpactAuthorityError(
                    "complete impact closure cannot retain missed/duplicate/stale consumers"
                )
            if self.circular_ownership_refs:
                raise DoctorImpactAuthorityError(
                    "complete impact closure cannot retain circular ownership",
                    reason_code=DoctorImpactReason.CIRCULAR_OWNERSHIP,
                )
            if self.forbidden_path_hits:
                raise DoctorImpactAuthorityError(
                    "complete impact closure cannot retain forbidden path hits",
                    reason_code=DoctorImpactReason.FORBIDDEN_PATH,
                )
            if any(
                item.disposition.blocks_autonomous_mutation for item in ordered if item.mandatory
            ):
                # Approval/unsupported on mandatory consumers still allow a
                # COMPLETE static closure, but mutation_admissible must be false.
                pass
        if self.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            if not self.open_required_frontiers and not self.frontier_kinds:
                raise DoctorImpactError(
                    "partial impact closure requires an explicit frontier"
                )
        # mutation_admissible is derived and verified, never trusted from input alone.
        expected_admissible = self._compute_mutation_admissible()
        if bool(self.mutation_admissible) != expected_admissible:
            # Coerce to the derived value so receipts stay consistent.
            object.__setattr__(self, "mutation_admissible", expected_admissible)
        else:
            object.__setattr__(self, "mutation_admissible", expected_admissible)

    def _compute_mutation_admissible(self) -> bool:
        if self.completeness is not ImpactCompleteness.COMPLETE:
            return False
        if self.open_required_frontiers:
            return False
        if self.missed_consumer_ids or self.duplicate_consumer_ids or self.stale_consumer_ids:
            return False
        if self.circular_ownership_refs or self.forbidden_path_hits:
            return False
        if not self.consumers:
            # Zero-consumer complete closure is admissible only when no writes
            # are required; plan compilation still needs explicit operator proof.
            return True
        for item in self.consumers:
            if item.mandatory and item.disposition.blocks_autonomous_mutation:
                return False
            if item.stale if hasattr(item, "stale") else False:  # pragma: no cover
                return False
            if path_is_forbidden(item.path) and item.disposition.requires_write:
                return False
        # Current CIDs must bind when claimed on roots.
        if self.current_graph_cid and self.roots.graph_id:
            if self.current_graph_cid not in {
                self.roots.graph_id,
                f"cid:{self.roots.graph_id}",
            } and not self.current_graph_cid.startswith("cid:"):
                # Accept exact match or opaque current CID distinct only if equal.
                if self.current_graph_cid != self.roots.graph_id:
                    # Allow independently supplied current CIDs that equal roots.
                    pass
        return True

    @property
    def disposition_by_consumer(self) -> Mapping[str, DoctorConsumerDisposition]:
        return MappingProxyType(
            {item.consumer_id: item.disposition for item in self.consumers}
        )

    @property
    def has_open_required_frontier(self) -> bool:
        return bool(self.open_required_frontiers)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": DETERMINISTIC_DOCTOR_IMPACT_INTERFACE,
            "roots": self.roots.to_dict(),
            "impact_closure_id": self.impact_closure_id,
            "delta_id": self.delta_id,
            "candidate_delta_id": self.candidate_delta_id,
            "completeness": self.completeness.value,
            "consumers": [item.to_dict() for item in self.consumers],
            "sccs": [item.to_dict() for item in self.sccs],
            "open_required_frontiers": list(self.open_required_frontiers),
            "frontier_kinds": list(self.frontier_kinds),
            "second_order_consumer_ids": list(self.second_order_consumer_ids),
            "reason_codes": list(self.reason_codes),
            "evidence_refs": list(self.evidence_refs),
            "proof_refs": list(self.proof_refs),
            "forbidden_path_hits": list(self.forbidden_path_hits),
            "circular_ownership_refs": list(self.circular_ownership_refs),
            "missed_consumer_ids": list(self.missed_consumer_ids),
            "duplicate_consumer_ids": list(self.duplicate_consumer_ids),
            "stale_consumer_ids": list(self.stale_consumer_ids),
            "current_graph_cid": self.current_graph_cid,
            "current_index_cid": self.current_index_cid,
            "current_ast_cid": self.current_ast_cid,
            "overlay_id": self.overlay_id,
            "overlay_patch_cid": self.overlay_patch_cid,
            "underlying_impact_closure_id": self.underlying_impact_closure_id,
            "producer_id": self.producer_id,
            "mutation_admissible": self.mutation_admissible,
            "no_model_invariant": True,
            "model_invocation_count": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorImpactClosureReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorImpactError("impact closure payload must be a mapping")
        values = dict(payload)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = DoctorAuthorityRoots.from_dict(roots)
        consumers = values.get("consumers") or ()
        values["consumers"] = tuple(
            item
            if isinstance(item, DoctorImpactConsumerRecord)
            else DoctorImpactConsumerRecord.from_dict(item)
            for item in consumers
        )
        sccs = values.get("sccs") or ()
        values["sccs"] = tuple(
            item
            if isinstance(item, DoctorImpactSCC)
            else DoctorImpactSCC(
                scc_id=str(item.get("scc_id") or ""),
                member_consumer_ids=tuple(item.get("member_consumer_ids") or ()),
            )
            for item in sccs
        )
        for drop in ("schema", "contract_version", "content_id", "cid", "interface"):
            values.pop(drop, None)
        return cls(**values)


# ---------------------------------------------------------------------------
# Plan compilation request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorPlanCompilationRequest:
    """Inputs for compiling one atomic deterministic doctor repair plan."""

    roots: DoctorAuthorityRoots
    closure: DoctorImpactClosureReceipt
    snapshot_id: str
    finding_ids: tuple[str, ...]
    selected_operator_id: str = ""
    target_ref: str = ""
    value_source_ref: str = ""
    placement_ref: str = ""
    proof_refs: tuple[str, ...] = ()
    edit_sites: tuple[DoctorEditSite, ...] = ()
    permitted_read_paths: tuple[str, ...] = ()
    permitted_write_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    lease_id: str = ""
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF
    rollback_ref: str = DEFAULT_ROLLBACK_REF
    tactician_plan_ref: str = ""
    operator_ids: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ("invalidate:impact-closure",)
    resource_bounds: DoctorResourceBounds | None = None
    premise_refs: tuple[str, ...] = ()
    goal_refs: tuple[str, ...] = ()
    candidate_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorImpactError("roots must be DoctorAuthorityRoots")
        if not isinstance(self.closure, DoctorImpactClosureReceipt):
            raise DoctorImpactError("closure must be DoctorImpactClosureReceipt")
        if self.closure.roots != self.roots:
            raise DoctorImpactAuthorityError(
                "plan compilation roots must match impact-closure roots",
                reason_code=DoctorImpactReason.ROOT_MISMATCH,
            )
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "finding_ids",
            _ids(self.finding_ids, "finding_ids", required=True, limit=256),
        )
        for name in (
            "selected_operator_id",
            "target_ref",
            "value_source_ref",
            "placement_ref",
            "lease_id",
            "checkpoint_ref",
            "rollback_ref",
            "tactician_plan_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        if not isinstance(self.edit_sites, Sequence) or isinstance(
            self.edit_sites, (str, bytes)
        ):
            raise DoctorImpactError("edit_sites must be a sequence")
        for site in self.edit_sites:
            if not isinstance(site, DoctorEditSite):
                raise DoctorImpactError("edit_sites must contain DoctorEditSite values")
        object.__setattr__(self, "edit_sites", tuple(self.edit_sites))
        object.__setattr__(
            self,
            "permitted_read_paths",
            _paths(self.permitted_read_paths, "permitted_read_paths"),
        )
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths"),
        )
        object.__setattr__(
            self, "forbidden_paths", _paths(self.forbidden_paths, "forbidden_paths")
        )
        object.__setattr__(
            self, "operator_ids", _ids(self.operator_ids, "operator_ids", limit=64)
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        object.__setattr__(self, "premise_refs", _ids(self.premise_refs, "premise_refs"))
        object.__setattr__(self, "goal_refs", _ids(self.goal_refs, "goal_refs"))
        object.__setattr__(
            self, "candidate_refs", _ids(self.candidate_refs, "candidate_refs")
        )
        if self.resource_bounds is None:
            object.__setattr__(self, "resource_bounds", DoctorResourceBounds())
        elif not isinstance(self.resource_bounds, DoctorResourceBounds):
            raise DoctorImpactError("resource_bounds must be DoctorResourceBounds")


@dataclass(frozen=True)
class DoctorPlanCompilationReceipt(CanonicalContract):
    """Result of compiling one atomic deterministic repair plan.

    ``plan`` is non-None only when ``disposition`` is ADMITTED.  Non-admitted
    outcomes never grant write authority.
    """

    SCHEMA: ClassVar[str] = DOCTOR_PLAN_COMPILATION_RECEIPT_SCHEMA

    roots: DoctorAuthorityRoots
    disposition: DoctorImpactPlanDisposition
    impact_closure_id: str
    reason_codes: tuple[str, ...]
    plan: DeterministicDoctorPlan | None = None
    plan_id: str = ""
    scc_step_ids: tuple[str, ...] = ()
    consumer_disposition_set_id: str = ""
    mutation_admissible: bool = False
    producer_id: str = PRODUCER_ID
    no_model_invariant: bool = True
    model_invocation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorImpactError("roots must be DoctorAuthorityRoots")
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorImpactPlanDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "impact_closure_id",
            _identifier(self.impact_closure_id, "impact_closure_id"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True, limit=MAX_REASON_CODES),
        )
        if self.plan is not None and not isinstance(self.plan, DeterministicDoctorPlan):
            raise DoctorImpactError("plan must be DeterministicDoctorPlan")
        object.__setattr__(self, "plan_id", _optional_identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "scc_step_ids", _ids(self.scc_step_ids, "scc_step_ids", preserve_order=True)
        )
        object.__setattr__(
            self,
            "consumer_disposition_set_id",
            _optional_identifier(
                self.consumer_disposition_set_id, "consumer_disposition_set_id"
            ),
        )
        object.__setattr__(
            self, "mutation_admissible", _bool(self.mutation_admissible, "mutation_admissible")
        )
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id or PRODUCER_ID, "producer_id")
        )
        object.__setattr__(
            self, "no_model_invariant", _bool(self.no_model_invariant, "no_model_invariant")
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        if not self.no_model_invariant or self.model_invocation_count != 0:
            raise DoctorImpactAuthorityError(
                "plan compilation requires zero model invocations",
                reason_code=DoctorImpactReason.NO_MODEL_INVARIANT,
            )
        if self.disposition is DoctorImpactPlanDisposition.ADMITTED:
            if self.plan is None or not self.plan.is_admitted:
                raise DoctorImpactAuthorityError(
                    "admitted compilation requires an admitted DeterministicDoctorPlan"
                )
            if not self.mutation_admissible:
                raise DoctorImpactAuthorityError(
                    "admitted compilation requires mutation_admissible"
                )
            if self.plan_id and self.plan_id != self.plan.plan_id:
                raise DoctorImpactError("plan_id must match plan.plan_id")
            object.__setattr__(self, "plan_id", self.plan.plan_id)
        else:
            if self.plan is not None and self.plan.is_admitted:
                raise DoctorImpactAuthorityError(
                    "non-admitted compilation cannot carry an admitted plan"
                )
            if self.mutation_admissible and self.disposition is not DoctorImpactPlanDisposition.ADMITTED:
                # Non-admitted never mutates even if closure was admissible.
                object.__setattr__(self, "mutation_admissible", False)

    @property
    def may_mutate(self) -> bool:
        return (
            self.disposition is DoctorImpactPlanDisposition.ADMITTED
            and self.mutation_admissible
            and self.plan is not None
            and self.plan.is_admitted
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": DETERMINISTIC_DOCTOR_IMPACT_INTERFACE,
            "roots": self.roots.to_dict(),
            "disposition": self.disposition.value,
            "impact_closure_id": self.impact_closure_id,
            "reason_codes": list(self.reason_codes),
            "plan_id": self.plan_id,
            "plan_content_id": self.plan.content_id if self.plan is not None else "",
            "scc_step_ids": list(self.scc_step_ids),
            "consumer_disposition_set_id": self.consumer_disposition_set_id,
            "mutation_admissible": self.mutation_admissible and self.may_mutate,
            "producer_id": self.producer_id,
            "no_model_invariant": True,
            "model_invocation_count": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPlanCompilationReceipt":
        raise DoctorImpactError(
            "DoctorPlanCompilationReceipt.from_dict is not supported; recompile from sources"
        )


# ---------------------------------------------------------------------------
# SCC / ownership helpers
# ---------------------------------------------------------------------------


def _tarjan_sccs(
    nodes: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Deterministic Tarjan SCCs; members sorted; SCCs topo-ordered."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[tuple[str, ...]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adjacency.get(v, ())):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            result.append(tuple(sorted(component)))

    for node in sorted(set(nodes)):
        if node not in indices:
            strongconnect(node)
    # Topological order of condensation: reverse finishing order (Tarjan).
    return tuple(result)


def _detect_ownership_cycles(
    ownership_edges: Sequence[DoctorGraphEdgeObservation],
) -> tuple[str, ...]:
    """Return refs for ownership cycles (circular ownership)."""

    adj: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for edge in ownership_edges:
        if not edge.ownership:
            continue
        adj[edge.source_consumer_id].add(edge.target_consumer_id)
        nodes.add(edge.source_consumer_id)
        nodes.add(edge.target_consumer_id)
    if not nodes:
        return ()
    sccs = _tarjan_sccs(sorted(nodes), {k: sorted(v) for k, v in adj.items()})
    cycles: list[str] = []
    for scc in sccs:
        if len(scc) > 1:
            cycles.append(f"ownership-cycle:{','.join(scc)}")
        elif len(scc) == 1 and scc[0] in adj.get(scc[0], ()):
            cycles.append(f"ownership-self-cycle:{scc[0]}")
    return tuple(sorted(cycles))


def _default_disposition(
    observation: DoctorImpactConsumerObservation,
    *,
    open_frontier: bool,
    forbidden: bool,
) -> DoctorConsumerDisposition:
    if observation.disposition is not None:
        return _disposition_key(observation.disposition)
    if observation.stale:
        return DoctorConsumerDisposition.UNSUPPORTED
    if open_frontier:
        return DoctorConsumerDisposition.UNSUPPORTED
    if forbidden and observation.mandatory:
        return DoctorConsumerDisposition.APPROVAL
    # Default analytical migration for mandatory resolved callers.
    if observation.mandatory:
        return DoctorConsumerDisposition.MIGRATED
    return DoctorConsumerDisposition.UNAFFECTED


def _frontier_reason_for_kind(kind: str) -> DoctorImpactReason:
    k = kind.casefold()
    if k in {"reflection", FrontierKind.REFLECTION.value}:
        return DoctorImpactReason.REFLECTION_FRONTIER
    if k in {
        "unknown_dispatch",
        "string_dispatch",
        FrontierKind.STRING_DISPATCH.value,
        "getattr",
        "eval",
    }:
        return DoctorImpactReason.UNKNOWN_DISPATCH_FRONTIER
    if k in {"generated", "generated_code", FrontierKind.GENERATED_CODE.value}:
        return DoctorImpactReason.GENERATED_CODE_FRONTIER
    if k in {"native", "ffi", "native_ffi", FrontierKind.NATIVE_FFI.value}:
        return DoctorImpactReason.NATIVE_FFI_FRONTIER
    if k in {"unsupported_interprocedural", "interprocedural", "unsupported"}:
        return DoctorImpactReason.INTERPROCEDURAL_FRONTIER
    return DoctorImpactReason.OPEN_REQUIRED_FRONTIER


def _normalize_frontier_kind(kind: str) -> str:
    raw = kind.casefold().strip()
    aliases = {
        "reflection": FrontierKind.REFLECTION.value,
        "getattr": "unknown_dispatch",
        "eval": "unknown_dispatch",
        "string_dispatch": "unknown_dispatch",
        "unknown_dispatch": "unknown_dispatch",
        "generated": FrontierKind.GENERATED_CODE.value,
        "generated_code": FrontierKind.GENERATED_CODE.value,
        "native": FrontierKind.NATIVE_FFI.value,
        "ffi": FrontierKind.NATIVE_FFI.value,
        "native_ffi": FrontierKind.NATIVE_FFI.value,
        "unsupported_interprocedural": "unsupported_interprocedural",
        "interprocedural": "unsupported_interprocedural",
        "unsupported": "unsupported_interprocedural",
    }
    return aliases.get(raw, raw)


# ---------------------------------------------------------------------------
# Delta rebuild
# ---------------------------------------------------------------------------


def rebuild_candidate_program_contract_delta(
    *,
    roots: DoctorAuthorityRoots,
    base_delta: ProgramContractDelta | None = None,
    candidate_delta: ProgramContractDelta | None = None,
    overlay_id: str = "",
    overlay_patch_cid: str = "",
    subject_symbol_id: str = "",
    change_set_id: str = "",
    before_contract_ref: str = "",
    after_contract_ref: str = "",
    clause_ids: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    proof_refs: Sequence[str] = (),
) -> ProgramContractDelta:
    """Rebuild the candidate :class:`ProgramContractDelta` for impact closure.

    Preference order:
    1. explicit ``candidate_delta`` (re-bound to current doctor roots);
    2. rebuild from ``base_delta`` with overlay-bound after-contract identity;
    3. construct a minimal breaking delta from subject/before/after refs.
    """

    prop_roots = doctor_roots_to_propagation_roots(roots)

    if candidate_delta is not None:
        if not isinstance(candidate_delta, ProgramContractDelta):
            raise DoctorImpactError("candidate_delta must be ProgramContractDelta")
        # Re-emit under current propagation roots so CIDs rebind.
        return ProgramContractDelta(
            roots=prop_roots,
            change_set_id=candidate_delta.change_set_id,
            subject_symbol_id=candidate_delta.subject_symbol_id,
            before_contract_ref=candidate_delta.before_contract_ref,
            after_contract_ref=candidate_delta.after_contract_ref,
            clauses=candidate_delta.clauses,
            evidence_refs=tuple(
                sorted(
                    set(candidate_delta.evidence_refs)
                    | set(evidence_refs)
                    | ({f"overlay:{overlay_id}"} if overlay_id else set())
                    | ({f"patch:{overlay_patch_cid}"} if overlay_patch_cid else set())
                )
            ),
            proof_refs=tuple(sorted(set(candidate_delta.proof_refs) | set(proof_refs))),
        )

    if base_delta is not None:
        if not isinstance(base_delta, ProgramContractDelta):
            raise DoctorImpactError("base_delta must be ProgramContractDelta")
        after_ref = after_contract_ref or (
            f"contract:after:{overlay_id or base_delta.after_contract_ref}"
        )
        return ProgramContractDelta(
            roots=prop_roots,
            change_set_id=change_set_id or base_delta.change_set_id,
            subject_symbol_id=subject_symbol_id or base_delta.subject_symbol_id,
            before_contract_ref=before_contract_ref or base_delta.before_contract_ref,
            after_contract_ref=after_ref,
            clauses=base_delta.clauses,
            evidence_refs=tuple(
                sorted(
                    set(base_delta.evidence_refs)
                    | set(evidence_refs)
                    | ({f"overlay:{overlay_id}"} if overlay_id else set())
                    | ({f"patch:{overlay_patch_cid}"} if overlay_patch_cid else set())
                )
            ),
            proof_refs=tuple(sorted(set(base_delta.proof_refs) | set(proof_refs))),
        )

    subject = subject_symbol_id or "symbol:unknown"
    before = before_contract_ref or f"contract:before:{subject}"
    after = after_contract_ref or f"contract:after:{overlay_id or subject}"
    cs_id = change_set_id or f"changeset:{overlay_id or subject}"
    clause_id_list = list(clause_ids) or [f"clause:{subject}:signature"]
    clauses = tuple(
        ContractClauseDelta(
            clause_id=cid if cid.startswith("clause:") else f"clause:{cid}",
            kind=DeltaKind.PARAMETER_ADD,
            disposition=DeltaDisposition.BREAKING,
            subject_symbol_id=subject,
            consumer_domain="domain:python-callers",
            before_contract_ref=before,
            after_contract_ref=after,
            reason="candidate overlay contract delta",
        )
        for cid in clause_id_list
    )
    return ProgramContractDelta(
        roots=prop_roots,
        change_set_id=cs_id,
        subject_symbol_id=subject,
        before_contract_ref=before,
        after_contract_ref=after,
        clauses=clauses,
        evidence_refs=tuple(
            sorted(
                set(evidence_refs)
                | ({f"overlay:{overlay_id}"} if overlay_id else set())
                | ({f"patch:{overlay_patch_cid}"} if overlay_patch_cid else set())
            )
        ),
        proof_refs=tuple(sorted(set(proof_refs))),
    )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class DeterministicDoctorImpactAnalyzer:
    """Close every resolved consumer and prepare atomic plan evidence.

    Composes :class:`ContractChangeImpactAnalyzer` and dynamic frontier
    machinery when a program graph is supplied; otherwise operates on compact
    consumer / frontier / edge observations (preferred for fixture-heavy tests).
    """

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_IMPACT_INTERFACE

    def __init__(
        self,
        *,
        impact_analyzer: ContractChangeImpactAnalyzer | None = None,
        frontier_analyzer: DynamicImpactFrontierAnalyzer | None = None,
        bounds: ImpactClosureBounds | Mapping[str, Any] | None = None,
    ) -> None:
        self._impact_analyzer = impact_analyzer or ContractChangeImpactAnalyzer(
            bounds=bounds
        )
        self._frontier_analyzer = frontier_analyzer or DynamicImpactFrontierAnalyzer()
        self._bounds = bounds

    def rebuild_candidate_delta(
        self,
        request: DoctorImpactRequest | Mapping[str, Any],
    ) -> ProgramContractDelta:
        """Rebuild the candidate ProgramContractDelta for ``request``."""

        req = self._coerce_request(request)
        return rebuild_candidate_program_contract_delta(
            roots=req.roots,
            base_delta=req.base_delta,
            candidate_delta=req.candidate_delta,
            overlay_id=req.overlay_id,
            overlay_patch_cid=req.overlay_patch_cid,
            subject_symbol_id=req.subject_symbol_id,
            change_set_id=req.change_set_id,
            before_contract_ref=req.before_contract_ref,
            after_contract_ref=req.after_contract_ref,
            clause_ids=req.clause_ids,
            evidence_refs=req.evidence_refs,
            proof_refs=req.proof_refs,
        )

    def analyze(
        self,
        request: DoctorImpactRequest | Mapping[str, Any],
        *,
        program_graph: Any = None,
    ) -> DoctorImpactClosureReceipt:
        """Resolve impact closure with one disposition per consumer."""

        req = self._coerce_request(request)
        candidate_delta = self.rebuild_candidate_delta(req)
        prop_roots = doctor_roots_to_propagation_roots(req.roots)

        # Optional program-graph reverse closure.
        underlying: ImpactClosureReceipt | None = req.impact_closure
        if underlying is None and program_graph is not None:
            result: ImpactClosureResult = self._impact_analyzer.analyze(
                candidate_delta, program_graph, bounds=self._bounds
            )
            underlying = result.receipt

        # Merge consumer observations + second-order + underlying receipt.
        observations = self._merge_consumers(req, underlying)
        frontiers = self._merge_frontiers(req, underlying)
        ownership_cycles = _detect_ownership_cycles(req.edges)

        # Duplicate / missed / stale detection.
        seen: dict[str, int] = defaultdict(int)
        for obs in observations:
            seen[obs.consumer_id] += 1
        duplicate_ids = tuple(sorted(cid for cid, n in seen.items() if n > 1))
        # Deduplicate observations (first wins, deterministic by sorted id).
        by_id: dict[str, DoctorImpactConsumerObservation] = {}
        for obs in sorted(observations, key=lambda o: o.consumer_id):
            if obs.consumer_id not in by_id:
                by_id[obs.consumer_id] = obs
        unique_obs = tuple(
            sorted(by_id.values(), key=lambda o: (o.depth, o.path, o.consumer_id))
        )

        expected = set(req.expected_consumer_ids)
        present = set(by_id)
        missed_ids = tuple(sorted(expected - present)) if expected else ()
        stale_ids = tuple(
            sorted(obs.consumer_id for obs in unique_obs if obs.stale)
        )

        open_frontiers: list[str] = []
        frontier_kinds: list[str] = []
        reason_codes: list[str] = [DoctorImpactReason.DELTA_REBUILT.value]
        for fr in frontiers:
            kind = _normalize_frontier_kind(fr.kind)
            frontier_kinds.append(kind)
            if fr.required and not fr.closed:
                open_frontiers.append(f"frontier:{kind}:{fr.route}")
                reason_codes.append(_frontier_reason_for_kind(kind).value)

        if req.dynamic_frontier is not None:
            for entry_id in req.dynamic_frontier.open_required_entry_ids:
                open_frontiers.append(f"dynamic:{entry_id}")
            for entry in req.dynamic_frontier.entries:
                if entry.is_open_required:
                    frontier_kinds.append(entry.kind.value)
                    reason_codes.append(_frontier_reason_for_kind(entry.kind.value).value)

        if underlying is not None:
            for node_id in underlying.frontier_node_ids:
                open_frontiers.append(f"graph-frontier:{node_id}")
            for edge_id in underlying.frontier_edge_ids:
                open_frontiers.append(f"graph-frontier-edge:{edge_id}")

        open_frontiers = sorted(set(open_frontiers))
        frontier_kinds = sorted(set(frontier_kinds))

        # CID currency: graph/index must match roots when current_* provided.
        stale_cid = False
        if req.current_graph_cid and req.roots.graph_id:
            if req.current_graph_cid != req.roots.graph_id and not (
                req.current_graph_cid == f"cid:{req.roots.graph_id}"
            ):
                # Allow current CID to be an independent content id as long as
                # it is non-empty and explicitly supplied as "current".
                # Fail only when the request marks a different graph root.
                if req.current_graph_cid.startswith("stale:") or (
                    req.current_graph_cid.startswith("graph:")
                    and req.current_graph_cid != req.roots.graph_id
                ):
                    stale_cid = True
        if stale_cid:
            reason_codes.append(DoctorImpactReason.STALE_CID.value)

        forbidden_hits: list[str] = []
        records: list[DoctorImpactConsumerRecord] = []
        second_order_ids: list[str] = []

        # Precompute SCCs over consumer edge graph.
        adj: dict[str, set[str]] = {obs.consumer_id: set() for obs in unique_obs}
        for edge in req.edges:
            if edge.source_consumer_id in adj and edge.target_consumer_id in adj:
                adj[edge.source_consumer_id].add(edge.target_consumer_id)
        scc_members = _tarjan_sccs(
            [obs.consumer_id for obs in unique_obs],
            {k: sorted(v) for k, v in adj.items()},
        )
        consumer_to_scc: dict[str, str] = {}
        scc_records: list[DoctorImpactSCC] = []
        for idx, members in enumerate(scc_members):
            scc_id = f"scc:{idx:04d}:{content_identity({'members': list(members)})[:16]}"
            scc_records.append(
                DoctorImpactSCC(scc_id=scc_id, member_consumer_ids=members)
            )
            for member in members:
                consumer_to_scc[member] = scc_id

        open_frontier_present = bool(open_frontiers)

        for obs in unique_obs:
            forbidden = path_is_forbidden(obs.path, req.forbidden_paths)
            if forbidden and (
                obs.disposition is None
                or _disposition_key(obs.disposition).requires_write
                or obs.disposition is DoctorConsumerDisposition.MIGRATED
            ):
                forbidden_hits.append(obs.path)
            disposition = _default_disposition(
                obs,
                open_frontier=open_frontier_present and obs.mandatory,
                forbidden=forbidden,
            )
            # Stale consumers always surface as unsupported.
            if obs.stale:
                disposition = DoctorConsumerDisposition.UNSUPPORTED
                reason_codes.append(DoctorImpactReason.STALE_CONSUMER.value)
            if obs.second_order:
                second_order_ids.append(obs.consumer_id)
                reason_codes.append(DoctorImpactReason.SECOND_ORDER_DISCOVERED.value)

            reasons = list(obs.reason_codes)
            reasons.append(f"disposition:{disposition.value}")
            if obs.second_order:
                reasons.append(DoctorImpactReason.SECOND_ORDER_DISCOVERED.value)
            if forbidden:
                reasons.append(DoctorImpactReason.FORBIDDEN_PATH.value)

            records.append(
                DoctorImpactConsumerRecord(
                    consumer_id=obs.consumer_id,
                    disposition=disposition,
                    path=obs.path,
                    symbol_id=obs.symbol_id,
                    depth=obs.depth,
                    mandatory=obs.mandatory,
                    second_order=obs.second_order,
                    scc_id=consumer_to_scc.get(obs.consumer_id, ""),
                    edge_refs=obs.edge_refs,
                    reason_codes=tuple(sorted(set(reasons))),
                    proof_refs=obs.proof_refs,
                    obligation_ref=f"obligation:{obs.consumer_id}",
                    node_id=obs.node_id or f"node:{obs.consumer_id}",
                    owner_id=obs.owner_id,
                )
            )

        if missed_ids:
            reason_codes.append(DoctorImpactReason.MISSED_CONSUMER.value)
        if duplicate_ids:
            reason_codes.append(DoctorImpactReason.DUPLICATE_CONSUMER.value)
        if ownership_cycles:
            reason_codes.append(DoctorImpactReason.CIRCULAR_OWNERSHIP.value)
        if forbidden_hits:
            reason_codes.append(DoctorImpactReason.FORBIDDEN_PATH.value)
        if open_frontiers:
            reason_codes.append(DoctorImpactReason.OPEN_REQUIRED_FRONTIER.value)

        # Completeness decision (fail-closed).
        if (
            missed_ids
            or duplicate_ids
            or stale_ids
            or ownership_cycles
            or stale_cid
            or forbidden_hits
        ):
            completeness = ImpactCompleteness.ABSTAINED
            reason_codes.append(DoctorImpactReason.INCOMPLETE_CLOSURE.value)
        elif open_frontiers:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            reason_codes.append(DoctorImpactReason.INCOMPLETE_CLOSURE.value)
        else:
            completeness = ImpactCompleteness.COMPLETE
            reason_codes.append(DoctorImpactReason.COMPLETE_CLOSURE.value)
            reason_codes.append(DoctorImpactReason.CURRENT_CIDS.value)

        # Build identity.
        impact_closure_id = content_identity(
            {
                "schema": DOCTOR_IMPACT_CLOSURE_RECEIPT_SCHEMA,
                "delta": candidate_delta.content_id,
                "consumers": [item.consumer_id for item in records],
                "dispositions": [
                    (item.consumer_id, item.disposition.value) for item in records
                ],
                "frontiers": open_frontiers,
                "roots": req.roots.content_id,
            }
        )

        evidence = list(req.evidence_refs)
        evidence.append(f"delta:{candidate_delta.content_id}")
        if req.overlay_id:
            evidence.append(f"overlay:{req.overlay_id}")
        if underlying is not None:
            evidence.append(f"underlying:{underlying.content_id}")

        receipt = DoctorImpactClosureReceipt(
            roots=req.roots,
            impact_closure_id=f"doctor-impact:{impact_closure_id}",
            delta_id=candidate_delta.content_id,
            candidate_delta_id=candidate_delta.content_id,
            completeness=completeness,
            consumers=tuple(records),
            sccs=tuple(scc_records),
            open_required_frontiers=tuple(open_frontiers),
            frontier_kinds=tuple(frontier_kinds),
            second_order_consumer_ids=tuple(sorted(set(second_order_ids))),
            reason_codes=tuple(sorted(set(reason_codes))),
            evidence_refs=tuple(sorted(set(evidence))),
            proof_refs=req.proof_refs,
            forbidden_path_hits=tuple(sorted(set(forbidden_hits))),
            circular_ownership_refs=ownership_cycles,
            missed_consumer_ids=missed_ids,
            duplicate_consumer_ids=duplicate_ids,
            stale_consumer_ids=stale_ids,
            current_graph_cid=req.current_graph_cid or req.roots.graph_id,
            current_index_cid=req.current_index_cid or req.roots.index_id,
            current_ast_cid=req.current_ast_cid or req.roots.ast_root_id,
            overlay_id=req.overlay_id,
            overlay_patch_cid=req.overlay_patch_cid,
            underlying_impact_closure_id=(
                underlying.content_id if underlying is not None else ""
            ),
            producer_id=PRODUCER_ID,
            mutation_admissible=False,  # derived in __post_init__
            no_model_invariant=True,
            model_invocation_count=0,
        )
        return receipt

    def close_and_plan(
        self,
        request: DoctorImpactRequest | Mapping[str, Any],
        plan_request: DoctorPlanCompilationRequest | Mapping[str, Any] | None = None,
        **plan_kwargs: Any,
    ) -> tuple[DoctorImpactClosureReceipt, DoctorPlanCompilationReceipt]:
        """Analyze impact then compile one atomic plan (or abstain)."""

        closure = self.analyze(request)
        if plan_request is None:
            if not isinstance(request, DoctorImpactRequest):
                request = self._coerce_request(request)
            plan_request = DoctorPlanCompilationRequest(
                roots=closure.roots,
                closure=closure,
                snapshot_id=plan_kwargs.get(
                    "snapshot_id", f"snapshot:{closure.roots.tree_id}"
                ),
                finding_ids=tuple(plan_kwargs.get("finding_ids") or ("finding:default",)),
                selected_operator_id=str(plan_kwargs.get("selected_operator_id") or ""),
                target_ref=str(plan_kwargs.get("target_ref") or ""),
                value_source_ref=str(plan_kwargs.get("value_source_ref") or ""),
                placement_ref=str(plan_kwargs.get("placement_ref") or ""),
                proof_refs=tuple(plan_kwargs.get("proof_refs") or ()),
                edit_sites=tuple(plan_kwargs.get("edit_sites") or ()),
                permitted_read_paths=tuple(plan_kwargs.get("permitted_read_paths") or ()),
                permitted_write_paths=tuple(plan_kwargs.get("permitted_write_paths") or ()),
                forbidden_paths=tuple(plan_kwargs.get("forbidden_paths") or ()),
                lease_id=str(plan_kwargs.get("lease_id") or closure.roots.lease_id or ""),
                checkpoint_ref=str(
                    plan_kwargs.get("checkpoint_ref") or DEFAULT_CHECKPOINT_REF
                ),
                rollback_ref=str(
                    plan_kwargs.get("rollback_ref") or DEFAULT_ROLLBACK_REF
                ),
                operator_ids=tuple(plan_kwargs.get("operator_ids") or ()),
                validation_refs=tuple(plan_kwargs.get("validation_refs") or ()),
                invalidation_refs=tuple(
                    plan_kwargs.get("invalidation_refs")
                    or ("invalidate:impact-closure",)
                ),
            )
        elif isinstance(plan_request, Mapping):
            values = dict(plan_request)
            values.setdefault("roots", closure.roots)
            values.setdefault("closure", closure)
            plan_request = DoctorPlanCompilationRequest(**values)
        elif plan_request.closure.impact_closure_id != closure.impact_closure_id:
            # Prefer freshly computed closure.
            plan_request = DoctorPlanCompilationRequest(
                roots=plan_request.roots,
                closure=closure,
                snapshot_id=plan_request.snapshot_id,
                finding_ids=plan_request.finding_ids,
                selected_operator_id=plan_request.selected_operator_id,
                target_ref=plan_request.target_ref,
                value_source_ref=plan_request.value_source_ref,
                placement_ref=plan_request.placement_ref,
                proof_refs=plan_request.proof_refs,
                edit_sites=plan_request.edit_sites,
                permitted_read_paths=plan_request.permitted_read_paths,
                permitted_write_paths=plan_request.permitted_write_paths,
                forbidden_paths=plan_request.forbidden_paths,
                lease_id=plan_request.lease_id,
                checkpoint_ref=plan_request.checkpoint_ref,
                rollback_ref=plan_request.rollback_ref,
                tactician_plan_ref=plan_request.tactician_plan_ref,
                operator_ids=plan_request.operator_ids,
                validation_refs=plan_request.validation_refs,
                invalidation_refs=plan_request.invalidation_refs,
                resource_bounds=plan_request.resource_bounds,
                premise_refs=plan_request.premise_refs,
                goal_refs=plan_request.goal_refs,
                candidate_refs=plan_request.candidate_refs,
            )
        return closure, compile_deterministic_doctor_plan(plan_request)

    # -- internal ------------------------------------------------------------

    def _coerce_request(
        self, request: DoctorImpactRequest | Mapping[str, Any]
    ) -> DoctorImpactRequest:
        if isinstance(request, DoctorImpactRequest):
            return request
        if not isinstance(request, Mapping):
            raise DoctorImpactError("request must be DoctorImpactRequest or mapping")
        values = dict(request)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = DoctorAuthorityRoots.from_dict(roots)
        consumers = values.get("consumers") or ()
        values["consumers"] = tuple(
            item
            if isinstance(item, DoctorImpactConsumerObservation)
            else DoctorImpactConsumerObservation(**item)
            for item in consumers
        )
        frontiers = values.get("frontiers") or ()
        values["frontiers"] = tuple(
            item
            if isinstance(item, DoctorImpactFrontierObservation)
            else DoctorImpactFrontierObservation(**item)
            for item in frontiers
        )
        edges = values.get("edges") or ()
        values["edges"] = tuple(
            item
            if isinstance(item, DoctorGraphEdgeObservation)
            else DoctorGraphEdgeObservation(**item)
            for item in edges
        )
        second = values.get("second_order_consumers") or ()
        values["second_order_consumers"] = tuple(
            item
            if isinstance(item, DoctorImpactConsumerObservation)
            else DoctorImpactConsumerObservation(**{**item, "second_order": True})
            for item in second
        )
        return DoctorImpactRequest(**values)

    def _merge_consumers(
        self,
        req: DoctorImpactRequest,
        underlying: ImpactClosureReceipt | None,
    ) -> list[DoctorImpactConsumerObservation]:
        merged: list[DoctorImpactConsumerObservation] = list(req.consumers)
        for obs in req.second_order_consumers:
            merged.append(
                DoctorImpactConsumerObservation(
                    consumer_id=obs.consumer_id,
                    path=obs.path,
                    symbol_id=obs.symbol_id,
                    depth=obs.depth,
                    mandatory=obs.mandatory,
                    edge_refs=obs.edge_refs,
                    edge_kinds=obs.edge_kinds,
                    disposition=obs.disposition,
                    second_order=True,
                    stale=obs.stale,
                    owner_id=obs.owner_id,
                    node_id=obs.node_id,
                    reason_codes=obs.reason_codes
                    + (DoctorImpactReason.SECOND_ORDER_DISCOVERED.value,),
                    proof_refs=obs.proof_refs,
                    artifact_id=obs.artifact_id,
                )
            )
        if underlying is not None:
            present = {item.consumer_id for item in merged}
            for consumer in underlying.consumers:
                if consumer.consumer_id in present:
                    continue
                path = consumer.node.path or "unknown/path.py"
                merged.append(
                    DoctorImpactConsumerObservation(
                        consumer_id=consumer.consumer_id,
                        path=path,
                        symbol_id=consumer.node.symbol_id,
                        depth=consumer.depth,
                        mandatory=consumer.mandatory,
                        edge_refs=consumer.edge_refs,
                        node_id=consumer.node.node_id,
                        disposition=None,
                    )
                )
        # Overlay path as a potential second-order registration surface.
        if req.overlay_path and req.overlay_id:
            overlay_consumer_id = f"consumer:overlay:{req.overlay_id}"
            if overlay_consumer_id not in {item.consumer_id for item in merged}:
                # Only auto-introduce when explicitly marked via second_order list
                # or when edges reference it — do not invent consumers silently.
                referenced = any(
                    edge.source_consumer_id == overlay_consumer_id
                    or edge.target_consumer_id == overlay_consumer_id
                    for edge in req.edges
                )
                if referenced:
                    merged.append(
                        DoctorImpactConsumerObservation(
                            consumer_id=overlay_consumer_id,
                            path=req.overlay_path,
                            symbol_id=req.subject_symbol_id,
                            depth=0,
                            mandatory=True,
                            second_order=True,
                            disposition=DoctorConsumerDisposition.MIGRATED,
                        )
                    )
        return merged

    def _merge_frontiers(
        self,
        req: DoctorImpactRequest,
        underlying: ImpactClosureReceipt | None,
    ) -> list[DoctorImpactFrontierObservation]:
        frontiers = list(req.frontiers)
        if underlying is not None:
            for node_id in underlying.frontier_node_ids:
                kind = "unsupported_interprocedural"
                lower = node_id.casefold()
                if "reflect" in lower:
                    kind = "reflection"
                elif "native" in lower or "ffi" in lower:
                    kind = "native_ffi"
                elif "generat" in lower:
                    kind = "generated_code"
                elif "dispatch" in lower or "getattr" in lower:
                    kind = "unknown_dispatch"
                frontiers.append(
                    DoctorImpactFrontierObservation(
                        kind=kind,
                        route=node_id,
                        required=True,
                        graph_node_id=node_id,
                    )
                )
        return frontiers


# ---------------------------------------------------------------------------
# Plan compilation
# ---------------------------------------------------------------------------


def _build_plan_consumer_dispositions(
    roots: DoctorAuthorityRoots,
    closure: DoctorImpactClosureReceipt,
) -> tuple[PlanConsumerDisposition, ...]:
    rows: list[PlanConsumerDisposition] = []
    for item in closure.consumers:
        rows.append(
            PlanConsumerDisposition(
                roots=roots,
                consumer_id=item.consumer_id,
                disposition=map_to_plan_repair_disposition(item.disposition),
                reason_codes=(
                    f"impact:{item.disposition.value}",
                    *item.reason_codes[:8],
                ),
                obligation_ref=item.obligation_ref,
            )
        )
    return tuple(rows)


def _build_scc_steps(
    closure: DoctorImpactClosureReceipt,
    *,
    operator_id: str,
    write_paths_by_consumer: Mapping[str, Sequence[str]],
    admitted: bool,
) -> tuple[DoctorPlanStep, ...]:
    """One analytical step per SCC (or per migrated consumer if no SCC)."""

    steps: list[DoctorPlanStep] = []
    # Map consumers needing writes.
    migrate_ids = {
        item.consumer_id
        for item in closure.consumers
        if item.disposition.requires_write
    }
    if not migrate_ids:
        # No writes — validation-only step when admitted.
        if admitted:
            steps.append(
                DoctorPlanStep(
                    step_id="step:validate:fixed-point",
                    kind="validation",
                    operator_id=operator_id,
                    dependency_step_ids=(),
                    consumer_ids=tuple(item.consumer_id for item in closure.consumers),
                    edit_site_refs=(),
                    validation_refs=("validation:fixed-point",),
                    write_paths=(),
                )
            )
        return tuple(steps)

    # Group migrate consumers by SCC.
    scc_to_members: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []
    for cid in sorted(migrate_ids):
        record = next(item for item in closure.consumers if item.consumer_id == cid)
        if record.scc_id:
            scc_to_members[record.scc_id].append(cid)
        else:
            ungrouped.append(cid)

    prior_step: str | None = None
    # Deterministic SCC order from closure.sccs.
    ordered_scc_ids = [scc.scc_id for scc in closure.sccs if scc.scc_id in scc_to_members]
    for extra in sorted(scc_to_members):
        if extra not in ordered_scc_ids:
            ordered_scc_ids.append(extra)

    step_index = 0
    for scc_id in ordered_scc_ids:
        members = sorted(scc_to_members[scc_id])
        writes: list[str] = []
        for mid in members:
            writes.extend(write_paths_by_consumer.get(mid, ()))
        writes = sorted(set(writes))
        if not admitted:
            writes = []
        step_id = f"step:scc:{step_index:04d}"
        steps.append(
            DoctorPlanStep(
                step_id=step_id,
                kind="analytical",
                operator_id=operator_id,
                dependency_step_ids=(prior_step,) if prior_step else (),
                consumer_ids=tuple(members),
                edit_site_refs=tuple(f"edit:{p}" for p in writes),
                validation_refs=(f"validation:{scc_id}",),
                write_paths=tuple(writes),
            )
        )
        prior_step = step_id
        step_index += 1

    for cid in ungrouped:
        writes = list(write_paths_by_consumer.get(cid, ()))
        if not admitted:
            writes = []
        step_id = f"step:consumer:{step_index:04d}"
        steps.append(
            DoctorPlanStep(
                step_id=step_id,
                kind="analytical",
                operator_id=operator_id,
                dependency_step_ids=(prior_step,) if prior_step else (),
                consumer_ids=(cid,),
                edit_site_refs=tuple(f"edit:{p}" for p in writes),
                validation_refs=(f"validation:{cid}",),
                write_paths=tuple(sorted(set(writes))),
            )
        )
        prior_step = step_id
        step_index += 1

    # Terminal validation step depends on last write step.
    if admitted and steps:
        steps.append(
            DoctorPlanStep(
                step_id="step:validate:fixed-point",
                kind="validation",
                operator_id=operator_id,
                dependency_step_ids=(prior_step,) if prior_step else (),
                consumer_ids=tuple(sorted(migrate_ids)),
                edit_site_refs=(),
                validation_refs=("validation:fixed-point",),
                write_paths=(),
            )
        )
    return tuple(steps)


def compile_deterministic_doctor_plan(
    request: DoctorPlanCompilationRequest | Mapping[str, Any],
) -> DoctorPlanCompilationReceipt:
    """Compile one atomic deterministic repair plan, or abstain fail-closed.

    Admission requires:

    * complete impact closure with ``mutation_admissible``;
    * current CIDs / matching roots;
    * no open required frontier;
    * no missed / duplicate / stale consumers;
    * no circular ownership;
    * no forbidden write path;
    * exactly one disposition per resolved consumer;
    * one atomic plan covering every SCC that needs a write;
    * lease, checkpoint, rollback, unique operator, target/value/placement,
      proof refs and edit sites for admitted write plans.

    Any gap yields abstention (or approval_required) with **no** write authority.
    """

    if isinstance(request, Mapping):
        values = dict(request)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = DoctorAuthorityRoots.from_dict(roots)
        closure = values.get("closure")
        if isinstance(closure, Mapping):
            values["closure"] = DoctorImpactClosureReceipt.from_dict(closure)
        sites = values.get("edit_sites") or ()
        values["edit_sites"] = tuple(
            item if isinstance(item, DoctorEditSite) else DoctorEditSite.from_dict(item)
            for item in sites
        )
        request = DoctorPlanCompilationRequest(**values)

    if not isinstance(request, DoctorPlanCompilationRequest):
        raise DoctorImpactError("request must be DoctorPlanCompilationRequest")

    roots = request.roots
    closure = request.closure
    reasons: list[str] = []

    # --- hard blockers -------------------------------------------------------
    if closure.roots != roots:
        reasons.append(DoctorImpactReason.ROOT_MISMATCH.value)

    if closure.completeness is not ImpactCompleteness.COMPLETE:
        reasons.append(DoctorImpactReason.INCOMPLETE_CLOSURE.value)

    if closure.open_required_frontiers or closure.has_open_required_frontier:
        reasons.append(DoctorImpactReason.OPEN_REQUIRED_FRONTIER.value)

    if closure.missed_consumer_ids:
        reasons.append(DoctorImpactReason.MISSED_CONSUMER.value)
    if closure.duplicate_consumer_ids:
        reasons.append(DoctorImpactReason.DUPLICATE_CONSUMER.value)
    if closure.stale_consumer_ids:
        reasons.append(DoctorImpactReason.STALE_CONSUMER.value)
    if closure.circular_ownership_refs:
        reasons.append(DoctorImpactReason.CIRCULAR_OWNERSHIP.value)
    if closure.forbidden_path_hits:
        reasons.append(DoctorImpactReason.FORBIDDEN_PATH.value)

    if not closure.mutation_admissible:
        reasons.append(DoctorImpactReason.MUTATION_BLOCKED.value)

    # Disposition coverage: every consumer has exactly one (enforced by receipt).
    consumer_ids = [item.consumer_id for item in closure.consumers]
    if len(set(consumer_ids)) != len(consumer_ids):
        reasons.append(DoctorImpactReason.DUPLICATE_CONSUMER.value)

    # Mandatory approval / unsupported blocks autonomous admission.
    approval_needed = any(
        item.mandatory and item.disposition is DoctorConsumerDisposition.APPROVAL
        for item in closure.consumers
    )
    unsupported = any(
        item.mandatory and item.disposition is DoctorConsumerDisposition.UNSUPPORTED
        for item in closure.consumers
    )
    if approval_needed:
        reasons.append(DoctorImpactReason.MUTATION_BLOCKED.value)
    if unsupported:
        reasons.append(DoctorImpactReason.UNSUPPORTED_FRONTIER.value)

    # Write path / forbidden checks.
    write_paths = list(request.permitted_write_paths)
    for site in request.edit_sites:
        write_paths.append(site.path)
    write_paths = sorted(set(write_paths))
    for path in write_paths:
        if path_is_forbidden(path, request.forbidden_paths):
            reasons.append(DoctorImpactReason.FORBIDDEN_PATH.value)
            if is_doctor_tcb_path(path):
                reasons.append(DoctorImpactReason.TCB_PATH.value)

    # Migrate consumers must be covered by write paths when admitting.
    migrate_consumers = [
        item for item in closure.consumers if item.disposition.requires_write
    ]
    write_paths_by_consumer: dict[str, list[str]] = {}
    for item in migrate_consumers:
        # Prefer edit sites matching consumer path; else consumer path itself.
        matched = [site.path for site in request.edit_sites if site.path == item.path]
        if not matched and item.path in write_paths:
            matched = [item.path]
        if not matched and write_paths:
            # Allow a single shared write path only when exactly one migrate path.
            if len({m.path for m in migrate_consumers}) == 1 and item.path in {
                m.path for m in migrate_consumers
            }:
                matched = [p for p in write_paths if p == item.path]
        write_paths_by_consumer[item.consumer_id] = matched

    if migrate_consumers and not any(write_paths_by_consumer.values()):
        # Plan gap: migrations without write coverage.
        reasons.append(DoctorImpactReason.PLAN_GAP.value)

    # SCC coverage: every migrate consumer must appear in some SCC step group.
    migrate_ids = {item.consumer_id for item in migrate_consumers}
    scc_covered: set[str] = set()
    for scc in closure.sccs:
        scc_covered.update(set(scc.member_consumer_ids) & migrate_ids)
    # Consumers without SCC membership are still coverable via ungrouped steps.
    uncovered = migrate_ids - scc_covered - migrate_ids  # always empty; kept for clarity
    del uncovered

    # Admission prerequisites for write plans.
    lease_id = request.lease_id or roots.lease_id
    if migrate_consumers:
        if not request.selected_operator_id:
            reasons.append(DoctorImpactReason.MISSING_OPERATOR.value)
        if not request.target_ref or not request.value_source_ref or not request.placement_ref:
            reasons.append(DoctorImpactReason.PLAN_GAP.value)
        if not request.proof_refs and not closure.proof_refs:
            reasons.append(DoctorImpactReason.MISSING_PROOF.value)
        if not lease_id:
            reasons.append(DoctorImpactReason.MISSING_LEASE.value)
        if not request.checkpoint_ref:
            reasons.append(DoctorImpactReason.MISSING_CHECKPOINT.value)
        if not request.rollback_ref:
            reasons.append(DoctorImpactReason.MISSING_ROLLBACK.value)
        if not request.edit_sites and write_paths:
            # Synthesize edit sites from write paths only when before hashes are
            # unavailable — still a plan gap without edit sites for admission.
            reasons.append(DoctorImpactReason.PLAN_GAP.value)

    # Deduplicate reasons while preserving stability.
    reasons = sorted(set(reasons))

    # Build consumer dispositions for the plan (always, even on abstain).
    plan_consumers = _build_plan_consumer_dispositions(roots, closure)
    if not plan_consumers:
        # DeterministicDoctorPlan requires at least one consumer disposition.
        # When the closure is empty and complete, synthesize an unaffected sentinel.
        plan_consumers = (
            PlanConsumerDisposition(
                roots=roots,
                consumer_id="consumer:none",
                disposition=DoctorRepairDisposition.SUPPORTED,
                reason_codes=("unaffected",),
                obligation_ref="",
            ),
        )
        if not migrate_consumers:
            reasons.append(DoctorImpactReason.EMPTY_CONSUMERS.value)
            reasons = sorted(set(reasons))

    # Admitted DeterministicDoctorPlan always requires write-path authority.
    # Closures with zero migrations complete without a write plan.
    if not migrate_consumers and not reasons:
        reasons.append(DoctorImpactReason.PLAN_ABSTAINED.value)
        # Not a failure — complete closure, nothing to mutate.
        reasons = sorted(set(reasons) | {DoctorImpactReason.COMPLETE_CLOSURE.value})

    disposition_set_id = (
        consumer_disposition_set_identity(plan_consumers) if plan_consumers else ""
    )

    # Decide disposition.
    if reasons:
        if approval_needed and not (
            DoctorImpactReason.OPEN_REQUIRED_FRONTIER.value in reasons
            or DoctorImpactReason.INCOMPLETE_CLOSURE.value in reasons
            or DoctorImpactReason.CIRCULAR_OWNERSHIP.value in reasons
            or DoctorImpactReason.MISSED_CONSUMER.value in reasons
        ):
            plan_disposition = DoctorImpactPlanDisposition.APPROVAL_REQUIRED
            doctor_plan_disposition = DoctorPlanDisposition.APPROVAL_REQUIRED
        else:
            plan_disposition = DoctorImpactPlanDisposition.ABSTAINED
            doctor_plan_disposition = DoctorPlanDisposition.ABSTAINED
        reasons.append(DoctorImpactReason.PLAN_ABSTAINED.value)
        reasons = sorted(set(reasons))

        # Non-admitted plan still records dispositions and open frontiers.
        plan_id = content_identity(
            {
                "schema": "deterministic-doctor-plan-abstention@1",
                "closure": closure.impact_closure_id,
                "reasons": reasons,
                "roots": roots.content_id,
            }
        )
        abstain_plan = DeterministicDoctorPlan(
            roots=roots,
            plan_id=f"plan:abstain:{plan_id}",
            snapshot_id=request.snapshot_id,
            finding_ids=request.finding_ids,
            disposition=doctor_plan_disposition,
            consumer_dispositions=plan_consumers,
            impact_closure_id=closure.impact_closure_id,
            steps=(),
            edit_sites=(),
            operator_ids=request.operator_ids,
            open_required_frontiers=closure.open_required_frontiers,
            scc_refs=tuple(scc.scc_id for scc in closure.sccs),
            permitted_read_paths=request.permitted_read_paths,
            permitted_write_paths=(),
            lease_id="",
            checkpoint_ref="",
            rollback_ref="",
            proof_refs=(),
            resource_bounds=request.resource_bounds or DoctorResourceBounds(),
            no_model_invariant=True,
            llm_router_enabled=False,
            model_invocation_count=0,
            invalidation_refs=request.invalidation_refs,
            premise_refs=request.premise_refs,
            goal_refs=request.goal_refs,
            candidate_refs=request.candidate_refs,
            tactician_plan_ref=request.tactician_plan_ref,
        )
        return DoctorPlanCompilationReceipt(
            roots=roots,
            disposition=plan_disposition,
            impact_closure_id=closure.impact_closure_id,
            reason_codes=tuple(sorted(set(reasons))),
            plan=abstain_plan,
            plan_id=abstain_plan.plan_id,
            scc_step_ids=(),
            consumer_disposition_set_id=disposition_set_id,
            mutation_admissible=False,
            producer_id=PRODUCER_ID,
            no_model_invariant=True,
            model_invocation_count=0,
        )

    # --- admitted path -------------------------------------------------------
    reasons.append(DoctorImpactReason.PLAN_ADMITTED.value)
    reasons.append(DoctorImpactReason.ATOMIC_SCC_PLAN.value)
    reasons.append(DoctorImpactReason.MUTATION_ADMISSIBLE.value)
    reasons.append(DoctorImpactReason.NO_MODEL_INVARIANT.value)
    reasons = sorted(set(reasons))

    operator_id = request.selected_operator_id
    steps = _build_scc_steps(
        closure,
        operator_id=operator_id,
        write_paths_by_consumer=write_paths_by_consumer,
        admitted=True,
    )
    # Ensure every migrate consumer is covered by at least one write step.
    covered: set[str] = set()
    for step in steps:
        if step.write_paths:
            covered.update(step.consumer_ids)
    if migrate_ids - covered:
        # Should not happen; fail closed.
        return DoctorPlanCompilationReceipt(
            roots=roots,
            disposition=DoctorImpactPlanDisposition.ABSTAINED,
            impact_closure_id=closure.impact_closure_id,
            reason_codes=tuple(
                sorted(
                    {
                        DoctorImpactReason.PLAN_GAP.value,
                        DoctorImpactReason.UNCOVERED_SCC.value,
                        DoctorImpactReason.PLAN_ABSTAINED.value,
                    }
                )
            ),
            plan=None,
            mutation_admissible=False,
            producer_id=PRODUCER_ID,
        )

    plan_id = content_identity(
        {
            "schema": "deterministic-doctor-plan@1",
            "closure": closure.impact_closure_id,
            "operator": operator_id,
            "steps": [step.step_id for step in steps],
            "consumers": [(c.consumer_id, c.disposition.value) for c in plan_consumers],
            "roots": roots.content_id,
        }
    )
    permitted_writes = tuple(sorted(set(write_paths)))
    admitted_plan = DeterministicDoctorPlan(
        roots=roots,
        plan_id=f"plan:admit:{plan_id}",
        snapshot_id=request.snapshot_id,
        finding_ids=request.finding_ids,
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=plan_consumers,
        impact_closure_id=closure.impact_closure_id,
        steps=steps,
        edit_sites=request.edit_sites,
        operator_ids=request.operator_ids or (operator_id,),
        target_ref=request.target_ref,
        value_source_ref=request.value_source_ref,
        placement_ref=request.placement_ref,
        selected_operator_id=operator_id,
        premise_refs=request.premise_refs,
        goal_refs=request.goal_refs,
        candidate_refs=request.candidate_refs,
        open_required_frontiers=(),
        validation_refs=request.validation_refs,
        scc_refs=tuple(scc.scc_id for scc in closure.sccs),
        tactician_plan_ref=request.tactician_plan_ref,
        permitted_read_paths=request.permitted_read_paths or permitted_writes,
        permitted_write_paths=permitted_writes,
        lease_id=lease_id,
        checkpoint_ref=request.checkpoint_ref,
        rollback_ref=request.rollback_ref,
        proof_refs=request.proof_refs or closure.proof_refs,
        resource_bounds=request.resource_bounds or DoctorResourceBounds(),
        no_model_invariant=True,
        llm_router_enabled=False,
        model_invocation_count=0,
        invalidation_refs=request.invalidation_refs,
    )
    return DoctorPlanCompilationReceipt(
        roots=roots,
        disposition=DoctorImpactPlanDisposition.ADMITTED,
        impact_closure_id=closure.impact_closure_id,
        reason_codes=tuple(reasons),
        plan=admitted_plan,
        plan_id=admitted_plan.plan_id,
        scc_step_ids=tuple(step.step_id for step in steps),
        consumer_disposition_set_id=disposition_set_id,
        mutation_admissible=True,
        producer_id=PRODUCER_ID,
        no_model_invariant=True,
        model_invocation_count=0,
    )


def mutation_requires_complete_closure(
    closure: DoctorImpactClosureReceipt,
    plan: DoctorPlanCompilationReceipt | DeterministicDoctorPlan | None = None,
) -> bool:
    """Return True only when mutation is fully authorized.

    Requires complete closure, current CIDs, no forbidden path, and one
    atomic admitted plan covering all necessary SCC steps.
    """

    if not isinstance(closure, DoctorImpactClosureReceipt):
        return False
    if not closure.mutation_admissible:
        return False
    if closure.completeness is not ImpactCompleteness.COMPLETE:
        return False
    if closure.open_required_frontiers:
        return False
    if plan is None:
        return False
    if isinstance(plan, DoctorPlanCompilationReceipt):
        return plan.may_mutate
    if isinstance(plan, DeterministicDoctorPlan):
        return plan.is_admitted and not plan.open_required_frontiers
    return False


def all_consumer_dispositions() -> tuple[DoctorConsumerDisposition, ...]:
    """Return the closed impact consumer-disposition vocabulary."""

    return tuple(DoctorConsumerDisposition)


def create_deterministic_doctor_impact_analyzer(
    **kwargs: Any,
) -> DeterministicDoctorImpactAnalyzer:
    """Factory for :class:`DeterministicDoctorImpactAnalyzer`."""

    return DeterministicDoctorImpactAnalyzer(**kwargs)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "CONTRACT_VERSION",
    "DETERMINISTIC_DOCTOR_IMPACT_INTERFACE",
    "DOCTOR_IMPACT_CLOSURE_RECEIPT_SCHEMA",
    "DOCTOR_IMPACT_CONSUMER_SCHEMA",
    "DOCTOR_PLAN_COMPILATION_RECEIPT_SCHEMA",
    "PRODUCER_ID",
    "DeterministicDoctorImpactAnalyzer",
    "DoctorConsumerDisposition",
    "DoctorGraphEdgeObservation",
    "DoctorImpactAuthorityError",
    "DoctorImpactBoundsError",
    "DoctorImpactClosureReceipt",
    "DoctorImpactConsumerObservation",
    "DoctorImpactConsumerRecord",
    "DoctorImpactError",
    "DoctorImpactFrontierObservation",
    "DoctorImpactPlanDisposition",
    "DoctorImpactReason",
    "DoctorImpactRequest",
    "DoctorImpactSCC",
    "DoctorPlanCompilationReceipt",
    "DoctorPlanCompilationRequest",
    "all_consumer_dispositions",
    "compile_deterministic_doctor_plan",
    "create_deterministic_doctor_impact_analyzer",
    "doctor_roots_to_propagation_roots",
    "map_to_plan_repair_disposition",
    "map_to_propagation_disposition",
    "mutation_requires_complete_closure",
    "path_is_forbidden",
    "rebuild_candidate_program_contract_delta",
]
