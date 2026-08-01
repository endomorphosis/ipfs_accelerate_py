"""Bounded, proof-gated contracts for transitive change propagation.

These records are deliberately *references*, never source containers.  Every
propagation stage exchanges immutable, content-addressed payloads that bind
exact base/candidate forest/tree/overlay roots plus graph/index/model/config/
translator/toolchain/policy identities.  State machines fail closed: forged
identities, authority promotion, incomplete consumer dispositions, and
completion without a fixed-point receipt are rejected at construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CHANGE_PROPAGATION_VERSION: Final[int] = 1
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_CLAUSE_COUNT: Final[int] = 256
MAX_CONSUMER_COUNT: Final[int] = 1_024
MAX_STEP_COUNT: Final[int] = 512
MAX_SCC_COUNT: Final[int] = 256
MAX_SPAN_OFFSET: Final[int] = 2**63 - 1

PROPAGATION_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/authority-roots@1"
)
GRAPH_NODE_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/graph-node-ref@1"
)
GRAPH_EDGE_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/graph-edge-ref@1"
)
PROGRAM_CHANGE_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/program-change-set@1"
)
CONTRACT_CLAUSE_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/contract-clause-delta@1"
)
PROGRAM_CONTRACT_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/program-contract-delta@1"
)
IMPACT_CONSUMER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/impact-consumer@1"
)
IMPACT_SCC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/impact-scc@1"
)
IMPACT_CLOSURE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/impact-closure-receipt@1"
)
CONSUMER_MIGRATION_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/consumer-migration-obligation@1"
)
MISSING_INPUT_REQUIREMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/missing-input-requirement@1"
)
VALUE_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/value-candidate@1"
)
REQUIRED_BEHAVIOR_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/required-behavior-contract@1"
)
ANALYTICAL_TRANSFORM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/analytical-transform@1"
)
PROPAGATION_PLAN_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/plan-step@1"
)
PROPAGATION_SCC_GROUP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/scc-group@1"
)
ATOMIC_PROPAGATION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/atomic-propagation-plan@1"
)
PROPAGATION_TRANSACTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/transaction@1"
)
PROPAGATION_COMPLETION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/completion-receipt@1"
)
FIXED_POINT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/fixed-point-receipt@1"
)


class ChangePropagationError(ContractValidationError):
    """Base class for change-propagation schema failures."""


class ChangePropagationBoundsError(ChangePropagationError):
    """A record attempted to exceed its declared compactness bounds."""


class ForgedChangePropagationIdentityError(ChangePropagationError):
    """A stored content identity did not match the canonical preimage."""


class ChangePropagationAuthorityError(ChangePropagationError):
    """Authority roots, paths, or disposition bindings did not match exactly."""


class ChangeSetKind(str, Enum):
    """Closed change-set sources."""

    REVIEWED_BASE_CANDIDATE = "reviewed_base_candidate"
    PROPOSED_CONTRACT_CHANGE = "proposed_contract_change"
    WORKTREE_DIFF = "worktree_diff"


class DeltaKind(str, Enum):
    """Closed semantic delta clause kinds."""

    PARAMETER_ADD = "parameter_add"
    PARAMETER_REMOVE = "parameter_remove"
    PARAMETER_RENAME = "parameter_rename"
    PARAMETER_REORDER = "parameter_reorder"
    PARAMETER_DEFAULT = "parameter_default"
    PARAMETER_KEYWORD = "parameter_keyword"
    PARAMETER_VARIANCE = "parameter_variance"
    RESULT_CHANGE = "result_change"
    GENERIC_CHANGE = "generic_change"
    NULLABILITY_CHANGE = "nullability_change"
    SCHEMA_CHANGE = "schema_change"
    SERIALIZATION_CHANGE = "serialization_change"
    PROTOCOL_CHANGE = "protocol_change"
    SYNC_ASYNC_CHANGE = "sync_async_change"
    CANCELLATION_CHANGE = "cancellation_change"
    ERROR_CHANGE = "error_change"
    EFFECT_CHANGE = "effect_change"
    CAPABILITY_CHANGE = "capability_change"
    AUTHORIZATION_CHANGE = "authorization_change"
    LIFECYCLE_CHANGE = "lifecycle_change"
    TEMPORAL_STATE_CHANGE = "temporal_state_change"
    CONSISTENCY_CHANGE = "consistency_change"
    RESOURCE_CHANGE = "resource_change"
    MEMORY_FACET_CHANGE = "memory_facet_change"
    SYMBOL_MOVE = "symbol_move"
    SYMBOL_RENAME = "symbol_rename"
    SYMBOL_REEXPORT = "symbol_reexport"
    SYMBOL_REGISTRATION = "symbol_registration"
    VISIBILITY_CHANGE = "visibility_change"
    CONSTRUCTOR_INTRO = "constructor_intro"
    CONSTRUCTOR_REMOVE = "constructor_remove"
    FIELD_INTRO = "field_intro"
    FIELD_REMOVE = "field_remove"
    METHOD_INTRO = "method_intro"
    METHOD_REMOVE = "method_remove"
    CLASS_INTRO = "class_intro"
    CLASS_REMOVE = "class_remove"
    DATA_STRUCTURE_INTRO = "data_structure_intro"
    DATA_STRUCTURE_REMOVE = "data_structure_remove"
    INTERFACE_INTRO = "interface_intro"
    INTERFACE_REMOVE = "interface_remove"
    FACTORY_INTRO = "factory_intro"
    FACTORY_REMOVE = "factory_remove"


class DeltaDisposition(str, Enum):
    """Closed compatibility outcome for one consumer domain."""

    BREAKING = "breaking"
    COMPATIBLE = "compatible"
    BEHAVIORAL = "behavioral"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class GraphProvenance(str, Enum):
    """Whether a graph ref is trusted or merely nominated."""

    TRUSTED = "trusted"
    NOMINATED = "nominated"
    FRONTIER = "frontier"


class GraphEdgeKind(str, Enum):
    """Closed edge categories in the program dependency graph."""

    CALL = "call"
    DATA_FLOW = "data_flow"
    STATE_FLOW = "state_flow"
    SCHEMA = "schema"
    WIRING = "wiring"
    OWNERSHIP = "ownership"
    VALIDATION = "validation"
    IMPORT = "import"
    OVERRIDE = "override"
    REGISTRATION = "registration"


class ConsumerDisposition(str, Enum):
    """Closed per-consumer migration outcomes."""

    MIGRATE = "migrate"
    ADAPTER = "adapter"
    COMPATIBLE = "compatible"
    UPSTREAM = "upstream"
    ABSTAIN = "abstain"
    REVIEW_ONLY = "review_only"
    EXCLUDED = "excluded"
    FRONTIER = "frontier"


class ImpactCompleteness(str, Enum):
    """Whether the reverse impact closure claims full coverage."""

    COMPLETE = "complete"
    PARTIAL_WITH_FRONTIER = "partial_with_frontier"
    ABSTAINED = "abstained"


class ValueCandidateKind(str, Enum):
    """Closed nomination sources for a missing input value."""

    LOCAL_NAME = "local_name"
    PARAMETER = "parameter"
    RECEIVER_STATE = "receiver_state"
    REACHING_DEFINITION = "reaching_definition"
    CONFIG_PROVIDER = "config_provider"
    REQUEST_CONTEXT = "request_context"
    DI_CONTAINER = "di_container"
    FACTORY = "factory"
    SCHEMA_DEFAULT = "schema_default"
    HISTORY = "history"
    GRAPH_NOMINATION = "graph_nomination"
    VECTOR_NOMINATION = "vector_nomination"
    CONSTRUCTION = "construction"


class ValueCandidateDisposition(str, Enum):
    """Closed analytical outcome for one value candidate."""

    PROVED = "proved"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    NOMINATED = "nominated"


class BehaviorKind(str, Enum):
    """Closed categories of required new behavior."""

    CLASS = "class"
    METHOD = "method"
    DATA_STRUCTURE = "data_structure"
    FACTORY = "factory"
    SCHEMA = "schema"
    STATE_TRANSITION = "state_transition"
    PROVIDER = "provider"
    ADAPTER = "adapter"
    SERIALIZER = "serializer"


class BehaviorEvidencePrecedence(str, Enum):
    """Independent evidence ranks; implementation is never first."""

    REVIEWED_IDL = "reviewed_idl"
    NORMATIVE_SPEC = "normative_spec"
    CALLER_POSTCONDITION = "caller_postcondition"
    CALLEE_PRECONDITION = "callee_precondition"
    DATA_INVARIANT = "data_invariant"
    MIGRATION_MANIFEST = "migration_manifest"
    ARCHITECTURE_OWNERSHIP = "architecture_ownership"
    HISTORY = "history"
    IMPLEMENTATION_HYPOTHESIS = "implementation_hypothesis"


class TransformKind(str, Enum):
    """Closed deterministic analytical transformations."""

    ADD_ARGUMENT = "add_argument"
    RENAME_ARGUMENT = "rename_argument"
    REORDER_ARGUMENT = "reorder_argument"
    THREAD_PARAMETER = "thread_parameter"
    ADD_IMPORT = "add_import"
    ADD_EXPORT = "add_export"
    ADD_REGISTRATION = "add_registration"
    ADD_ADAPTER = "add_adapter"
    UPDATE_CONSTRUCTOR = "update_constructor"
    UPDATE_SCHEMA_FIELD = "update_schema_field"
    UPDATE_SERIALIZER = "update_serializer"
    UPDATE_FIXTURE = "update_fixture"
    UPDATE_GENERATED_MANIFEST = "update_generated_manifest"


class TransformDisposition(str, Enum):
    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class PlanStepKind(str, Enum):
    ANALYTICAL = "analytical"
    LLM_BOUNDED = "llm_bounded"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


class PlanDisposition(str, Enum):
    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class TransactionState(str, Enum):
    PENDING = "pending"
    CHECKPOINTED = "checkpointed"
    EXECUTING = "executing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class CompletionDisposition(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    ABSTAINED = "abstained"
    FAILED = "failed"


_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
    }
)


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ChangePropagationError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ChangePropagationError(f"{field_name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ChangePropagationBoundsError(f"{field_name} exceeds its byte bound")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise ChangePropagationError(f"{field_name} must be an opaque compact identifier")
    return value


def _bounded_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChangePropagationError(f"{field_name} must be a finite integer")
    if value < minimum or value > MAX_SPAN_OFFSET:
        raise ChangePropagationBoundsError(f"{field_name} is outside the supported bound")
    return value


def _path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise ChangePropagationAuthorityError(
            f"{field_name} must be a relative repository path"
        )
    return candidate.as_posix()


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ChangePropagationError(f"{field_name} must be one of: {allowed}") from exc


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise ChangePropagationError(f"{field_name} must be a sequence of identifiers")
    else:
        raw = values
    if len(raw) > limit:
        raise ChangePropagationBoundsError(f"{field_name} exceeds its item bound")
    result_list: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = _identifier(value, field_name)
        if item not in seen:
            seen.add(item)
            result_list.append(item)
    result = tuple(result_list if preserve_order else sorted(result_list))
    if required and not result:
        raise ChangePropagationError(f"{field_name} must not be empty")
    return result


def _paths(
    values: Any, field_name: str, *, limit: int = MAX_REFERENCE_COUNT
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise ChangePropagationError(f"{field_name} must be a sequence of paths")
    else:
        raw = values
    if len(raw) > limit:
        raise ChangePropagationBoundsError(f"{field_name} exceeds its item bound")
    return tuple(sorted({_path(value, field_name) for value in raw}))


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    """Reject source bodies even when smuggled through an opaque mapping."""
    if isinstance(value, float):
        raise ChangePropagationError(f"{field_name} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ChangePropagationError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ChangePropagationError(f"{field_name} may not contain source bodies")
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ChangePropagationError(f"{field_name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str) -> None:
    _assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > MAX_RECORD_BYTES:
        raise ChangePropagationBoundsError(f"{name} exceeds its serialized byte bound")


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ForgedChangePropagationIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    """Fail-closed decoder shared by every externally supplied record."""
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ChangePropagationError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, CHANGE_PROPAGATION_VERSION):
        raise ChangePropagationError(f"{name} has an unsupported contract version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ChangePropagationError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    try:
        return {field_name: payload[field_name] for field_name in fields if field_name in payload}
    except KeyError as exc:
        raise ChangePropagationError(f"{name} omits a required field") from exc


def _decode_nested(
    value: Any,
    cls: type[CanonicalContract],
    field_name: str,
) -> CanonicalContract:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return cls.from_dict(value)  # type: ignore[attr-defined, return-value]
        return cls(**value)  # type: ignore[arg-type, call-arg, return-value]
    raise ChangePropagationError(f"{field_name} must be {cls.__name__}")


def _decode_sequence(
    values: Any,
    cls: type[CanonicalContract],
    field_name: str,
    *,
    limit: int,
    required: bool = False,
) -> tuple[CanonicalContract, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ChangePropagationError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise ChangePropagationBoundsError(f"{field_name} exceeds its item bound")
    items: list[CanonicalContract] = []
    seen: set[str] = set()
    for item in raw:
        decoded = _decode_nested(item, cls, field_name)
        if decoded.content_id not in seen:
            seen.add(decoded.content_id)
            items.append(decoded)
    result = tuple(sorted(items, key=lambda record: record.content_id))
    if required and not result:
        raise ChangePropagationError(f"{field_name} must not be empty")
    return result


# ---------------------------------------------------------------------------
# Authority roots (base + candidate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationAuthorityRoots(CanonicalContract):
    """Every root whose drift invalidates a propagation record.

    Binds base/candidate forest/tree/overlay identities plus the shared
    graph/index/model/config/translator/toolchain/policy roots.
    """

    SCHEMA: ClassVar[str] = PROPAGATION_ROOTS_SCHEMA

    repository_id: str
    base_forest_id: str
    base_tree_id: str
    base_overlay_id: str
    candidate_forest_id: str
    candidate_tree_id: str
    candidate_overlay_id: str
    graph_id: str
    index_id: str
    model_id: str
    config_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name != "SCHEMA":
                object.__setattr__(
                    self, field_name, _identifier(getattr(self, field_name), field_name)
                )
        if self.base_tree_id == self.candidate_tree_id and self.base_overlay_id == self.candidate_overlay_id:
            raise ChangePropagationAuthorityError(
                "base and candidate tree/overlay identities must differ"
            )
        _bounded(self, "propagation authority roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "SCHEMA"
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationAuthorityRoots":
        names = tuple(name for name in cls.__dataclass_fields__ if name != "SCHEMA")
        value = cls(**_decode_fields(payload, cls.SCHEMA, names, "propagation authority roots"))
        _verify_identity(payload, value)
        return value


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**value)
        )
    raise ChangePropagationError("roots must be PropagationAuthorityRoots")


# ---------------------------------------------------------------------------
# Graph node / edge refs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class GraphNodeRef(CanonicalContract):
    """Content-addressed pointer to one program-graph node."""

    SCHEMA: ClassVar[str] = GRAPH_NODE_REF_SCHEMA

    node_id: str
    kind: str
    path: str
    symbol_id: str
    artifact_id: str
    provenance: GraphProvenance
    extractor_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, "node_id"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "symbol_id", _identifier(self.symbol_id, "symbol_id"))
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self, "provenance", _enum(self.provenance, GraphProvenance, "provenance")
        )
        object.__setattr__(self, "extractor_id", _text(self.extractor_id, "extractor_id"))
        if self.provenance is GraphProvenance.TRUSTED and not self.extractor_id:
            raise ChangePropagationAuthorityError(
                "trusted graph nodes require an extractor identity"
            )
        _bounded(self, "graph node ref")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "node_id": self.node_id,
            "kind": self.kind,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "artifact_id": self.artifact_id,
            "provenance": self.provenance.value,
            "extractor_id": self.extractor_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphNodeRef":
        fields = (
            "node_id",
            "kind",
            "path",
            "symbol_id",
            "artifact_id",
            "provenance",
            "extractor_id",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "graph node ref"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True, order=True)
class GraphEdgeRef(CanonicalContract):
    """Content-addressed pointer to one program-graph edge."""

    SCHEMA: ClassVar[str] = GRAPH_EDGE_REF_SCHEMA

    edge_id: str
    kind: GraphEdgeKind
    source_node_id: str
    target_node_id: str
    provenance: GraphProvenance
    extractor_id: str = ""
    confidence_millipercent: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _identifier(self.edge_id, "edge_id"))
        object.__setattr__(self, "kind", _enum(self.kind, GraphEdgeKind, "kind"))
        object.__setattr__(
            self, "source_node_id", _identifier(self.source_node_id, "source_node_id")
        )
        object.__setattr__(
            self, "target_node_id", _identifier(self.target_node_id, "target_node_id")
        )
        object.__setattr__(
            self, "provenance", _enum(self.provenance, GraphProvenance, "provenance")
        )
        object.__setattr__(self, "extractor_id", _text(self.extractor_id, "extractor_id"))
        object.__setattr__(
            self,
            "confidence_millipercent",
            _bounded_int(self.confidence_millipercent, "confidence_millipercent"),
        )
        if self.confidence_millipercent > 100_000:
            raise ChangePropagationBoundsError(
                "confidence_millipercent cannot exceed 100000"
            )
        if self.source_node_id == self.target_node_id:
            raise ChangePropagationError("graph edge source and target must differ")
        if self.provenance is GraphProvenance.TRUSTED and not self.extractor_id:
            raise ChangePropagationAuthorityError(
                "trusted graph edges require an extractor identity"
            )
        if self.provenance is GraphProvenance.NOMINATED and self.confidence_millipercent >= 100_000:
            raise ChangePropagationAuthorityError(
                "nominated edges cannot claim full confidence"
            )
        # Nominated / frontier edges never promote themselves to trusted authority.
        if self.provenance is GraphProvenance.FRONTIER and self.extractor_id:
            # Frontier may name a partial extractor, but cannot claim trusted provenance.
            pass
        _bounded(self, "graph edge ref")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "edge_id": self.edge_id,
            "kind": self.kind.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "provenance": self.provenance.value,
            "extractor_id": self.extractor_id,
            "confidence_millipercent": self.confidence_millipercent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphEdgeRef":
        fields = (
            "edge_id",
            "kind",
            "source_node_id",
            "target_node_id",
            "provenance",
            "extractor_id",
            "confidence_millipercent",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "graph edge ref"))
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# ProgramChangeSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramChangeSet(CanonicalContract):
    """Base/candidate identity pair with changed spans and producer roots.

    Separates source-edit identity from derived semantic deltas so formatting
    and generated churn do not manufacture migration obligations.
    """

    SCHEMA: ClassVar[str] = PROGRAM_CHANGE_SET_SCHEMA

    roots: PropagationAuthorityRoots
    kind: ChangeSetKind
    producer_id: str
    changed_paths: tuple[str, ...]
    tombstone_paths: tuple[str, ...] = ()
    span_refs: tuple[str, ...] = ()
    submodule_root_ids: tuple[str, ...] = ()
    build_manifest_ids: tuple[str, ...] = ()
    generated_manifest_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "kind", _enum(self.kind, ChangeSetKind, "kind"))
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        object.__setattr__(
            self, "changed_paths", _paths(self.changed_paths, "changed_paths")
        )
        object.__setattr__(
            self, "tombstone_paths", _paths(self.tombstone_paths, "tombstone_paths")
        )
        object.__setattr__(self, "span_refs", _ids(self.span_refs, "span_refs"))
        object.__setattr__(
            self, "submodule_root_ids", _ids(self.submodule_root_ids, "submodule_root_ids")
        )
        object.__setattr__(
            self, "build_manifest_ids", _ids(self.build_manifest_ids, "build_manifest_ids")
        )
        object.__setattr__(
            self,
            "generated_manifest_ids",
            _ids(self.generated_manifest_ids, "generated_manifest_ids"),
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        if not self.changed_paths and not self.tombstone_paths:
            raise ChangePropagationError(
                "change set requires at least one changed or tombstone path"
            )
        overlap = set(self.changed_paths) & set(self.tombstone_paths)
        if overlap:
            raise ChangePropagationError(
                "changed_paths and tombstone_paths must be disjoint"
            )
        _bounded(self, "program change set")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "kind": self.kind.value,
            "producer_id": self.producer_id,
            "changed_paths": list(self.changed_paths),
            "tombstone_paths": list(self.tombstone_paths),
            "span_refs": list(self.span_refs),
            "submodule_root_ids": list(self.submodule_root_ids),
            "build_manifest_ids": list(self.build_manifest_ids),
            "generated_manifest_ids": list(self.generated_manifest_ids),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramChangeSet":
        fields = (
            "roots",
            "kind",
            "producer_id",
            "changed_paths",
            "tombstone_paths",
            "span_refs",
            "submodule_root_ids",
            "build_manifest_ids",
            "generated_manifest_ids",
            "evidence_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "program change set")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Semantic delta
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ContractClauseDelta(CanonicalContract):
    """One finite semantic clause comparison for a stated consumer domain."""

    SCHEMA: ClassVar[str] = CONTRACT_CLAUSE_DELTA_SCHEMA

    clause_id: str
    kind: DeltaKind
    disposition: DeltaDisposition
    subject_symbol_id: str
    consumer_domain: str
    before_contract_ref: str = ""
    after_contract_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "clause_id", _identifier(self.clause_id, "clause_id"))
        object.__setattr__(self, "kind", _enum(self.kind, DeltaKind, "kind"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, DeltaDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "consumer_domain",
            _identifier(self.consumer_domain, "consumer_domain"),
        )
        object.__setattr__(
            self, "before_contract_ref", _text(self.before_contract_ref, "before_contract_ref")
        )
        object.__setattr__(
            self, "after_contract_ref", _text(self.after_contract_ref, "after_contract_ref")
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if not self.before_contract_ref and not self.after_contract_ref:
            raise ChangePropagationError(
                "clause delta requires a before and/or after contract reference"
            )
        _bounded(self, "contract clause delta")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "clause_id": self.clause_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "subject_symbol_id": self.subject_symbol_id,
            "consumer_domain": self.consumer_domain,
            "before_contract_ref": self.before_contract_ref,
            "after_contract_ref": self.after_contract_ref,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractClauseDelta":
        fields = (
            "clause_id",
            "kind",
            "disposition",
            "subject_symbol_id",
            "consumer_domain",
            "before_contract_ref",
            "after_contract_ref",
            "reason",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "contract clause delta"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ProgramContractDelta(CanonicalContract):
    """Independently extracted before/after ProgramContract comparison."""

    SCHEMA: ClassVar[str] = PROGRAM_CONTRACT_DELTA_SCHEMA

    roots: PropagationAuthorityRoots
    change_set_id: str
    subject_symbol_id: str
    before_contract_ref: str
    after_contract_ref: str
    clauses: tuple[ContractClauseDelta, ...]
    evidence_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "before_contract_ref",
            _identifier(self.before_contract_ref, "before_contract_ref"),
        )
        object.__setattr__(
            self,
            "after_contract_ref",
            _identifier(self.after_contract_ref, "after_contract_ref"),
        )
        clauses = _decode_sequence(
            self.clauses, ContractClauseDelta, "clauses", limit=MAX_CLAUSE_COUNT, required=True
        )
        object.__setattr__(self, "clauses", clauses)
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        clause_ids = [item.clause_id for item in self.clauses]
        if len(clause_ids) != len(set(clause_ids)):
            raise ChangePropagationError("delta clauses must have unique clause_ids")
        for clause in self.clauses:
            if clause.subject_symbol_id != self.subject_symbol_id:
                raise ChangePropagationError(
                    "clause subject_symbol_id must match the delta subject"
                )
        _bounded(self, "program contract delta")

    @property
    def breaking_clauses(self) -> tuple[ContractClauseDelta, ...]:
        return tuple(
            item for item in self.clauses if item.disposition is DeltaDisposition.BREAKING
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "change_set_id": self.change_set_id,
            "subject_symbol_id": self.subject_symbol_id,
            "before_contract_ref": self.before_contract_ref,
            "after_contract_ref": self.after_contract_ref,
            "clauses": [item.to_dict() for item in self.clauses],
            "evidence_refs": list(self.evidence_refs),
            "proof_refs": list(self.proof_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramContractDelta":
        fields = (
            "roots",
            "change_set_id",
            "subject_symbol_id",
            "before_contract_ref",
            "after_contract_ref",
            "clauses",
            "evidence_refs",
            "proof_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "program contract delta")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Impact closure / frontier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ImpactConsumer(CanonicalContract):
    """One resolved or frontier consumer reached from a semantic delta."""

    SCHEMA: ClassVar[str] = IMPACT_CONSUMER_SCHEMA

    consumer_id: str
    node: GraphNodeRef
    depth: int
    mandatory: bool
    edge_refs: tuple[str, ...] = ()
    path_condition_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(
            self, "node", _decode_nested(self.node, GraphNodeRef, "node")
        )
        object.__setattr__(self, "depth", _bounded_int(self.depth, "depth"))
        if not isinstance(self.mandatory, bool):
            raise ChangePropagationError("mandatory must be a boolean")
        object.__setattr__(self, "edge_refs", _ids(self.edge_refs, "edge_refs"))
        object.__setattr__(
            self, "path_condition_ref", _text(self.path_condition_ref, "path_condition_ref")
        )
        _bounded(self, "impact consumer")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "consumer_id": self.consumer_id,
            "node": self.node.to_dict(),
            "depth": self.depth,
            "mandatory": self.mandatory,
            "edge_refs": list(self.edge_refs),
            "path_condition_ref": self.path_condition_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactConsumer":
        fields = (
            "consumer_id",
            "node",
            "depth",
            "mandatory",
            "edge_refs",
            "path_condition_ref",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "impact consumer")
        values["node"] = _decode_nested(values["node"], GraphNodeRef, "node")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True, order=True)
class ImpactSCC(CanonicalContract):
    """One strongly connected consumer group treated as a transaction unit."""

    SCHEMA: ClassVar[str] = IMPACT_SCC_SCHEMA

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
                limit=MAX_CONSUMER_COUNT,
            ),
        )
        _bounded(self, "impact scc")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "scc_id": self.scc_id,
            "member_consumer_ids": list(self.member_consumer_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactSCC":
        fields = ("scc_id", "member_consumer_ids")
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "impact scc"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ImpactClosureReceipt(CanonicalContract):
    """Reverse transitive impact closure with an explicit unknown frontier."""

    SCHEMA: ClassVar[str] = IMPACT_CLOSURE_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    delta_id: str
    completeness: ImpactCompleteness
    consumers: tuple[ImpactConsumer, ...]
    sccs: tuple[ImpactSCC, ...] = ()
    frontier_node_ids: tuple[str, ...] = ()
    frontier_edge_ids: tuple[str, ...] = ()
    excluded_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    resource_bound_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, ImpactCompleteness, "completeness"),
        )
        consumers = _decode_sequence(
            self.consumers,
            ImpactConsumer,
            "consumers",
            limit=MAX_CONSUMER_COUNT,
        )
        object.__setattr__(self, "consumers", consumers)
        sccs = _decode_sequence(self.sccs, ImpactSCC, "sccs", limit=MAX_SCC_COUNT)
        object.__setattr__(self, "sccs", sccs)
        object.__setattr__(
            self,
            "frontier_node_ids",
            _ids(self.frontier_node_ids, "frontier_node_ids", limit=MAX_CONSUMER_COUNT),
        )
        object.__setattr__(
            self,
            "frontier_edge_ids",
            _ids(self.frontier_edge_ids, "frontier_edge_ids", limit=MAX_CONSUMER_COUNT),
        )
        object.__setattr__(
            self, "excluded_refs", _ids(self.excluded_refs, "excluded_refs")
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self,
            "resource_bound_refs",
            _ids(self.resource_bound_refs, "resource_bound_refs"),
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        consumer_ids = {item.consumer_id for item in self.consumers}
        if len(consumer_ids) != len(self.consumers):
            raise ChangePropagationError("impact consumers must have unique consumer_ids")
        for scc in self.sccs:
            missing = set(scc.member_consumer_ids) - consumer_ids
            if missing:
                raise ChangePropagationError(
                    "scc members must reference known impact consumers"
                )
        if self.completeness is ImpactCompleteness.COMPLETE:
            if self.frontier_node_ids or self.frontier_edge_ids:
                raise ChangePropagationError(
                    "complete impact closure cannot retain an open frontier"
                )
        if self.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            if not self.frontier_node_ids and not self.frontier_edge_ids:
                raise ChangePropagationError(
                    "partial impact closure requires an explicit frontier"
                )
        if self.completeness is ImpactCompleteness.ABSTAINED and self.consumers:
            # Abstention may retain diagnostics but cannot claim mandatory coverage.
            if any(item.mandatory for item in self.consumers):
                raise ChangePropagationError(
                    "abstained impact closure cannot mark consumers mandatory"
                )
        _bounded(self, "impact closure receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "completeness": self.completeness.value,
            "consumers": [item.to_dict() for item in self.consumers],
            "sccs": [item.to_dict() for item in self.sccs],
            "frontier_node_ids": list(self.frontier_node_ids),
            "frontier_edge_ids": list(self.frontier_edge_ids),
            "excluded_refs": list(self.excluded_refs),
            "validation_refs": list(self.validation_refs),
            "resource_bound_refs": list(self.resource_bound_refs),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactClosureReceipt":
        fields = (
            "roots",
            "delta_id",
            "completeness",
            "consumers",
            "sccs",
            "frontier_node_ids",
            "frontier_edge_ids",
            "excluded_refs",
            "validation_refs",
            "resource_bound_refs",
            "evidence_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "impact closure receipt")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Per-consumer obligation, missing input, value candidate, behavior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerMigrationObligation(CanonicalContract):
    """Exact migration work for one impacted consumer (not a search hit)."""

    SCHEMA: ClassVar[str] = CONSUMER_MIGRATION_OBLIGATION_SCHEMA

    roots: PropagationAuthorityRoots
    obligation_id: str
    consumer_id: str
    delta_id: str
    disposition: ConsumerDisposition
    clause_ids: tuple[str, ...]
    node: GraphNodeRef
    proof_refs: tuple[str, ...] = ()
    missing_input_ids: tuple[str, ...] = ()
    behavior_contract_ids: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "obligation_id", _identifier(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, ConsumerDisposition, "disposition")
        )
        object.__setattr__(
            self, "clause_ids", _ids(self.clause_ids, "clause_ids", required=True)
        )
        object.__setattr__(
            self, "node", _decode_nested(self.node, GraphNodeRef, "node")
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "missing_input_ids", _ids(self.missing_input_ids, "missing_input_ids")
        )
        object.__setattr__(
            self,
            "behavior_contract_ids",
            _ids(self.behavior_contract_ids, "behavior_contract_ids"),
        )
        object.__setattr__(
            self, "invalidation_refs", _ids(self.invalidation_refs, "invalidation_refs")
        )
        if self.disposition is ConsumerDisposition.MIGRATE and not self.proof_refs:
            # Migration may be planned before full proof, but cannot claim proof-free
            # discharge; proof_refs may be empty only for non-admitted dispositions.
            pass
        if self.disposition in {
            ConsumerDisposition.COMPATIBLE,
            ConsumerDisposition.EXCLUDED,
        } and (self.missing_input_ids or self.behavior_contract_ids):
            raise ChangePropagationError(
                "compatible/excluded obligations cannot require missing inputs or new behavior"
            )
        if self.disposition is ConsumerDisposition.FRONTIER and self.proof_refs:
            raise ChangePropagationAuthorityError(
                "frontier obligations cannot carry proof authority"
            )
        _bounded(self, "consumer migration obligation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "obligation_id": self.obligation_id,
            "consumer_id": self.consumer_id,
            "delta_id": self.delta_id,
            "disposition": self.disposition.value,
            "clause_ids": list(self.clause_ids),
            "node": self.node.to_dict(),
            "proof_refs": list(self.proof_refs),
            "missing_input_ids": list(self.missing_input_ids),
            "behavior_contract_ids": list(self.behavior_contract_ids),
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsumerMigrationObligation":
        fields = (
            "roots",
            "obligation_id",
            "consumer_id",
            "delta_id",
            "disposition",
            "clause_ids",
            "node",
            "proof_refs",
            "missing_input_ids",
            "behavior_contract_ids",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "consumer migration obligation"
        )
        values["roots"] = _roots(values["roots"])
        values["node"] = _decode_nested(values["node"], GraphNodeRef, "node")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class MissingInputRequirement(CanonicalContract):
    """Exact missing argument/value requirement for one consumer path."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_REQUIREMENT_SCHEMA

    roots: PropagationAuthorityRoots
    requirement_id: str
    obligation_id: str
    clause_id: str
    parameter_name: str
    type_ref: str
    nullability: str
    information_content_ref: str
    construction_precondition_refs: tuple[str, ...] = ()
    result_postcondition_refs: tuple[str, ...] = ()
    allowed_error_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    authorization_refs: tuple[str, ...] = ()
    resource_refs: tuple[str, ...] = ()
    ownership_refs: tuple[str, ...] = ()
    propagation_depth_bound: int = 0
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "requirement_id", _identifier(self.requirement_id, "requirement_id")
        )
        object.__setattr__(
            self, "obligation_id", _identifier(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "clause_id", _identifier(self.clause_id, "clause_id"))
        object.__setattr__(
            self, "parameter_name", _identifier(self.parameter_name, "parameter_name")
        )
        object.__setattr__(self, "type_ref", _identifier(self.type_ref, "type_ref"))
        object.__setattr__(
            self, "nullability", _identifier(self.nullability, "nullability")
        )
        object.__setattr__(
            self,
            "information_content_ref",
            _identifier(self.information_content_ref, "information_content_ref"),
        )
        object.__setattr__(
            self,
            "construction_precondition_refs",
            _ids(self.construction_precondition_refs, "construction_precondition_refs"),
        )
        object.__setattr__(
            self,
            "result_postcondition_refs",
            _ids(self.result_postcondition_refs, "result_postcondition_refs"),
        )
        object.__setattr__(
            self, "allowed_error_refs", _ids(self.allowed_error_refs, "allowed_error_refs")
        )
        object.__setattr__(self, "effect_refs", _ids(self.effect_refs, "effect_refs"))
        object.__setattr__(
            self, "capability_refs", _ids(self.capability_refs, "capability_refs")
        )
        object.__setattr__(
            self, "authorization_refs", _ids(self.authorization_refs, "authorization_refs")
        )
        object.__setattr__(
            self, "resource_refs", _ids(self.resource_refs, "resource_refs")
        )
        object.__setattr__(
            self, "ownership_refs", _ids(self.ownership_refs, "ownership_refs")
        )
        object.__setattr__(
            self,
            "propagation_depth_bound",
            _bounded_int(self.propagation_depth_bound, "propagation_depth_bound"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        _bounded(self, "missing input requirement")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "requirement_id": self.requirement_id,
            "obligation_id": self.obligation_id,
            "clause_id": self.clause_id,
            "parameter_name": self.parameter_name,
            "type_ref": self.type_ref,
            "nullability": self.nullability,
            "information_content_ref": self.information_content_ref,
            "construction_precondition_refs": list(self.construction_precondition_refs),
            "result_postcondition_refs": list(self.result_postcondition_refs),
            "allowed_error_refs": list(self.allowed_error_refs),
            "effect_refs": list(self.effect_refs),
            "capability_refs": list(self.capability_refs),
            "authorization_refs": list(self.authorization_refs),
            "resource_refs": list(self.resource_refs),
            "ownership_refs": list(self.ownership_refs),
            "propagation_depth_bound": self.propagation_depth_bound,
            "proof_refs": list(self.proof_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputRequirement":
        fields = (
            "roots",
            "requirement_id",
            "obligation_id",
            "clause_id",
            "parameter_name",
            "type_ref",
            "nullability",
            "information_content_ref",
            "construction_precondition_refs",
            "result_postcondition_refs",
            "allowed_error_refs",
            "effect_refs",
            "capability_refs",
            "authorization_refs",
            "resource_refs",
            "ownership_refs",
            "propagation_depth_bound",
            "proof_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "missing input requirement")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ValueCandidate(CanonicalContract):
    """Nominated or proved source for one missing input; never path authority."""

    SCHEMA: ClassVar[str] = VALUE_CANDIDATE_SCHEMA

    roots: PropagationAuthorityRoots
    candidate_id: str
    requirement_id: str
    kind: ValueCandidateKind
    disposition: ValueCandidateDisposition
    source_node: GraphNodeRef
    expression_ref: str
    type_ref: str
    semantic_authority: bool = False
    proof_refs: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "requirement_id", _identifier(self.requirement_id, "requirement_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ValueCandidateKind, "kind")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ValueCandidateDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "source_node",
            _decode_nested(self.source_node, GraphNodeRef, "source_node"),
        )
        object.__setattr__(
            self, "expression_ref", _identifier(self.expression_ref, "expression_ref")
        )
        object.__setattr__(self, "type_ref", _identifier(self.type_ref, "type_ref"))
        if not isinstance(self.semantic_authority, bool):
            raise ChangePropagationError("semantic_authority must be a boolean")
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons")
        )
        # Vector/history/graph nominations never hold semantic authority.
        if self.kind in {
            ValueCandidateKind.VECTOR_NOMINATION,
            ValueCandidateKind.GRAPH_NOMINATION,
            ValueCandidateKind.HISTORY,
        } and self.semantic_authority:
            raise ChangePropagationAuthorityError(
                "nominated value candidates cannot claim semantic authority"
            )
        if self.disposition is ValueCandidateDisposition.PROVED:
            if not self.proof_refs:
                raise ChangePropagationError("proved value candidates require proof refs")
            if not self.semantic_authority:
                raise ChangePropagationAuthorityError(
                    "proved value candidates require semantic authority from reconstruction"
                )
        if self.disposition is ValueCandidateDisposition.NOMINATED and self.semantic_authority:
            raise ChangePropagationAuthorityError(
                "nominated value candidates cannot claim semantic authority"
            )
        if self.disposition is ValueCandidateDisposition.REFUTED and not self.rejection_reasons:
            raise ChangePropagationError(
                "refuted value candidates require rejection reasons"
            )
        _bounded(self, "value candidate")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "candidate_id": self.candidate_id,
            "requirement_id": self.requirement_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "source_node": self.source_node.to_dict(),
            "expression_ref": self.expression_ref,
            "type_ref": self.type_ref,
            "semantic_authority": self.semantic_authority,
            "proof_refs": list(self.proof_refs),
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueCandidate":
        fields = (
            "roots",
            "candidate_id",
            "requirement_id",
            "kind",
            "disposition",
            "source_node",
            "expression_ref",
            "type_ref",
            "semantic_authority",
            "proof_refs",
            "rejection_reasons",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "value candidate")
        values["roots"] = _roots(values["roots"])
        values["source_node"] = _decode_nested(
            values["source_node"], GraphNodeRef, "source_node"
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class RequiredBehaviorContract(CanonicalContract):
    """Independently sourced required behavior before placement or codegen."""

    SCHEMA: ClassVar[str] = REQUIRED_BEHAVIOR_CONTRACT_SCHEMA

    roots: PropagationAuthorityRoots
    behavior_id: str
    kind: BehaviorKind
    subject_symbol_id: str
    evidence_precedence: BehaviorEvidencePrecedence
    field_refs: tuple[str, ...] = ()
    constructor_refs: tuple[str, ...] = ()
    method_refs: tuple[str, ...] = ()
    invariant_refs: tuple[str, ...] = ()
    state_transition_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    authorization_refs: tuple[str, ...] = ()
    resource_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    placement_decision_ref: str = ""
    implementation_hypothesis: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "behavior_id", _identifier(self.behavior_id, "behavior_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, BehaviorKind, "kind"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "evidence_precedence",
            _enum(self.evidence_precedence, BehaviorEvidencePrecedence, "evidence_precedence"),
        )
        object.__setattr__(self, "field_refs", _ids(self.field_refs, "field_refs"))
        object.__setattr__(
            self, "constructor_refs", _ids(self.constructor_refs, "constructor_refs")
        )
        object.__setattr__(self, "method_refs", _ids(self.method_refs, "method_refs"))
        object.__setattr__(
            self, "invariant_refs", _ids(self.invariant_refs, "invariant_refs")
        )
        object.__setattr__(
            self,
            "state_transition_refs",
            _ids(self.state_transition_refs, "state_transition_refs"),
        )
        object.__setattr__(self, "effect_refs", _ids(self.effect_refs, "effect_refs"))
        object.__setattr__(
            self, "capability_refs", _ids(self.capability_refs, "capability_refs")
        )
        object.__setattr__(
            self, "authorization_refs", _ids(self.authorization_refs, "authorization_refs")
        )
        object.__setattr__(
            self, "resource_refs", _ids(self.resource_refs, "resource_refs")
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "placement_decision_ref",
            _text(self.placement_decision_ref, "placement_decision_ref"),
        )
        if not isinstance(self.implementation_hypothesis, bool):
            raise ChangePropagationError("implementation_hypothesis must be a boolean")
        # Implementation observations are never authoritative behavior sources.
        if (
            self.evidence_precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and not self.implementation_hypothesis
        ):
            raise ChangePropagationError(
                "implementation_hypothesis precedence requires implementation_hypothesis=true"
            )
        if (
            self.evidence_precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and self.implementation_hypothesis
        ):
            raise ChangePropagationAuthorityError(
                "implementation hypotheses cannot promote over independent evidence"
            )
        if (
            self.evidence_precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and self.proof_refs
        ):
            raise ChangePropagationAuthorityError(
                "implementation hypotheses cannot carry proof authority"
            )
        if not (
            self.field_refs
            or self.constructor_refs
            or self.method_refs
            or self.invariant_refs
            or self.state_transition_refs
        ):
            raise ChangePropagationError(
                "required behavior contract must state at least one structural clause"
            )
        _bounded(self, "required behavior contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "behavior_id": self.behavior_id,
            "kind": self.kind.value,
            "subject_symbol_id": self.subject_symbol_id,
            "evidence_precedence": self.evidence_precedence.value,
            "field_refs": list(self.field_refs),
            "constructor_refs": list(self.constructor_refs),
            "method_refs": list(self.method_refs),
            "invariant_refs": list(self.invariant_refs),
            "state_transition_refs": list(self.state_transition_refs),
            "effect_refs": list(self.effect_refs),
            "capability_refs": list(self.capability_refs),
            "authorization_refs": list(self.authorization_refs),
            "resource_refs": list(self.resource_refs),
            "proof_refs": list(self.proof_refs),
            "placement_decision_ref": self.placement_decision_ref,
            "implementation_hypothesis": self.implementation_hypothesis,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiredBehaviorContract":
        fields = (
            "roots",
            "behavior_id",
            "kind",
            "subject_symbol_id",
            "evidence_precedence",
            "field_refs",
            "constructor_refs",
            "method_refs",
            "invariant_refs",
            "state_transition_refs",
            "effect_refs",
            "capability_refs",
            "authorization_refs",
            "resource_refs",
            "proof_refs",
            "placement_decision_ref",
            "implementation_hypothesis",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "required behavior contract")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Analytical transform, plan, SCC group, transaction, completion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalTransform(CanonicalContract):
    """Closed deterministic transform with exact paths; never executes edits."""

    SCHEMA: ClassVar[str] = ANALYTICAL_TRANSFORM_SCHEMA

    roots: PropagationAuthorityRoots
    transform_id: str
    kind: TransformKind
    disposition: TransformDisposition
    obligation_ids: tuple[str, ...]
    target_paths: tuple[str, ...]
    expression_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    dependency_transform_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "transform_id", _identifier(self.transform_id, "transform_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, TransformKind, "kind"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, TransformDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", required=True),
        )
        object.__setattr__(
            self, "target_paths", _paths(self.target_paths, "target_paths")
        )
        object.__setattr__(
            self, "expression_refs", _ids(self.expression_refs, "expression_refs")
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "dependency_transform_ids",
            _ids(self.dependency_transform_ids, "dependency_transform_ids"),
        )
        object.__setattr__(
            self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons")
        )
        if self.disposition is TransformDisposition.ADMITTED:
            if not self.target_paths:
                raise ChangePropagationAuthorityError(
                    "admitted analytical transforms require exact target paths"
                )
            if not self.proof_refs:
                raise ChangePropagationError(
                    "admitted analytical transforms require proof refs"
                )
        else:
            if self.target_paths:
                raise ChangePropagationAuthorityError(
                    "non-admitted transforms cannot grant target path authority"
                )
        if (
            self.disposition is TransformDisposition.REJECTED
            and not self.rejection_reasons
        ):
            raise ChangePropagationError(
                "rejected transforms require rejection reasons"
            )
        _bounded(self, "analytical transform")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "transform_id": self.transform_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "obligation_ids": list(self.obligation_ids),
            "target_paths": list(self.target_paths),
            "expression_refs": list(self.expression_refs),
            "proof_refs": list(self.proof_refs),
            "dependency_transform_ids": list(self.dependency_transform_ids),
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyticalTransform":
        fields = (
            "roots",
            "transform_id",
            "kind",
            "disposition",
            "obligation_ids",
            "target_paths",
            "expression_refs",
            "proof_refs",
            "dependency_transform_ids",
            "rejection_reasons",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "analytical transform")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True, order=True)
class PropagationPlanStep(CanonicalContract):
    """One dependency-addressable step inside an atomic propagation plan."""

    SCHEMA: ClassVar[str] = PROPAGATION_PLAN_STEP_SCHEMA

    step_id: str
    kind: PlanStepKind
    obligation_ids: tuple[str, ...]
    dependency_step_ids: tuple[str, ...] = ()
    transform_id: str = ""
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    precondition_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    scc_group_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "kind", _enum(self.kind, PlanStepKind, "kind"))
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", required=True),
        )
        object.__setattr__(
            self,
            "dependency_step_ids",
            _ids(self.dependency_step_ids, "dependency_step_ids", preserve_order=True),
        )
        object.__setattr__(self, "transform_id", _text(self.transform_id, "transform_id"))
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self, "precondition_refs", _ids(self.precondition_refs, "precondition_refs")
        )
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self, "scc_group_id", _text(self.scc_group_id, "scc_group_id")
        )
        if self.kind is PlanStepKind.ANALYTICAL and not self.transform_id:
            raise ChangePropagationError(
                "analytical plan steps require a transform_id"
            )
        if self.kind is PlanStepKind.LLM_BOUNDED and not self.write_paths:
            raise ChangePropagationAuthorityError(
                "llm_bounded steps require exact write path authority"
            )
        if self.step_id in self.dependency_step_ids:
            raise ChangePropagationError("plan step cannot depend on itself")
        _bounded(self, "propagation plan step")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "step_id": self.step_id,
            "kind": self.kind.value,
            "obligation_ids": list(self.obligation_ids),
            "dependency_step_ids": list(self.dependency_step_ids),
            "transform_id": self.transform_id,
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "precondition_refs": list(self.precondition_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "validation_refs": list(self.validation_refs),
            "scc_group_id": self.scc_group_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationPlanStep":
        fields = (
            "step_id",
            "kind",
            "obligation_ids",
            "dependency_step_ids",
            "transform_id",
            "read_paths",
            "write_paths",
            "precondition_refs",
            "postcondition_refs",
            "validation_refs",
            "scc_group_id",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "propagation plan step"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True, order=True)
class PropagationSCCGroup(CanonicalContract):
    """SCC members executed as one atomic transaction group."""

    SCHEMA: ClassVar[str] = PROPAGATION_SCC_GROUP_SCHEMA

    group_id: str
    scc_id: str
    step_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        object.__setattr__(self, "scc_id", _identifier(self.scc_id, "scc_id"))
        object.__setattr__(
            self, "step_ids", _ids(self.step_ids, "step_ids", required=True, limit=MAX_STEP_COUNT)
        )
        object.__setattr__(
            self,
            "consumer_ids",
            _ids(self.consumer_ids, "consumer_ids", required=True, limit=MAX_CONSUMER_COUNT),
        )
        _bounded(self, "propagation scc group")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "group_id": self.group_id,
            "scc_id": self.scc_id,
            "step_ids": list(self.step_ids),
            "consumer_ids": list(self.consumer_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationSCCGroup":
        fields = ("group_id", "scc_id", "step_ids", "consumer_ids")
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "propagation scc group"))
        _verify_identity(payload, value)
        return value


def obligation_set_identity(obligations: Sequence[ConsumerMigrationObligation]) -> str:
    """Derive the identity of the complete, deterministically ordered obligation set."""
    if not obligations or len(obligations) > MAX_CONSUMER_COUNT:
        raise ChangePropagationBoundsError(
            "obligation set must contain a bounded nonempty set"
        )
    ids = tuple(sorted(item.content_id for item in obligations))
    if len(set(ids)) != len(ids):
        raise ChangePropagationError("obligation set contains duplicate obligations")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/change-propagation/obligation-set@1",
            "obligation_ids": list(ids),
        }
    )


@dataclass(frozen=True)
class AtomicPropagationPlan(CanonicalContract):
    """Complete atomic migration plan with one disposition per consumer."""

    SCHEMA: ClassVar[str] = ATOMIC_PROPAGATION_PLAN_SCHEMA

    roots: PropagationAuthorityRoots
    plan_id: str
    change_set_id: str
    delta_id: str
    impact_closure_id: str
    disposition: PlanDisposition
    obligations: tuple[ConsumerMigrationObligation, ...]
    obligation_set_id: str
    steps: tuple[PropagationPlanStep, ...]
    scc_groups: tuple[PropagationSCCGroup, ...] = ()
    permitted_read_paths: tuple[str, ...] = ()
    permitted_write_paths: tuple[str, ...] = ()
    checkpoint_strategy_ref: str = ""
    rollback_strategy_ref: str = ""
    fixed_point_obligation_ref: str = ""
    proof_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self,
            "impact_closure_id",
            _identifier(self.impact_closure_id, "impact_closure_id"),
        )
        object.__setattr__(
            self, "disposition", _enum(self.disposition, PlanDisposition, "disposition")
        )
        obligations = _decode_sequence(
            self.obligations,
            ConsumerMigrationObligation,
            "obligations",
            limit=MAX_CONSUMER_COUNT,
            required=True,
        )
        object.__setattr__(self, "obligations", obligations)
        for obligation in self.obligations:
            if obligation.roots != self.roots:
                raise ChangePropagationAuthorityError(
                    "all obligations must bind the plan authority roots"
                )
        expected_set_id = obligation_set_identity(self.obligations)
        if _identifier(self.obligation_set_id, "obligation_set_id") != expected_set_id:
            raise ForgedChangePropagationIdentityError(
                "obligation_set_id must identify the complete obligation set"
            )
        steps = _decode_sequence(
            self.steps, PropagationPlanStep, "steps", limit=MAX_STEP_COUNT
        )
        # Preserve step dependency order by sorting only for uniqueness, re-sort by step_id.
        object.__setattr__(self, "steps", steps)
        scc_groups = _decode_sequence(
            self.scc_groups, PropagationSCCGroup, "scc_groups", limit=MAX_SCC_COUNT
        )
        object.__setattr__(self, "scc_groups", scc_groups)
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
            self,
            "checkpoint_strategy_ref",
            _text(self.checkpoint_strategy_ref, "checkpoint_strategy_ref"),
        )
        object.__setattr__(
            self,
            "rollback_strategy_ref",
            _text(self.rollback_strategy_ref, "rollback_strategy_ref"),
        )
        object.__setattr__(
            self,
            "fixed_point_obligation_ref",
            _text(self.fixed_point_obligation_ref, "fixed_point_obligation_ref"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        consumer_ids = {item.consumer_id for item in self.obligations}
        if len(consumer_ids) != len(self.obligations):
            raise ChangePropagationError(
                "plan requires exactly one obligation per consumer"
            )
        # Complete consumer dispositions: every obligation has a closed disposition
        # (enforced by enum).  Plans must not leave any consumer without one.
        dispositions = {item.disposition for item in self.obligations}
        if not dispositions:
            raise ChangePropagationError("plan requires complete consumer dispositions")

        step_ids = {item.step_id for item in self.steps}
        if len(step_ids) != len(self.steps):
            raise ChangePropagationError("plan steps must have unique step_ids")
        for step in self.steps:
            missing_deps = set(step.dependency_step_ids) - step_ids
            if missing_deps:
                raise ChangePropagationError(
                    "plan step dependencies must reference known steps"
                )
            unknown_obligations = set(step.obligation_ids) - {
                item.obligation_id for item in self.obligations
            }
            if unknown_obligations:
                raise ChangePropagationError(
                    "plan steps must reference known obligations"
                )

        for group in self.scc_groups:
            if set(group.step_ids) - step_ids:
                raise ChangePropagationError(
                    "scc group steps must reference known plan steps"
                )
            if set(group.consumer_ids) - consumer_ids:
                raise ChangePropagationError(
                    "scc group consumers must reference known obligations"
                )

        if self.disposition is PlanDisposition.ADMITTED:
            if not self.steps:
                raise ChangePropagationError("admitted plans require steps")
            if not self.permitted_write_paths:
                raise ChangePropagationAuthorityError(
                    "admitted plans require exact write path authority"
                )
            if not self.checkpoint_strategy_ref or not self.rollback_strategy_ref:
                raise ChangePropagationError(
                    "admitted plans require checkpoint and rollback strategy refs"
                )
            if not self.fixed_point_obligation_ref:
                raise ChangePropagationError(
                    "admitted plans require a fixed-point obligation ref"
                )
            if not self.proof_refs:
                raise ChangePropagationError("admitted plans require proof refs")
            # No unresolved mandatory migrate without a covering step.
            migrate_ids = {
                item.obligation_id
                for item in self.obligations
                if item.disposition is ConsumerDisposition.MIGRATE
            }
            covered = {oid for step in self.steps for oid in step.obligation_ids}
            if migrate_ids - covered:
                raise ChangePropagationError(
                    "admitted plans must cover every migrate obligation with a step"
                )
            step_write_paths = {path for step in self.steps for path in step.write_paths}
            if not step_write_paths.issubset(set(self.permitted_write_paths)):
                raise ChangePropagationAuthorityError(
                    "step write paths must be within plan write authority"
                )
        else:
            if self.permitted_write_paths:
                raise ChangePropagationAuthorityError(
                    "non-admitted plans cannot grant write path authority"
                )
            if self.steps and self.disposition is PlanDisposition.ABSTAINED:
                # Abstention may retain diagnostic steps only if they write nothing.
                if any(step.write_paths for step in self.steps):
                    raise ChangePropagationAuthorityError(
                        "abstained plans cannot schedule write steps"
                    )
        _bounded(self, "atomic propagation plan")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "plan_id": self.plan_id,
            "change_set_id": self.change_set_id,
            "delta_id": self.delta_id,
            "impact_closure_id": self.impact_closure_id,
            "disposition": self.disposition.value,
            "obligations": [item.to_dict() for item in self.obligations],
            "obligation_set_id": self.obligation_set_id,
            "steps": [item.to_dict() for item in self.steps],
            "scc_groups": [item.to_dict() for item in self.scc_groups],
            "permitted_read_paths": list(self.permitted_read_paths),
            "permitted_write_paths": list(self.permitted_write_paths),
            "checkpoint_strategy_ref": self.checkpoint_strategy_ref,
            "rollback_strategy_ref": self.rollback_strategy_ref,
            "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            "proof_refs": list(self.proof_refs),
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomicPropagationPlan":
        fields = (
            "roots",
            "plan_id",
            "change_set_id",
            "delta_id",
            "impact_closure_id",
            "disposition",
            "obligations",
            "obligation_set_id",
            "steps",
            "scc_groups",
            "permitted_read_paths",
            "permitted_write_paths",
            "checkpoint_strategy_ref",
            "rollback_strategy_ref",
            "fixed_point_obligation_ref",
            "proof_refs",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "atomic propagation plan")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PropagationTransaction(CanonicalContract):
    """Isolated candidate-worktree transaction bound to one admitted plan."""

    SCHEMA: ClassVar[str] = PROPAGATION_TRANSACTION_SCHEMA

    roots: PropagationAuthorityRoots
    transaction_id: str
    plan_id: str
    state: TransactionState
    checkpoint_id: str
    active_scc_group_id: str = ""
    completed_step_ids: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()
    lease_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "state", _enum(self.state, TransactionState, "state"))
        object.__setattr__(
            self, "checkpoint_id", _identifier(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self, "active_scc_group_id", _text(self.active_scc_group_id, "active_scc_group_id")
        )
        object.__setattr__(
            self,
            "completed_step_ids",
            _ids(self.completed_step_ids, "completed_step_ids", limit=MAX_STEP_COUNT, preserve_order=True),
        )
        object.__setattr__(
            self, "diagnostic_refs", _ids(self.diagnostic_refs, "diagnostic_refs")
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        if self.state is TransactionState.PENDING and self.completed_step_ids:
            raise ChangePropagationError(
                "pending transactions cannot report completed steps"
            )
        if self.state is TransactionState.COMMITTED and self.active_scc_group_id:
            raise ChangePropagationError(
                "committed transactions cannot retain an active scc group"
            )
        if self.state is TransactionState.ROLLED_BACK and not self.diagnostic_refs:
            raise ChangePropagationError(
                "rolled-back transactions require diagnostic refs"
            )
        if self.state is TransactionState.FAILED and not self.diagnostic_refs:
            raise ChangePropagationError("failed transactions require diagnostic refs")
        if self.state is TransactionState.EXECUTING and not self.lease_id:
            raise ChangePropagationAuthorityError(
                "executing transactions require a writer lease"
            )
        _bounded(self, "propagation transaction")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "transaction_id": self.transaction_id,
            "plan_id": self.plan_id,
            "state": self.state.value,
            "checkpoint_id": self.checkpoint_id,
            "active_scc_group_id": self.active_scc_group_id,
            "completed_step_ids": list(self.completed_step_ids),
            "diagnostic_refs": list(self.diagnostic_refs),
            "lease_id": self.lease_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationTransaction":
        fields = (
            "roots",
            "transaction_id",
            "plan_id",
            "state",
            "checkpoint_id",
            "active_scc_group_id",
            "completed_step_ids",
            "diagnostic_refs",
            "lease_id",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "propagation transaction")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class FixedPointReceipt(CanonicalContract):
    """Evidence that re-diff and re-closure reached a fixed point on the candidate tree."""

    SCHEMA: ClassVar[str] = FIXED_POINT_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    receipt_id: str
    plan_id: str
    iteration_count: int
    residual_delta_ids: tuple[str, ...] = ()
    residual_consumer_ids: tuple[str, ...] = ()
    residual_frontier_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=1),
        )
        object.__setattr__(
            self, "residual_delta_ids", _ids(self.residual_delta_ids, "residual_delta_ids")
        )
        object.__setattr__(
            self,
            "residual_consumer_ids",
            _ids(self.residual_consumer_ids, "residual_consumer_ids"),
        )
        object.__setattr__(
            self,
            "residual_frontier_ids",
            _ids(self.residual_frontier_ids, "residual_frontier_ids"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        _bounded(self, "fixed point receipt")

    @property
    def is_fixed_point(self) -> bool:
        return (
            not self.residual_delta_ids
            and not self.residual_consumer_ids
            and not self.residual_frontier_ids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "iteration_count": self.iteration_count,
            "residual_delta_ids": list(self.residual_delta_ids),
            "residual_consumer_ids": list(self.residual_consumer_ids),
            "residual_frontier_ids": list(self.residual_frontier_ids),
            "proof_refs": list(self.proof_refs),
            "validation_refs": list(self.validation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixedPointReceipt":
        fields = (
            "roots",
            "receipt_id",
            "plan_id",
            "iteration_count",
            "residual_delta_ids",
            "residual_consumer_ids",
            "residual_frontier_ids",
            "proof_refs",
            "validation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "fixed point receipt")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PropagationCompletionReceipt(CanonicalContract):
    """Candidate-tree-bound completion; requires a fixed-point receipt for success."""

    SCHEMA: ClassVar[str] = PROPAGATION_COMPLETION_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    completion_id: str
    plan_id: str
    transaction_id: str
    disposition: CompletionDisposition
    fixed_point_receipt: FixedPointReceipt | None
    discharged_obligation_ids: tuple[str, ...] = ()
    unresolved_mandatory_ids: tuple[str, ...] = ()
    omitted_dependent_ids: tuple[str, ...] = ()
    uncovered_frontier_ids: tuple[str, ...] = ()
    unplanned_breaking_delta_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "completion_id", _identifier(self.completion_id, "completion_id")
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CompletionDisposition, "disposition"),
        )
        if self.fixed_point_receipt is not None:
            object.__setattr__(
                self,
                "fixed_point_receipt",
                _decode_nested(
                    self.fixed_point_receipt, FixedPointReceipt, "fixed_point_receipt"
                ),
            )
            if self.fixed_point_receipt.roots != self.roots:
                raise ChangePropagationAuthorityError(
                    "fixed-point receipt roots must match completion roots"
                )
            if self.fixed_point_receipt.plan_id != self.plan_id:
                raise ChangePropagationError(
                    "fixed-point receipt plan_id must match completion plan_id"
                )
        object.__setattr__(
            self,
            "discharged_obligation_ids",
            _ids(
                self.discharged_obligation_ids,
                "discharged_obligation_ids",
                limit=MAX_CONSUMER_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "unresolved_mandatory_ids",
            _ids(self.unresolved_mandatory_ids, "unresolved_mandatory_ids"),
        )
        object.__setattr__(
            self,
            "omitted_dependent_ids",
            _ids(self.omitted_dependent_ids, "omitted_dependent_ids"),
        )
        object.__setattr__(
            self,
            "uncovered_frontier_ids",
            _ids(self.uncovered_frontier_ids, "uncovered_frontier_ids"),
        )
        object.__setattr__(
            self,
            "unplanned_breaking_delta_ids",
            _ids(self.unplanned_breaking_delta_ids, "unplanned_breaking_delta_ids"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if self.disposition is CompletionDisposition.COMPLETE:
            if self.fixed_point_receipt is None:
                raise ChangePropagationError(
                    "completion without a fixed-point receipt is forbidden"
                )
            if not self.fixed_point_receipt.is_fixed_point:
                raise ChangePropagationError(
                    "complete disposition requires a residual-free fixed-point receipt"
                )
            if (
                self.unresolved_mandatory_ids
                or self.omitted_dependent_ids
                or self.uncovered_frontier_ids
                or self.unplanned_breaking_delta_ids
            ):
                raise ChangePropagationError(
                    "complete disposition forbids residual mandatory consumers, "
                    "omitted dependents, uncovered frontier, or unplanned breaking deltas"
                )
            if not self.discharged_obligation_ids:
                raise ChangePropagationError(
                    "complete disposition requires discharged obligation identities"
                )
            if not self.proof_refs or not self.validation_refs:
                raise ChangePropagationError(
                    "complete disposition requires proof and validation refs"
                )
        else:
            # Incomplete/abstained/failed may omit fixed-point, but cannot claim zero residual
            # while missing the receipt.
            if (
                self.fixed_point_receipt is None
                and not (
                    self.unresolved_mandatory_ids
                    or self.omitted_dependent_ids
                    or self.uncovered_frontier_ids
                    or self.unplanned_breaking_delta_ids
                    or self.disposition
                    in {CompletionDisposition.ABSTAINED, CompletionDisposition.FAILED}
                )
            ):
                raise ChangePropagationError(
                    "non-complete disposition without residual diagnostics is invalid"
                )
        _bounded(self, "propagation completion receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "completion_id": self.completion_id,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "disposition": self.disposition.value,
            "fixed_point_receipt": (
                self.fixed_point_receipt.to_dict() if self.fixed_point_receipt else None
            ),
            "discharged_obligation_ids": list(self.discharged_obligation_ids),
            "unresolved_mandatory_ids": list(self.unresolved_mandatory_ids),
            "omitted_dependent_ids": list(self.omitted_dependent_ids),
            "uncovered_frontier_ids": list(self.uncovered_frontier_ids),
            "unplanned_breaking_delta_ids": list(self.unplanned_breaking_delta_ids),
            "proof_refs": list(self.proof_refs),
            "validation_refs": list(self.validation_refs),
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationCompletionReceipt":
        fields = (
            "roots",
            "completion_id",
            "plan_id",
            "transaction_id",
            "disposition",
            "fixed_point_receipt",
            "discharged_obligation_ids",
            "unresolved_mandatory_ids",
            "omitted_dependent_ids",
            "uncovered_frontier_ids",
            "unplanned_breaking_delta_ids",
            "proof_refs",
            "validation_refs",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "propagation completion receipt"
        )
        values["roots"] = _roots(values["roots"])
        if values.get("fixed_point_receipt") is not None:
            values["fixed_point_receipt"] = _decode_nested(
                values["fixed_point_receipt"], FixedPointReceipt, "fixed_point_receipt"
            )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


__all__ = [
    "ANALYTICAL_TRANSFORM_SCHEMA",
    "ATOMIC_PROPAGATION_PLAN_SCHEMA",
    "AnalyticalTransform",
    "AtomicPropagationPlan",
    "BehaviorEvidencePrecedence",
    "BehaviorKind",
    "CHANGE_PROPAGATION_VERSION",
    "CONSUMER_MIGRATION_OBLIGATION_SCHEMA",
    "CONTRACT_CLAUSE_DELTA_SCHEMA",
    "ChangePropagationAuthorityError",
    "ChangePropagationBoundsError",
    "ChangePropagationError",
    "ChangeSetKind",
    "CompletionDisposition",
    "ConsumerDisposition",
    "ConsumerMigrationObligation",
    "ContractClauseDelta",
    "DeltaDisposition",
    "DeltaKind",
    "FIXED_POINT_RECEIPT_SCHEMA",
    "FixedPointReceipt",
    "ForgedChangePropagationIdentityError",
    "GRAPH_EDGE_REF_SCHEMA",
    "GRAPH_NODE_REF_SCHEMA",
    "GraphEdgeKind",
    "GraphEdgeRef",
    "GraphNodeRef",
    "GraphProvenance",
    "IMPACT_CLOSURE_RECEIPT_SCHEMA",
    "IMPACT_CONSUMER_SCHEMA",
    "IMPACT_SCC_SCHEMA",
    "ImpactClosureReceipt",
    "ImpactCompleteness",
    "ImpactConsumer",
    "ImpactSCC",
    "MAX_CLAUSE_COUNT",
    "MAX_CONSUMER_COUNT",
    "MAX_RECORD_BYTES",
    "MAX_REFERENCE_COUNT",
    "MAX_SCC_COUNT",
    "MAX_STEP_COUNT",
    "MISSING_INPUT_REQUIREMENT_SCHEMA",
    "MissingInputRequirement",
    "PROGRAM_CHANGE_SET_SCHEMA",
    "PROGRAM_CONTRACT_DELTA_SCHEMA",
    "PROPAGATION_COMPLETION_RECEIPT_SCHEMA",
    "PROPAGATION_PLAN_STEP_SCHEMA",
    "PROPAGATION_ROOTS_SCHEMA",
    "PROPAGATION_SCC_GROUP_SCHEMA",
    "PROPAGATION_TRANSACTION_SCHEMA",
    "PlanDisposition",
    "PlanStepKind",
    "ProgramChangeSet",
    "ProgramContractDelta",
    "PropagationAuthorityRoots",
    "PropagationCompletionReceipt",
    "PropagationPlanStep",
    "PropagationSCCGroup",
    "PropagationTransaction",
    "REQUIRED_BEHAVIOR_CONTRACT_SCHEMA",
    "RequiredBehaviorContract",
    "TransactionState",
    "TransformDisposition",
    "TransformKind",
    "VALUE_CANDIDATE_SCHEMA",
    "ValueCandidate",
    "ValueCandidateDisposition",
    "ValueCandidateKind",
    "obligation_set_identity",
]
