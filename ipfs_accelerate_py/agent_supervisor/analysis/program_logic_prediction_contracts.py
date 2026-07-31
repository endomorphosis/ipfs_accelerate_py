"""Bounded, proof-gated contracts for program-logic prediction (LPR-001).

These records are deliberately *references*, never source containers.  Every
prediction stage exchanges immutable, content-addressed payloads that bind
exact objective/trace/change/consumer identities plus forest/tree/overlay/
graph/index/corpus/model/translator/toolchain/policy/environment roots.

Authority lattice (fail-closed):

* Source authority is recorded separately from proof status and nomination
  scores.  Tactician, vector, knowledge-graph, and LLM nominations never carry
  semantic authority.
* Solver-only claims cannot assert verified or validated-refuted dispositions;
  those require kernel reconstruction or independently replayed countermodel
  evidence.
* :class:`LogicGuidedRepairPacket` is a context overlay over an existing RPR
  packet/plan/lease — never a third write-authority root.
* :class:`LogicFixedPointEvidenceAttachment` extends an existing completion
  receipt rather than replacing it.
* Memory, resource, and type facets remain distinct fields and cannot promote
  one another.
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


PROGRAM_LOGIC_PREDICTION_VERSION: Final[int] = 1
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_SUBGOAL_COUNT: Final[int] = 256
MAX_EDGE_COUNT: Final[int] = 1_024
MAX_SCORE_MILLIPERCENT: Final[int] = 100_000
MAX_SPAN_OFFSET: Final[int] = 2**63 - 1

PROGRAM_LOGIC_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic/authority-roots@1"
)
LOGIC_FACET_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic/facet-ref@1"
)
PROGRAM_LOGIC_GOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-goal@1"
)
LOGIC_GAP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-gap@1"
)
LOGIC_SUBGOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-subgoal@1"
)
TACTICIAN_SEARCH_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-search-plan@1"
)
LOGIC_HYPOTHESIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-hypothesis@1"
)
LOGIC_PREDICTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-prediction-receipt@1"
)
SEMANTIC_ROUND_TRIP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-round-trip-receipt@1"
)
PROGRAM_LOGIC_NATIVE_GOAL_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-native-goal-binding@1"
)
COUNTERMODEL_VALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/countermodel-validation-receipt@1"
)
LOGIC_GUIDED_REPAIR_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-guided-repair-packet@1"
)
LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-fixed-point-evidence-attachment@1"
)


class ProgramLogicPredictionError(ContractValidationError):
    """Base class for program-logic prediction schema failures."""


class ProgramLogicPredictionBoundsError(ProgramLogicPredictionError):
    """A record attempted to exceed its declared compactness bounds."""


class ForgedProgramLogicIdentityError(ProgramLogicPredictionError):
    """A stored content identity did not match the canonical preimage."""


class ProgramLogicAuthorityError(ProgramLogicPredictionError):
    """Authority roots, dispositions, or write-scope bindings did not match."""


# ---------------------------------------------------------------------------
# Closed dispositions
# ---------------------------------------------------------------------------


class GoalDisposition(str, Enum):
    """Closed lifecycle outcomes for one program-logic goal."""

    OPEN = "open"
    PLANNED = "planned"
    ADMITTED = "admitted"
    DISCHARGED = "discharged"
    RESIDUAL = "residual"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    ABSTAINED = "abstained"


class GapDisposition(str, Enum):
    """Closed outcomes for a static information gap."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    FRONTIER = "frontier"
    COVERED = "covered"
    UNSUPPORTED = "unsupported"
    STALE = "stale"


class SourceRouteKind(str, Enum):
    """Closed source-route classes for premise retrieval."""

    LOCAL_STATIC = "local_static"
    REVIEWED_CONTRACT = "reviewed_contract"
    NORMATIVE_SPEC = "normative_spec"
    REVIEWED_TEST = "reviewed_test"
    HISTORY = "history"
    DATAFLOW = "dataflow"
    GRAPH = "graph"
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TACTICIAN = "tactician"
    RUNTIME_WITNESS = "runtime_witness"
    LLM = "llm"
    SOLVER = "solver"


class SourceAuthorityClass(str, Enum):
    """Independent source authority lattice; never conflated with proof status."""

    AUTHORITATIVE = "authoritative"
    CONFORMANCE = "conformance"
    NOMINATING = "nominating"
    DIAGNOSTIC = "diagnostic"
    NONE = "none"


class ProofStatus(str, Enum):
    """Closed proof/refutation status independent of source authority and scores."""

    UNPROVED = "unproved"
    CANDIDATE = "candidate"
    SOLVER_CHECKED = "solver_checked"
    KERNEL_VERIFIED = "kernel_verified"
    VALIDATED_REFUTED = "validated_refuted"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    ERROR = "error"


class SubgoalDisposition(str, Enum):
    """Closed outcomes for one plan subgoal."""

    PENDING = "pending"
    PLANNED = "planned"
    REFINED = "refined"
    DISCHARGED = "discharged"
    RESIDUAL = "residual"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class HypothesisDisposition(str, Enum):
    """Closed hard-gate outcomes for a nominated hypothesis."""

    NOMINATED = "nominated"
    PLAN_ADMITTED = "plan_admitted"
    PROVED = "proved"
    VALIDATED_REFUTED = "validated_refuted"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"
    ABSTAINED = "abstained"


class PredictionDisposition(str, Enum):
    """Closed admission outcomes for a prediction receipt."""

    PROVED = "proved"
    VALIDATED_REFUTATION = "validated_refutation"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    ABSTAINED = "abstained"
    ERROR = "error"


class NativeGoalDisposition(str, Enum):
    """Closed outcomes for native goal binding / round-trip."""

    BOUND = "bound"
    ROUND_TRIP_OK = "round_trip_ok"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    INCONSISTENT = "inconsistent"
    ABSTAINED = "abstained"


class CountermodelDisposition(str, Enum):
    """Closed outcomes separating raw diagnostics from rejection authority."""

    DIAGNOSTIC_ONLY = "diagnostic_only"
    VALIDATED = "validated"
    REPLAY_FAILED = "replay_failed"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    INCONSISTENT = "inconsistent"


class ContextOverlayDisposition(str, Enum):
    """Closed outcomes for a logic-guided repair context overlay."""

    DETERMINISTIC = "deterministic"
    MODEL_REQUIRED = "model_required"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class FixedPointAttachmentDisposition(str, Enum):
    """Closed outcomes for logic evidence attached to an existing completion."""

    ATTACHED = "attached"
    RESIDUAL = "residual"
    INCOMPLETE = "incomplete"
    ROLLED_BACK = "rolled_back"


class GoalFamily(str, Enum):
    """Closed families of program-logic goals."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    COUNTEREXAMPLE = "counterexample"
    CONSISTENCY = "consistency"
    REFINEMENT = "refinement"
    PLACEMENT = "placement"
    VALUE = "value"
    BEHAVIOR = "behavior"


class LogicFacetKind(str, Enum):
    """Closed facet kinds; memory, resource, and type remain distinct."""

    TYPE = "type"
    EFFECT = "effect"
    RESOURCE = "resource"
    MEMORY = "memory"
    LIFETIME = "lifetime"
    AUTHORIZATION = "authorization"
    STATE = "state"
    SCHEMA = "schema"
    PLACEMENT = "placement"
    INFORMATION = "information"
    ERROR = "error"
    TEMPORAL = "temporal"


class GapMissingClass(str, Enum):
    """Closed missing-information classes for a logic gap."""

    PREMISE = "premise"
    CONTRACT = "contract"
    VALUE = "value"
    CONSTRUCTION = "construction"
    PLACEMENT = "placement"
    BEHAVIOR = "behavior"
    TRANSLATION = "translation"
    FRONTIER = "frontier"
    CONSISTENCY = "consistency"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"


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
        "theorem_text",
        "proof_script",
        "prompt_body",
    }
)

_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "access_token",
        "refresh_token",
        "bearer",
        "credential",
        "ssh_key",
        "client_secret",
    }
)

_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "api_key=",
    "apikey=",
    "password=",
    "secret=",
    "private_key",
    "authorization:",
    "bearer ",
    "-----begin",
    "client_secret=",
)

_NOMINATING_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.VECTOR,
        SourceRouteKind.KNOWLEDGE_GRAPH,
        SourceRouteKind.TACTICIAN,
        SourceRouteKind.LLM,
        SourceRouteKind.SOLVER,
        SourceRouteKind.RUNTIME_WITNESS,
        SourceRouteKind.HISTORY,
    }
)

_AUTHORITATIVE_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.NORMATIVE_SPEC,
        SourceRouteKind.REVIEWED_TEST,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
    }
)


# ---------------------------------------------------------------------------
# Validators / codecs
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ProgramLogicPredictionError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ProgramLogicPredictionError(f"{field_name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ProgramLogicPredictionBoundsError(f"{field_name} exceeds its byte bound")
    _assert_no_secret_text(value, field_name)
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise ProgramLogicPredictionError(
            f"{field_name} must be an opaque compact identifier"
        )
    return value


def _bounded_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramLogicPredictionError(f"{field_name} must be a finite integer")
    if value < minimum or value > MAX_SPAN_OFFSET:
        raise ProgramLogicPredictionBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _score_millipercent(value: Any, field_name: str) -> int:
    score = _bounded_int(value, field_name)
    if score > MAX_SCORE_MILLIPERCENT:
        raise ProgramLogicPredictionBoundsError(
            f"{field_name} cannot exceed {MAX_SCORE_MILLIPERCENT}"
        )
    return score


def _path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise ProgramLogicAuthorityError(
            f"{field_name} must be a relative repository path"
        )
    return candidate.as_posix()


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ProgramLogicPredictionError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


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
        raise ProgramLogicPredictionError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise ProgramLogicPredictionBoundsError(f"{field_name} exceeds its item bound")
    result_list: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = _identifier(value, field_name)
        if item not in seen:
            seen.add(item)
            result_list.append(item)
    result = tuple(result_list if preserve_order else sorted(result_list))
    if required and not result:
        raise ProgramLogicPredictionError(f"{field_name} must not be empty")
    return result


def _paths(
    values: Any, field_name: str, *, limit: int = MAX_REFERENCE_COUNT
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise ProgramLogicPredictionError(f"{field_name} must be a sequence of paths")
    else:
        raw = values
    if len(raw) > limit:
        raise ProgramLogicPredictionBoundsError(f"{field_name} exceeds its item bound")
    return tuple(sorted({_path(value, field_name) for value in raw}))


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramLogicPredictionError(f"{field_name} must be a boolean")
    return value


def _assert_no_secret_text(value: str, field_name: str) -> None:
    lowered = value.lower()
    for marker in _SECRET_VALUE_MARKERS:
        if marker in lowered:
            raise ProgramLogicPredictionError(
                f"{field_name} may not contain secret material"
            )


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    """Reject source bodies and secrets even when smuggled through mappings."""
    if isinstance(value, float):
        raise ProgramLogicPredictionError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramLogicPredictionError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ProgramLogicPredictionError(
                    f"{field_name} may not contain source bodies"
                )
            if normalized in _SECRET_KEY_MARKERS:
                raise ProgramLogicPredictionError(
                    f"{field_name} may not contain secret material"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ProgramLogicPredictionError(
            f"{field_name} may not contain binary bodies"
        )
    elif isinstance(value, str):
        _assert_no_secret_text(value, field_name)


def _bounded(record: CanonicalContract, name: str) -> None:
    _assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > MAX_RECORD_BYTES:
        raise ProgramLogicPredictionBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ForgedProgramLogicIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    """Fail-closed decoder shared by every externally supplied record."""
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ProgramLogicPredictionError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (
        None,
        PROGRAM_LOGIC_PREDICTION_VERSION,
    ):
        raise ProgramLogicPredictionError(
            f"{name} has an unsupported contract version"
        )
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ProgramLogicPredictionError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    try:
        return {
            field_name: payload[field_name]
            for field_name in fields
            if field_name in payload
        }
    except KeyError as exc:
        raise ProgramLogicPredictionError(f"{name} omits a required field") from exc


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
    raise ProgramLogicPredictionError(f"{field_name} must be {cls.__name__}")


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
        raise ProgramLogicPredictionError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise ProgramLogicPredictionBoundsError(f"{field_name} exceeds its item bound")
    items: list[CanonicalContract] = []
    seen: set[str] = set()
    for item in raw:
        decoded = _decode_nested(item, cls, field_name)
        if decoded.content_id not in seen:
            seen.add(decoded.content_id)
            items.append(decoded)
    result = tuple(sorted(items, key=lambda record: record.content_id))
    if required and not result:
        raise ProgramLogicPredictionError(f"{field_name} must not be empty")
    return result


def _assert_acyclic_subgoals(subgoals: Sequence["LogicSubgoal"]) -> None:
    """Reject cyclic subgoal dependency graphs (finite DAG only)."""
    if not subgoals:
        return
    ids = {item.subgoal_id for item in subgoals}
    if len(ids) != len(subgoals):
        raise ProgramLogicPredictionError("subgoal identities must be unique")
    adjacency: dict[str, tuple[str, ...]] = {
        item.subgoal_id: tuple(dep for dep in item.depends_on if dep in ids)
        for item in subgoals
    }
    # Also treat parent edges as dependency edges for cycle detection.
    for item in subgoals:
        parent = item.parent_subgoal_id
        if parent and parent in ids:
            existing = list(adjacency[item.subgoal_id])
            if parent not in existing:
                existing.append(parent)
            adjacency[item.subgoal_id] = tuple(existing)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ProgramLogicPredictionError(
                "subgoal dependency graph contains a cycle"
            )
        visiting.add(node)
        for dep in adjacency.get(node, ()):
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def _default_authority_for_route(route: SourceRouteKind) -> SourceAuthorityClass:
    if route in _AUTHORITATIVE_ROUTES:
        return SourceAuthorityClass.AUTHORITATIVE
    if route in _NOMINATING_ROUTES:
        return SourceAuthorityClass.NOMINATING
    return SourceAuthorityClass.NONE


# ---------------------------------------------------------------------------
# Authority roots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramLogicAuthorityRoots(CanonicalContract):
    """Every root whose drift invalidates a program-logic prediction record.

    Binds objective/trace/change/consumer identities plus forest/tree/overlay/
    graph/index/corpus/model/translator/toolchain/policy/environment roots.
    """

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_ROOTS_SCHEMA

    repository_id: str
    objective_id: str
    trace_id: str
    change_id: str
    consumer_id: str
    forest_id: str
    tree_id: str
    overlay_id: str
    graph_id: str
    index_id: str
    corpus_id: str
    model_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str
    environment_id: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name != "SCHEMA":
                object.__setattr__(
                    self,
                    field_name,
                    _identifier(getattr(self, field_name), field_name),
                )
        _bounded(self, "program logic authority roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "SCHEMA"
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicAuthorityRoots":
        names = tuple(name for name in cls.__dataclass_fields__ if name != "SCHEMA")
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "program logic authority roots")
        )
        _verify_identity(payload, value)
        return value


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**value)
        )
    raise ProgramLogicPredictionError("roots must be ProgramLogicAuthorityRoots")


# ---------------------------------------------------------------------------
# Facet reference (memory / resource / type remain distinct)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class LogicFacetRef(CanonicalContract):
    """Typed facet pointer; kinds never promote across memory/resource/type."""

    SCHEMA: ClassVar[str] = LOGIC_FACET_REF_SCHEMA

    facet_id: str
    kind: LogicFacetKind
    subject_symbol_id: str
    contract_ref: str = ""
    unsupported: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "facet_id", _identifier(self.facet_id, "facet_id"))
        object.__setattr__(self, "kind", _enum(self.kind, LogicFacetKind, "kind"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(self, "contract_ref", _text(self.contract_ref, "contract_ref"))
        object.__setattr__(self, "unsupported", _bool(self.unsupported, "unsupported"))
        # Resource quantities never live on memory facets (and vice versa).
        if self.kind is LogicFacetKind.MEMORY and self.contract_ref.startswith(
            "resource:"
        ):
            raise ProgramLogicAuthorityError(
                "memory facets cannot bind resource contracts"
            )
        if self.kind is LogicFacetKind.RESOURCE and self.contract_ref.startswith(
            "memory:"
        ):
            raise ProgramLogicAuthorityError(
                "resource facets cannot bind memory contracts"
            )
        if self.kind is LogicFacetKind.TYPE and self.contract_ref.startswith(
            ("memory:", "resource:")
        ):
            raise ProgramLogicAuthorityError(
                "type facets cannot bind memory or resource contracts"
            )
        _bounded(self, "logic facet ref")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "facet_id": self.facet_id,
            "kind": self.kind.value,
            "subject_symbol_id": self.subject_symbol_id,
            "contract_ref": self.contract_ref,
            "unsupported": self.unsupported,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicFacetRef":
        fields = (
            "facet_id",
            "kind",
            "subject_symbol_id",
            "contract_ref",
            "unsupported",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "logic facet ref"))
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# ProgramLogicGoal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramLogicGoal(CanonicalContract):
    """Finite positive/negative/counterexample goal bound to authority roots."""

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_GOAL_SCHEMA

    roots: ProgramLogicAuthorityRoots
    goal_id: str
    family: GoalFamily
    disposition: GoalDisposition
    positive_statement_ref: str
    negative_target_ref: str = ""
    counterexample_target_ref: str = ""
    parent_goal_id: str = ""
    affected_symbol_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    required_facets: tuple[LogicFacetRef, ...] = ()
    unsupported_facets: tuple[LogicFacetRef, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    assumption_authority: SourceAuthorityClass = SourceAuthorityClass.NONE
    proof_status: ProofStatus = ProofStatus.UNPROVED
    translation_requirement_refs: tuple[str, ...] = ()
    logic_family_refs: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "family", _enum(self.family, GoalFamily, "family"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, GoalDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "positive_statement_ref",
            _identifier(self.positive_statement_ref, "positive_statement_ref"),
        )
        object.__setattr__(
            self, "negative_target_ref", _text(self.negative_target_ref, "negative_target_ref")
        )
        object.__setattr__(
            self,
            "counterexample_target_ref",
            _text(self.counterexample_target_ref, "counterexample_target_ref"),
        )
        object.__setattr__(
            self, "parent_goal_id", _text(self.parent_goal_id, "parent_goal_id")
        )
        if self.parent_goal_id == self.goal_id:
            raise ProgramLogicPredictionError("a goal cannot be its own parent")
        object.__setattr__(
            self,
            "affected_symbol_ids",
            _ids(self.affected_symbol_ids, "affected_symbol_ids"),
        )
        object.__setattr__(self, "source_refs", _ids(self.source_refs, "source_refs"))
        object.__setattr__(
            self,
            "required_facets",
            _decode_sequence(
                self.required_facets, LogicFacetRef, "required_facets", limit=MAX_REFERENCE_COUNT
            ),
        )
        object.__setattr__(
            self,
            "unsupported_facets",
            _decode_sequence(
                self.unsupported_facets,
                LogicFacetRef,
                "unsupported_facets",
                limit=MAX_REFERENCE_COUNT,
            ),
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self,
            "assumption_authority",
            _enum(self.assumption_authority, SourceAuthorityClass, "assumption_authority"),
        )
        object.__setattr__(
            self, "proof_status", _enum(self.proof_status, ProofStatus, "proof_status")
        )
        object.__setattr__(
            self,
            "translation_requirement_refs",
            _ids(self.translation_requirement_refs, "translation_requirement_refs"),
        )
        object.__setattr__(
            self, "logic_family_refs", _ids(self.logic_family_refs, "logic_family_refs")
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if self.family is GoalFamily.NEGATIVE and not self.negative_target_ref:
            raise ProgramLogicPredictionError(
                "negative goals require a negative_target_ref"
            )
        if (
            self.family is GoalFamily.COUNTEREXAMPLE
            and not self.counterexample_target_ref
        ):
            raise ProgramLogicPredictionError(
                "counterexample goals require a counterexample_target_ref"
            )
        if self.disposition is GoalDisposition.DISCHARGED and self.proof_status not in {
            ProofStatus.KERNEL_VERIFIED,
            ProofStatus.VALIDATED_REFUTED,
        }:
            raise ProgramLogicAuthorityError(
                "discharged goals require kernel verification or validated refutation"
            )
        if self.proof_status is ProofStatus.KERNEL_VERIFIED and (
            self.assumption_authority
            in {SourceAuthorityClass.NOMINATING, SourceAuthorityClass.DIAGNOSTIC}
        ):
            raise ProgramLogicAuthorityError(
                "kernel-verified goals cannot rest on nominating/diagnostic assumptions"
            )
        # Facet kinds in required vs unsupported must not collide by identity.
        required_ids = {facet.facet_id for facet in self.required_facets}
        unsupported_ids = {facet.facet_id for facet in self.unsupported_facets}
        if required_ids & unsupported_ids:
            raise ProgramLogicPredictionError(
                "required and unsupported facets must be disjoint"
            )
        _bounded(self, "program logic goal")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "goal_id": self.goal_id,
            "family": self.family.value,
            "disposition": self.disposition.value,
            "positive_statement_ref": self.positive_statement_ref,
            "negative_target_ref": self.negative_target_ref,
            "counterexample_target_ref": self.counterexample_target_ref,
            "parent_goal_id": self.parent_goal_id,
            "affected_symbol_ids": list(self.affected_symbol_ids),
            "source_refs": list(self.source_refs),
            "required_facets": [item.to_dict() for item in self.required_facets],
            "unsupported_facets": [item.to_dict() for item in self.unsupported_facets],
            "assumption_refs": list(self.assumption_refs),
            "assumption_authority": self.assumption_authority.value,
            "proof_status": self.proof_status.value,
            "translation_requirement_refs": list(self.translation_requirement_refs),
            "logic_family_refs": list(self.logic_family_refs),
            "bound_refs": list(self.bound_refs),
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicGoal":
        fields = (
            "roots",
            "goal_id",
            "family",
            "disposition",
            "positive_statement_ref",
            "negative_target_ref",
            "counterexample_target_ref",
            "parent_goal_id",
            "affected_symbol_ids",
            "source_refs",
            "required_facets",
            "unsupported_facets",
            "assumption_refs",
            "assumption_authority",
            "proof_status",
            "translation_requirement_refs",
            "logic_family_refs",
            "bound_refs",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "program logic goal")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicGap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicGap(CanonicalContract):
    """Minimal static information demand; never carries body text or authority."""

    SCHEMA: ClassVar[str] = LOGIC_GAP_SCHEMA

    roots: ProgramLogicAuthorityRoots
    gap_id: str
    goal_id: str
    missing_class: GapMissingClass
    disposition: GapDisposition
    observed_fact_ref: str
    required_fact_ref: str
    discrepancy_ref: str
    dependency_slice_refs: tuple[str, ...] = ()
    candidate_source_routes: tuple[SourceRouteKind, ...] = ()
    unknown_frontier_refs: tuple[str, ...] = ()
    coverage_refs: tuple[str, ...] = ()
    severity: str = "mandatory"
    automation_eligible: bool = False
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "gap_id", _identifier(self.gap_id, "gap_id"))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self,
            "missing_class",
            _enum(self.missing_class, GapMissingClass, "missing_class"),
        )
        object.__setattr__(
            self, "disposition", _enum(self.disposition, GapDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "observed_fact_ref",
            _identifier(self.observed_fact_ref, "observed_fact_ref"),
        )
        object.__setattr__(
            self,
            "required_fact_ref",
            _identifier(self.required_fact_ref, "required_fact_ref"),
        )
        object.__setattr__(
            self, "discrepancy_ref", _identifier(self.discrepancy_ref, "discrepancy_ref")
        )
        object.__setattr__(
            self,
            "dependency_slice_refs",
            _ids(self.dependency_slice_refs, "dependency_slice_refs"),
        )
        if self.candidate_source_routes is None:
            routes: Sequence[Any] = ()
        elif isinstance(self.candidate_source_routes, Sequence) and not isinstance(
            self.candidate_source_routes, (str, bytes, bytearray)
        ):
            routes = self.candidate_source_routes
        else:
            raise ProgramLogicPredictionError(
                "candidate_source_routes must be a sequence"
            )
        if len(routes) > MAX_REFERENCE_COUNT:
            raise ProgramLogicPredictionBoundsError(
                "candidate_source_routes exceeds its item bound"
            )
        decoded_routes = tuple(
            sorted(
                {
                    _enum(item, SourceRouteKind, "candidate_source_routes")
                    for item in routes
                },
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "candidate_source_routes", decoded_routes)
        object.__setattr__(
            self,
            "unknown_frontier_refs",
            _ids(self.unknown_frontier_refs, "unknown_frontier_refs"),
        )
        object.__setattr__(
            self, "coverage_refs", _ids(self.coverage_refs, "coverage_refs")
        )
        object.__setattr__(
            self, "severity", _identifier(self.severity, "severity")
        )
        object.__setattr__(
            self,
            "automation_eligible",
            _bool(self.automation_eligible, "automation_eligible"),
        )
        # Gaps never claim semantic authority.
        if self.semantic_authority is not False:
            raise ProgramLogicAuthorityError(
                "logic gaps cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if self.disposition is GapDisposition.FRONTIER and not self.unknown_frontier_refs:
            raise ProgramLogicPredictionError(
                "frontier gaps require unknown_frontier_refs"
            )
        if self.disposition is GapDisposition.COVERED and not self.coverage_refs:
            raise ProgramLogicPredictionError("covered gaps require coverage_refs")
        if (
            self.disposition is GapDisposition.REQUIRED
            and self.automation_eligible
            and not self.candidate_source_routes
        ):
            raise ProgramLogicPredictionError(
                "automation-eligible required gaps need candidate source routes"
            )
        _bounded(self, "logic gap")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "gap_id": self.gap_id,
            "goal_id": self.goal_id,
            "missing_class": self.missing_class.value,
            "disposition": self.disposition.value,
            "observed_fact_ref": self.observed_fact_ref,
            "required_fact_ref": self.required_fact_ref,
            "discrepancy_ref": self.discrepancy_ref,
            "dependency_slice_refs": list(self.dependency_slice_refs),
            "candidate_source_routes": [
                item.value for item in self.candidate_source_routes
            ],
            "unknown_frontier_refs": list(self.unknown_frontier_refs),
            "coverage_refs": list(self.coverage_refs),
            "severity": self.severity,
            "automation_eligible": self.automation_eligible,
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicGap":
        fields = (
            "roots",
            "gap_id",
            "goal_id",
            "missing_class",
            "disposition",
            "observed_fact_ref",
            "required_fact_ref",
            "discrepancy_ref",
            "dependency_slice_refs",
            "candidate_source_routes",
            "unknown_frontier_refs",
            "coverage_refs",
            "severity",
            "automation_eligible",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "logic gap")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicSubgoal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class LogicSubgoal(CanonicalContract):
    """One node in a finite acyclic subgoal DAG."""

    SCHEMA: ClassVar[str] = LOGIC_SUBGOAL_SCHEMA

    subgoal_id: str
    goal_id: str
    disposition: SubgoalDisposition
    claim_ref: str
    parent_subgoal_id: str = ""
    depends_on: tuple[str, ...] = ()
    source_route: SourceRouteKind = SourceRouteKind.LOCAL_STATIC
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE
    proof_status: ProofStatus = ProofStatus.UNPROVED
    score_millipercent: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subgoal_id", _identifier(self.subgoal_id, "subgoal_id")
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SubgoalDisposition, "disposition"),
        )
        object.__setattr__(self, "claim_ref", _identifier(self.claim_ref, "claim_ref"))
        object.__setattr__(
            self,
            "parent_subgoal_id",
            _text(self.parent_subgoal_id, "parent_subgoal_id"),
        )
        object.__setattr__(
            self, "depends_on", _ids(self.depends_on, "depends_on", preserve_order=True)
        )
        object.__setattr__(
            self,
            "source_route",
            _enum(self.source_route, SourceRouteKind, "source_route"),
        )
        object.__setattr__(
            self,
            "source_authority",
            _enum(self.source_authority, SourceAuthorityClass, "source_authority"),
        )
        object.__setattr__(
            self, "proof_status", _enum(self.proof_status, ProofStatus, "proof_status")
        )
        object.__setattr__(
            self,
            "score_millipercent",
            _score_millipercent(self.score_millipercent, "score_millipercent"),
        )
        if self.subgoal_id in self.depends_on:
            raise ProgramLogicPredictionError("a subgoal cannot depend on itself")
        if self.parent_subgoal_id == self.subgoal_id:
            raise ProgramLogicPredictionError("a subgoal cannot be its own parent")
        # Nominating routes cannot claim authoritative source class or proof.
        if self.source_route in _NOMINATING_ROUTES:
            if self.source_authority is SourceAuthorityClass.AUTHORITATIVE:
                raise ProgramLogicAuthorityError(
                    "nominating source routes cannot claim authoritative source class"
                )
            if self.proof_status in {
                ProofStatus.KERNEL_VERIFIED,
                ProofStatus.VALIDATED_REFUTED,
            }:
                raise ProgramLogicAuthorityError(
                    "nominating routes cannot assert verified or validated-refuted proof status"
                )
            if self.source_authority is SourceAuthorityClass.NONE:
                object.__setattr__(
                    self, "source_authority", SourceAuthorityClass.NOMINATING
                )
        if (
            self.source_authority is SourceAuthorityClass.AUTHORITATIVE
            and self.source_route not in _AUTHORITATIVE_ROUTES
        ):
            raise ProgramLogicAuthorityError(
                "authoritative source class requires an authoritative source route"
            )
        _bounded(self, "logic subgoal")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "subgoal_id": self.subgoal_id,
            "goal_id": self.goal_id,
            "disposition": self.disposition.value,
            "claim_ref": self.claim_ref,
            "parent_subgoal_id": self.parent_subgoal_id,
            "depends_on": list(self.depends_on),
            "source_route": self.source_route.value,
            "source_authority": self.source_authority.value,
            "proof_status": self.proof_status.value,
            "score_millipercent": self.score_millipercent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicSubgoal":
        fields = (
            "subgoal_id",
            "goal_id",
            "disposition",
            "claim_ref",
            "parent_subgoal_id",
            "depends_on",
            "source_route",
            "source_authority",
            "proof_status",
            "score_millipercent",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "logic subgoal"))
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# TacticianSearchPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticianSearchPlan(CanonicalContract):
    """Bounded advisory proof-search plan; semantic_authority is always false."""

    SCHEMA: ClassVar[str] = TACTICIAN_SEARCH_PLAN_SCHEMA

    roots: ProgramLogicAuthorityRoots
    plan_id: str
    goal_ids: tuple[str, ...]
    ordered_source_routes: tuple[SourceRouteKind, ...]
    query_refs: tuple[str, ...] = ()
    selected_premise_ids: tuple[str, ...] = ()
    excluded_premise_ids: tuple[str, ...] = ()
    exclusion_rationale_refs: tuple[str, ...] = ()
    subgoals: tuple[LogicSubgoal, ...] = ()
    planned_logic_family_refs: tuple[str, ...] = ()
    translation_refs: tuple[str, ...] = ()
    stop_policy_ref: str = ""
    escalation_policy_ref: str = ""
    abstention_policy_ref: str = ""
    resource_policy_ref: str = ""
    planner_id: str = ""
    model_id: str = ""
    config_id: str = ""
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "goal_ids", _ids(self.goal_ids, "goal_ids", required=True)
        )
        if self.ordered_source_routes is None:
            routes: Sequence[Any] = ()
        elif isinstance(self.ordered_source_routes, Sequence) and not isinstance(
            self.ordered_source_routes, (str, bytes, bytearray)
        ):
            routes = self.ordered_source_routes
        else:
            raise ProgramLogicPredictionError(
                "ordered_source_routes must be a sequence"
            )
        if not routes:
            raise ProgramLogicPredictionError(
                "ordered_source_routes must not be empty"
            )
        if len(routes) > MAX_REFERENCE_COUNT:
            raise ProgramLogicPredictionBoundsError(
                "ordered_source_routes exceeds its item bound"
            )
        decoded_routes = tuple(
            _enum(item, SourceRouteKind, "ordered_source_routes") for item in routes
        )
        object.__setattr__(self, "ordered_source_routes", decoded_routes)
        object.__setattr__(self, "query_refs", _ids(self.query_refs, "query_refs"))
        object.__setattr__(
            self,
            "selected_premise_ids",
            _ids(self.selected_premise_ids, "selected_premise_ids"),
        )
        object.__setattr__(
            self,
            "excluded_premise_ids",
            _ids(self.excluded_premise_ids, "excluded_premise_ids"),
        )
        selected = set(self.selected_premise_ids)
        excluded = set(self.excluded_premise_ids)
        if selected & excluded:
            raise ProgramLogicPredictionError(
                "selected and excluded premises must be disjoint"
            )
        object.__setattr__(
            self,
            "exclusion_rationale_refs",
            _ids(self.exclusion_rationale_refs, "exclusion_rationale_refs"),
        )
        object.__setattr__(
            self,
            "subgoals",
            _decode_sequence(
                self.subgoals, LogicSubgoal, "subgoals", limit=MAX_SUBGOAL_COUNT
            ),
        )
        _assert_acyclic_subgoals(self.subgoals)
        for subgoal in self.subgoals:
            if subgoal.goal_id not in self.goal_ids:
                raise ProgramLogicPredictionError(
                    "subgoal goal_id must be listed in plan goal_ids"
                )
        object.__setattr__(
            self,
            "planned_logic_family_refs",
            _ids(self.planned_logic_family_refs, "planned_logic_family_refs"),
        )
        object.__setattr__(
            self, "translation_refs", _ids(self.translation_refs, "translation_refs")
        )
        object.__setattr__(
            self, "stop_policy_ref", _text(self.stop_policy_ref, "stop_policy_ref")
        )
        object.__setattr__(
            self,
            "escalation_policy_ref",
            _text(self.escalation_policy_ref, "escalation_policy_ref"),
        )
        object.__setattr__(
            self,
            "abstention_policy_ref",
            _text(self.abstention_policy_ref, "abstention_policy_ref"),
        )
        object.__setattr__(
            self,
            "resource_policy_ref",
            _text(self.resource_policy_ref, "resource_policy_ref"),
        )
        object.__setattr__(self, "planner_id", _text(self.planner_id, "planner_id"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "config_id", _text(self.config_id, "config_id"))
        if self.semantic_authority is not False:
            raise ProgramLogicAuthorityError(
                "tactician search plans cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        _bounded(self, "tactician search plan")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "plan_id": self.plan_id,
            "goal_ids": list(self.goal_ids),
            "ordered_source_routes": [
                item.value for item in self.ordered_source_routes
            ],
            "query_refs": list(self.query_refs),
            "selected_premise_ids": list(self.selected_premise_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "exclusion_rationale_refs": list(self.exclusion_rationale_refs),
            "subgoals": [item.to_dict() for item in self.subgoals],
            "planned_logic_family_refs": list(self.planned_logic_family_refs),
            "translation_refs": list(self.translation_refs),
            "stop_policy_ref": self.stop_policy_ref,
            "escalation_policy_ref": self.escalation_policy_ref,
            "abstention_policy_ref": self.abstention_policy_ref,
            "resource_policy_ref": self.resource_policy_ref,
            "planner_id": self.planner_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticianSearchPlan":
        fields = (
            "roots",
            "plan_id",
            "goal_ids",
            "ordered_source_routes",
            "query_refs",
            "selected_premise_ids",
            "excluded_premise_ids",
            "exclusion_rationale_refs",
            "subgoals",
            "planned_logic_family_refs",
            "translation_refs",
            "stop_policy_ref",
            "escalation_policy_ref",
            "abstention_policy_ref",
            "resource_policy_ref",
            "planner_id",
            "model_id",
            "config_id",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "tactician search plan")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicHypothesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicHypothesis(CanonicalContract):
    """Nominated consequence with scores separated from hard-gate disposition."""

    SCHEMA: ClassVar[str] = LOGIC_HYPOTHESIS_SCHEMA

    roots: ProgramLogicAuthorityRoots
    hypothesis_id: str
    target_goal_id: str
    disposition: HypothesisDisposition
    claimed_consequence_ref: str
    construction_ref: str = ""
    placement_ref: str = ""
    value_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    evidence_route_kinds: tuple[SourceRouteKind, ...] = ()
    selected_premise_ids: tuple[str, ...] = ()
    counterexample_target_ref: str = ""
    source_authority: SourceAuthorityClass = SourceAuthorityClass.NOMINATING
    proof_status: ProofStatus = ProofStatus.UNPROVED
    completeness: bool = False
    unsupported_flags: tuple[str, ...] = ()
    nomination_score_millipercent: int = 0
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "hypothesis_id", _identifier(self.hypothesis_id, "hypothesis_id")
        )
        object.__setattr__(
            self, "target_goal_id", _identifier(self.target_goal_id, "target_goal_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, HypothesisDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "claimed_consequence_ref",
            _identifier(self.claimed_consequence_ref, "claimed_consequence_ref"),
        )
        object.__setattr__(
            self, "construction_ref", _text(self.construction_ref, "construction_ref")
        )
        object.__setattr__(
            self, "placement_ref", _text(self.placement_ref, "placement_ref")
        )
        object.__setattr__(self, "value_ref", _text(self.value_ref, "value_ref"))
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        if self.evidence_route_kinds is None:
            routes: Sequence[Any] = ()
        elif isinstance(self.evidence_route_kinds, Sequence) and not isinstance(
            self.evidence_route_kinds, (str, bytes, bytearray)
        ):
            routes = self.evidence_route_kinds
        else:
            raise ProgramLogicPredictionError(
                "evidence_route_kinds must be a sequence"
            )
        if len(routes) > MAX_REFERENCE_COUNT:
            raise ProgramLogicPredictionBoundsError(
                "evidence_route_kinds exceeds its item bound"
            )
        decoded_routes = tuple(
            sorted(
                {_enum(item, SourceRouteKind, "evidence_route_kinds") for item in routes},
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "evidence_route_kinds", decoded_routes)
        object.__setattr__(
            self,
            "selected_premise_ids",
            _ids(self.selected_premise_ids, "selected_premise_ids"),
        )
        object.__setattr__(
            self,
            "counterexample_target_ref",
            _text(self.counterexample_target_ref, "counterexample_target_ref"),
        )
        object.__setattr__(
            self,
            "source_authority",
            _enum(self.source_authority, SourceAuthorityClass, "source_authority"),
        )
        object.__setattr__(
            self, "proof_status", _enum(self.proof_status, ProofStatus, "proof_status")
        )
        object.__setattr__(
            self, "completeness", _bool(self.completeness, "completeness")
        )
        object.__setattr__(
            self, "unsupported_flags", _ids(self.unsupported_flags, "unsupported_flags")
        )
        object.__setattr__(
            self,
            "nomination_score_millipercent",
            _score_millipercent(
                self.nomination_score_millipercent, "nomination_score_millipercent"
            ),
        )
        if self.semantic_authority is not False:
            raise ProgramLogicAuthorityError(
                "logic hypotheses cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        nominating = any(route in _NOMINATING_ROUTES for route in decoded_routes)
        if nominating and self.source_authority is SourceAuthorityClass.AUTHORITATIVE:
            raise ProgramLogicAuthorityError(
                "Tactician/vector/KG/LLM nominations cannot claim authoritative source class"
            )
        if self.disposition is HypothesisDisposition.PROVED:
            if self.proof_status is not ProofStatus.KERNEL_VERIFIED:
                raise ProgramLogicAuthorityError(
                    "proved hypotheses require kernel_verified proof status; "
                    "solver-only claims are rejected"
                )
            if self.source_authority in {
                SourceAuthorityClass.NOMINATING,
                SourceAuthorityClass.DIAGNOSTIC,
                SourceAuthorityClass.NONE,
            }:
                raise ProgramLogicAuthorityError(
                    "proved hypotheses require independent authoritative premises"
                )
        if self.disposition is HypothesisDisposition.VALIDATED_REFUTED:
            if self.proof_status is not ProofStatus.VALIDATED_REFUTED:
                raise ProgramLogicAuthorityError(
                    "validated_refuted disposition requires matching proof status"
                )
            if not self.counterexample_target_ref:
                raise ProgramLogicPredictionError(
                    "validated refutation requires a counterexample target reference"
                )
        if self.disposition is HypothesisDisposition.NOMINATED and self.completeness:
            raise ProgramLogicPredictionError(
                "nominated hypotheses cannot claim completeness"
            )
        if self.proof_status is ProofStatus.SOLVER_CHECKED and self.disposition in {
            HypothesisDisposition.PROVED,
            HypothesisDisposition.VALIDATED_REFUTED,
        }:
            raise ProgramLogicAuthorityError(
                "solver-only proof status cannot support proved or validated_refuted dispositions"
            )
        _bounded(self, "logic hypothesis")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "hypothesis_id": self.hypothesis_id,
            "target_goal_id": self.target_goal_id,
            "disposition": self.disposition.value,
            "claimed_consequence_ref": self.claimed_consequence_ref,
            "construction_ref": self.construction_ref,
            "placement_ref": self.placement_ref,
            "value_ref": self.value_ref,
            "evidence_refs": list(self.evidence_refs),
            "evidence_route_kinds": [item.value for item in self.evidence_route_kinds],
            "selected_premise_ids": list(self.selected_premise_ids),
            "counterexample_target_ref": self.counterexample_target_ref,
            "source_authority": self.source_authority.value,
            "proof_status": self.proof_status.value,
            "completeness": self.completeness,
            "unsupported_flags": list(self.unsupported_flags),
            "nomination_score_millipercent": self.nomination_score_millipercent,
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicHypothesis":
        fields = (
            "roots",
            "hypothesis_id",
            "target_goal_id",
            "disposition",
            "claimed_consequence_ref",
            "construction_ref",
            "placement_ref",
            "value_ref",
            "evidence_refs",
            "evidence_route_kinds",
            "selected_premise_ids",
            "counterexample_target_ref",
            "source_authority",
            "proof_status",
            "completeness",
            "unsupported_flags",
            "nomination_score_millipercent",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "logic hypothesis")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicPredictionReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicPredictionReceipt(CanonicalContract):
    """Admission boundary for a reconstructed prediction consequence."""

    SCHEMA: ClassVar[str] = LOGIC_PREDICTION_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    goal_id: str
    hypothesis_id: str
    tactician_plan_id: str
    corpus_id: str
    disposition: PredictionDisposition
    hammer_request_id: str = ""
    translation_id: str = ""
    candidate_id: str = ""
    reconstruction_id: str = ""
    kernel_receipt_id: str = ""
    environment_receipt_id: str = ""
    countermodel_validation_id: str = ""
    derived_clause_ref: str = ""
    derived_value_ref: str = ""
    derived_placement_ref: str = ""
    assumption_refs: tuple[str, ...] = ()
    counterexample_refs: tuple[str, ...] = ()
    residual_gap_ids: tuple[str, ...] = ()
    source_authority: SourceAuthorityClass = SourceAuthorityClass.NONE
    proof_status: ProofStatus = ProofStatus.UNPROVED
    automation_eligible: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "hypothesis_id", _identifier(self.hypothesis_id, "hypothesis_id")
        )
        object.__setattr__(
            self,
            "tactician_plan_id",
            _identifier(self.tactician_plan_id, "tactician_plan_id"),
        )
        object.__setattr__(
            self, "corpus_id", _identifier(self.corpus_id, "corpus_id")
        )
        if self.corpus_id != self.roots.corpus_id:
            raise ProgramLogicAuthorityError(
                "prediction receipt corpus_id must match authority roots corpus_id"
            )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PredictionDisposition, "disposition"),
        )
        for name in (
            "hammer_request_id",
            "translation_id",
            "candidate_id",
            "reconstruction_id",
            "kernel_receipt_id",
            "environment_receipt_id",
            "countermodel_validation_id",
            "derived_clause_ref",
            "derived_value_ref",
            "derived_placement_ref",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self,
            "counterexample_refs",
            _ids(self.counterexample_refs, "counterexample_refs"),
        )
        object.__setattr__(
            self, "residual_gap_ids", _ids(self.residual_gap_ids, "residual_gap_ids")
        )
        object.__setattr__(
            self,
            "source_authority",
            _enum(self.source_authority, SourceAuthorityClass, "source_authority"),
        )
        object.__setattr__(
            self, "proof_status", _enum(self.proof_status, ProofStatus, "proof_status")
        )
        object.__setattr__(
            self,
            "automation_eligible",
            _bool(self.automation_eligible, "automation_eligible"),
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if self.disposition is PredictionDisposition.PROVED:
            if not self.kernel_receipt_id or not self.reconstruction_id:
                raise ProgramLogicAuthorityError(
                    "proved predictions require kernel and reconstruction receipts; "
                    "solver-only verified claims are rejected"
                )
            if self.proof_status is not ProofStatus.KERNEL_VERIFIED:
                raise ProgramLogicAuthorityError(
                    "proved predictions require kernel_verified proof status"
                )
            if self.source_authority not in {
                SourceAuthorityClass.AUTHORITATIVE,
                SourceAuthorityClass.CONFORMANCE,
            }:
                raise ProgramLogicAuthorityError(
                    "proved predictions require authoritative or conformance source authority"
                )
        if self.disposition is PredictionDisposition.VALIDATED_REFUTATION:
            if not self.countermodel_validation_id:
                raise ProgramLogicAuthorityError(
                    "validated refutation requires a countermodel validation receipt; "
                    "raw solver countermodels are insufficient"
                )
            if self.proof_status is not ProofStatus.VALIDATED_REFUTED:
                raise ProgramLogicAuthorityError(
                    "validated refutation requires validated_refuted proof status"
                )
        if self.proof_status is ProofStatus.SOLVER_CHECKED and self.disposition in {
            PredictionDisposition.PROVED,
            PredictionDisposition.VALIDATED_REFUTATION,
        }:
            raise ProgramLogicAuthorityError(
                "solver-only proof status cannot support verified or refuted prediction dispositions"
            )
        if self.automation_eligible and self.disposition not in {
            PredictionDisposition.PROVED,
            PredictionDisposition.VALIDATED_REFUTATION,
        }:
            raise ProgramLogicAuthorityError(
                "automation eligibility requires proved or validated-refutation disposition"
            )
        if self.disposition is PredictionDisposition.STALE and not self.invalidation_refs:
            raise ProgramLogicPredictionError(
                "stale predictions require invalidation refs"
            )
        _bounded(self, "logic prediction receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "tactician_plan_id": self.tactician_plan_id,
            "corpus_id": self.corpus_id,
            "disposition": self.disposition.value,
            "hammer_request_id": self.hammer_request_id,
            "translation_id": self.translation_id,
            "candidate_id": self.candidate_id,
            "reconstruction_id": self.reconstruction_id,
            "kernel_receipt_id": self.kernel_receipt_id,
            "environment_receipt_id": self.environment_receipt_id,
            "countermodel_validation_id": self.countermodel_validation_id,
            "derived_clause_ref": self.derived_clause_ref,
            "derived_value_ref": self.derived_value_ref,
            "derived_placement_ref": self.derived_placement_ref,
            "assumption_refs": list(self.assumption_refs),
            "counterexample_refs": list(self.counterexample_refs),
            "residual_gap_ids": list(self.residual_gap_ids),
            "source_authority": self.source_authority.value,
            "proof_status": self.proof_status.value,
            "automation_eligible": self.automation_eligible,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicPredictionReceipt":
        fields = (
            "roots",
            "receipt_id",
            "goal_id",
            "hypothesis_id",
            "tactician_plan_id",
            "corpus_id",
            "disposition",
            "hammer_request_id",
            "translation_id",
            "candidate_id",
            "reconstruction_id",
            "kernel_receipt_id",
            "environment_receipt_id",
            "countermodel_validation_id",
            "derived_clause_ref",
            "derived_value_ref",
            "derived_placement_ref",
            "assumption_refs",
            "counterexample_refs",
            "residual_gap_ids",
            "source_authority",
            "proof_status",
            "automation_eligible",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "logic prediction receipt"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# ProgramLogicNativeGoalBinding + SemanticRoundTripReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticRoundTripReceipt(CanonicalContract):
    """Proof that a native statement denotes the same admitted LogicIR claim."""

    SCHEMA: ClassVar[str] = SEMANTIC_ROUND_TRIP_RECEIPT_SCHEMA

    receipt_id: str
    logic_ir_claim_id: str
    native_statement_id: str
    equivalence_method: str
    disposition: NativeGoalDisposition
    assumption_refs: tuple[str, ...] = ()
    unsupported_construct_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self,
            "logic_ir_claim_id",
            _identifier(self.logic_ir_claim_id, "logic_ir_claim_id"),
        )
        object.__setattr__(
            self,
            "native_statement_id",
            _identifier(self.native_statement_id, "native_statement_id"),
        )
        object.__setattr__(
            self,
            "equivalence_method",
            _identifier(self.equivalence_method, "equivalence_method"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, NativeGoalDisposition, "disposition"),
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self,
            "unsupported_construct_refs",
            _ids(self.unsupported_construct_refs, "unsupported_construct_refs"),
        )
        if self.disposition is NativeGoalDisposition.ROUND_TRIP_OK:
            if self.unsupported_construct_refs:
                raise ProgramLogicPredictionError(
                    "round-trip ok receipts cannot list unsupported constructs"
                )
        if self.disposition is NativeGoalDisposition.UNSUPPORTED and not (
            self.unsupported_construct_refs
        ):
            raise ProgramLogicPredictionError(
                "unsupported round-trip requires unsupported construct refs"
            )
        _bounded(self, "semantic round trip receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "receipt_id": self.receipt_id,
            "logic_ir_claim_id": self.logic_ir_claim_id,
            "native_statement_id": self.native_statement_id,
            "equivalence_method": self.equivalence_method,
            "disposition": self.disposition.value,
            "assumption_refs": list(self.assumption_refs),
            "unsupported_construct_refs": list(self.unsupported_construct_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRoundTripReceipt":
        fields = (
            "receipt_id",
            "logic_ir_claim_id",
            "native_statement_id",
            "equivalence_method",
            "disposition",
            "assumption_refs",
            "unsupported_construct_refs",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, fields, "semantic round trip receipt")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ProgramLogicNativeGoalBinding(CanonicalContract):
    """Exact GoalSnapshot / native source / kernel binding with round-trip receipt."""

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_NATIVE_GOAL_BINDING_SCHEMA

    roots: ProgramLogicAuthorityRoots
    binding_id: str
    logic_ir_obligation_id: str
    premise_ids: tuple[str, ...]
    native_itp_id: str
    goal_snapshot_id: str
    native_theorem_source_id: str
    proof_hole_id: str
    kernel_id: str
    semantic_round_trip: SemanticRoundTripReceipt
    disposition: NativeGoalDisposition
    import_ids: tuple[str, ...] = ()
    environment_id: str = ""
    source_position_id: str = ""
    unsupported_native_construct_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "binding_id", _identifier(self.binding_id, "binding_id")
        )
        object.__setattr__(
            self,
            "logic_ir_obligation_id",
            _identifier(self.logic_ir_obligation_id, "logic_ir_obligation_id"),
        )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids", required=True)
        )
        object.__setattr__(
            self, "native_itp_id", _identifier(self.native_itp_id, "native_itp_id")
        )
        object.__setattr__(
            self,
            "goal_snapshot_id",
            _identifier(self.goal_snapshot_id, "goal_snapshot_id"),
        )
        object.__setattr__(
            self,
            "native_theorem_source_id",
            _identifier(self.native_theorem_source_id, "native_theorem_source_id"),
        )
        object.__setattr__(
            self, "proof_hole_id", _identifier(self.proof_hole_id, "proof_hole_id")
        )
        object.__setattr__(
            self, "kernel_id", _identifier(self.kernel_id, "kernel_id")
        )
        object.__setattr__(
            self,
            "semantic_round_trip",
            _decode_nested(
                self.semantic_round_trip,
                SemanticRoundTripReceipt,
                "semantic_round_trip",
            ),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, NativeGoalDisposition, "disposition"),
        )
        object.__setattr__(self, "import_ids", _ids(self.import_ids, "import_ids"))
        object.__setattr__(
            self, "environment_id", _text(self.environment_id, "environment_id")
        )
        if self.environment_id and self.environment_id != self.roots.environment_id:
            raise ProgramLogicAuthorityError(
                "native binding environment_id must match authority roots"
            )
        if not self.environment_id:
            object.__setattr__(self, "environment_id", self.roots.environment_id)
        object.__setattr__(
            self,
            "source_position_id",
            _text(self.source_position_id, "source_position_id"),
        )
        object.__setattr__(
            self,
            "unsupported_native_construct_refs",
            _ids(
                self.unsupported_native_construct_refs,
                "unsupported_native_construct_refs",
            ),
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if (
            self.semantic_round_trip.logic_ir_claim_id
            != self.logic_ir_obligation_id
        ):
            raise ProgramLogicAuthorityError(
                "round-trip LogicIR claim must equal binding obligation identity"
            )
        if self.disposition is NativeGoalDisposition.ROUND_TRIP_OK:
            if (
                self.semantic_round_trip.disposition
                is not NativeGoalDisposition.ROUND_TRIP_OK
            ):
                raise ProgramLogicAuthorityError(
                    "round_trip_ok binding requires matching round-trip receipt disposition"
                )
            if self.unsupported_native_construct_refs:
                raise ProgramLogicPredictionError(
                    "round_trip_ok binding cannot list unsupported native constructs"
                )
        if self.disposition is NativeGoalDisposition.INCONSISTENT:
            raise ProgramLogicPredictionError(
                "inconsistent native bindings are rejected at construction"
            )
        if (
            self.roots.toolchain_id
            and self.kernel_id
            and self.kernel_id == "solver-only"
        ):
            raise ProgramLogicAuthorityError(
                "native goal bindings require a kernel identity, not solver-only"
            )
        _bounded(self, "program logic native goal binding")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "binding_id": self.binding_id,
            "logic_ir_obligation_id": self.logic_ir_obligation_id,
            "premise_ids": list(self.premise_ids),
            "native_itp_id": self.native_itp_id,
            "goal_snapshot_id": self.goal_snapshot_id,
            "native_theorem_source_id": self.native_theorem_source_id,
            "proof_hole_id": self.proof_hole_id,
            "kernel_id": self.kernel_id,
            "semantic_round_trip": self.semantic_round_trip.to_dict(),
            "disposition": self.disposition.value,
            "import_ids": list(self.import_ids),
            "environment_id": self.environment_id,
            "source_position_id": self.source_position_id,
            "unsupported_native_construct_refs": list(
                self.unsupported_native_construct_refs
            ),
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicNativeGoalBinding":
        fields = (
            "roots",
            "binding_id",
            "logic_ir_obligation_id",
            "premise_ids",
            "native_itp_id",
            "goal_snapshot_id",
            "native_theorem_source_id",
            "proof_hole_id",
            "kernel_id",
            "semantic_round_trip",
            "disposition",
            "import_ids",
            "environment_id",
            "source_position_id",
            "unsupported_native_construct_refs",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "program logic native goal binding"
        )
        values["roots"] = _roots(values["roots"])
        values["semantic_round_trip"] = _decode_nested(
            values["semantic_round_trip"],
            SemanticRoundTripReceipt,
            "semantic_round_trip",
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# CountermodelValidationReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountermodelValidationReceipt(CanonicalContract):
    """Separates raw solver diagnostics from replayed rejection evidence."""

    SCHEMA: ClassVar[str] = COUNTERMODEL_VALIDATION_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    solver_countermodel_id: str
    translation_map_id: str
    originating_logic_ir_id: str
    disposition: CountermodelDisposition
    raw_diagnostic_refs: tuple[str, ...] = ()
    replayed_rejection_evidence_refs: tuple[str, ...] = ()
    proof_of_negation_id: str = ""
    replay_method: str = ""
    assumption_refs: tuple[str, ...] = ()
    toolchain_id: str = ""
    policy_id: str = ""
    resource_policy_ref: str = ""
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self,
            "solver_countermodel_id",
            _identifier(self.solver_countermodel_id, "solver_countermodel_id"),
        )
        object.__setattr__(
            self,
            "translation_map_id",
            _identifier(self.translation_map_id, "translation_map_id"),
        )
        object.__setattr__(
            self,
            "originating_logic_ir_id",
            _identifier(self.originating_logic_ir_id, "originating_logic_ir_id"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CountermodelDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "raw_diagnostic_refs",
            _ids(self.raw_diagnostic_refs, "raw_diagnostic_refs"),
        )
        object.__setattr__(
            self,
            "replayed_rejection_evidence_refs",
            _ids(
                self.replayed_rejection_evidence_refs,
                "replayed_rejection_evidence_refs",
            ),
        )
        object.__setattr__(
            self,
            "proof_of_negation_id",
            _text(self.proof_of_negation_id, "proof_of_negation_id"),
        )
        object.__setattr__(
            self, "replay_method", _text(self.replay_method, "replay_method")
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self, "toolchain_id", _text(self.toolchain_id, "toolchain_id")
        )
        if self.toolchain_id and self.toolchain_id != self.roots.toolchain_id:
            raise ProgramLogicAuthorityError(
                "countermodel toolchain_id must match authority roots"
            )
        if not self.toolchain_id:
            object.__setattr__(self, "toolchain_id", self.roots.toolchain_id)
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        if self.policy_id and self.policy_id != self.roots.policy_id:
            raise ProgramLogicAuthorityError(
                "countermodel policy_id must match authority roots"
            )
        if not self.policy_id:
            object.__setattr__(self, "policy_id", self.roots.policy_id)
        object.__setattr__(
            self,
            "resource_policy_ref",
            _text(self.resource_policy_ref, "resource_policy_ref"),
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        raw = set(self.raw_diagnostic_refs)
        replayed = set(self.replayed_rejection_evidence_refs)
        if raw & replayed:
            raise ProgramLogicPredictionError(
                "raw diagnostic refs and replayed rejection evidence must be disjoint"
            )
        if self.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY:
            if self.replayed_rejection_evidence_refs or self.proof_of_negation_id:
                raise ProgramLogicAuthorityError(
                    "diagnostic-only countermodels cannot carry rejection evidence"
                )
            if not self.raw_diagnostic_refs:
                raise ProgramLogicPredictionError(
                    "diagnostic-only countermodels require raw diagnostic refs"
                )
        if self.disposition is CountermodelDisposition.VALIDATED:
            if not (
                self.replayed_rejection_evidence_refs or self.proof_of_negation_id
            ):
                raise ProgramLogicAuthorityError(
                    "validated countermodels require replayed rejection evidence "
                    "or a proof of negation"
                )
            if not self.replay_method and not self.proof_of_negation_id:
                raise ProgramLogicPredictionError(
                    "validated countermodels require a replay method or proof of negation"
                )
        if self.disposition is CountermodelDisposition.INCONSISTENT:
            raise ProgramLogicPredictionError(
                "inconsistent countermodel receipts are rejected at construction"
            )
        _bounded(self, "countermodel validation receipt")

    @property
    def may_reject_hypothesis(self) -> bool:
        """Only independently validated receipts may narrow or reject."""
        return self.disposition is CountermodelDisposition.VALIDATED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "solver_countermodel_id": self.solver_countermodel_id,
            "translation_map_id": self.translation_map_id,
            "originating_logic_ir_id": self.originating_logic_ir_id,
            "disposition": self.disposition.value,
            "raw_diagnostic_refs": list(self.raw_diagnostic_refs),
            "replayed_rejection_evidence_refs": list(
                self.replayed_rejection_evidence_refs
            ),
            "proof_of_negation_id": self.proof_of_negation_id,
            "replay_method": self.replay_method,
            "assumption_refs": list(self.assumption_refs),
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "resource_policy_ref": self.resource_policy_ref,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CountermodelValidationReceipt":
        fields = (
            "roots",
            "receipt_id",
            "solver_countermodel_id",
            "translation_map_id",
            "originating_logic_ir_id",
            "disposition",
            "raw_diagnostic_refs",
            "replayed_rejection_evidence_refs",
            "proof_of_negation_id",
            "replay_method",
            "assumption_refs",
            "toolchain_id",
            "policy_id",
            "resource_policy_ref",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "countermodel validation receipt"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicGuidedRepairPacket (context overlay; not write authority)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicGuidedRepairPacket(CanonicalContract):
    """Context overlay over an existing RPR packet/plan/lease authority.

    This record never originates write scope.  Any permitted write paths must
    already be granted by an admitted RPR plan and writer lease referenced here.
    """

    SCHEMA: ClassVar[str] = LOGIC_GUIDED_REPAIR_PACKET_SCHEMA

    roots: ProgramLogicAuthorityRoots
    packet_id: str
    admitted_prediction_id: str
    rpr_packet_id: str
    rpr_plan_id: str
    rpr_plan_step_id: str
    writer_lease_id: str
    disposition: ContextOverlayDisposition
    context_capsule_id: str
    scope_path_refs: tuple[str, ...] = ()
    before_hash_refs: tuple[str, ...] = ()
    permitted_read_paths: tuple[str, ...] = ()
    permitted_write_paths: tuple[str, ...] = ()
    forbidden_path_refs: tuple[str, ...] = ()
    forbidden_semantic_change_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    rollback_policy_ref: str = ""
    expansion_handle_refs: tuple[str, ...] = ()
    provider_id: str = ""
    model_id: str = ""
    config_id: str = ""
    write_authority: bool = False
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "packet_id", _identifier(self.packet_id, "packet_id")
        )
        object.__setattr__(
            self,
            "admitted_prediction_id",
            _identifier(self.admitted_prediction_id, "admitted_prediction_id"),
        )
        object.__setattr__(
            self, "rpr_packet_id", _text(self.rpr_packet_id, "rpr_packet_id")
        )
        object.__setattr__(
            self, "rpr_plan_id", _text(self.rpr_plan_id, "rpr_plan_id")
        )
        object.__setattr__(
            self,
            "rpr_plan_step_id",
            _text(self.rpr_plan_step_id, "rpr_plan_step_id"),
        )
        object.__setattr__(
            self,
            "writer_lease_id",
            _text(self.writer_lease_id, "writer_lease_id"),
        )
        for required_when_named in (
            "rpr_packet_id",
            "rpr_plan_id",
            "rpr_plan_step_id",
            "writer_lease_id",
        ):
            value = getattr(self, required_when_named)
            if value and any(char.isspace() for char in value):
                raise ProgramLogicPredictionError(
                    f"{required_when_named} must be an opaque compact identifier"
                )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ContextOverlayDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "context_capsule_id",
            _identifier(self.context_capsule_id, "context_capsule_id"),
        )
        object.__setattr__(
            self, "scope_path_refs", _ids(self.scope_path_refs, "scope_path_refs")
        )
        object.__setattr__(
            self, "before_hash_refs", _ids(self.before_hash_refs, "before_hash_refs")
        )
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
            "forbidden_path_refs",
            _ids(self.forbidden_path_refs, "forbidden_path_refs"),
        )
        object.__setattr__(
            self,
            "forbidden_semantic_change_refs",
            _ids(
                self.forbidden_semantic_change_refs, "forbidden_semantic_change_refs"
            ),
        )
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self,
            "rollback_policy_ref",
            _text(self.rollback_policy_ref, "rollback_policy_ref"),
        )
        object.__setattr__(
            self,
            "expansion_handle_refs",
            _ids(self.expansion_handle_refs, "expansion_handle_refs"),
        )
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "config_id", _text(self.config_id, "config_id"))
        # Overlay never originates write or semantic authority.
        if self.write_authority is not False:
            raise ProgramLogicAuthorityError(
                "logic-guided repair packets cannot claim write authority; "
                "they overlay existing RPR plan/lease authority only"
            )
        object.__setattr__(self, "write_authority", False)
        if self.semantic_authority is not False:
            raise ProgramLogicAuthorityError(
                "logic-guided repair packets cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        # Write scope without an admitted RPR plan/lease is forbidden.
        if self.permitted_write_paths and not (
            self.rpr_plan_id and self.writer_lease_id and self.rpr_packet_id
        ):
            raise ProgramLogicAuthorityError(
                "write scope requires an existing admitted RPR plan and writer lease"
            )
        if self.disposition in {
            ContextOverlayDisposition.ABSTAINED,
            ContextOverlayDisposition.REJECTED,
        } and self.permitted_write_paths:
            raise ProgramLogicAuthorityError(
                "abstained/rejected overlays cannot carry write paths"
            )
        if (
            self.disposition is ContextOverlayDisposition.MODEL_REQUIRED
            and not self.model_id
        ):
            raise ProgramLogicPredictionError(
                "model_required overlays require a model identity"
            )
        if (
            self.disposition is ContextOverlayDisposition.DETERMINISTIC
            and self.model_id
        ):
            raise ProgramLogicPredictionError(
                "deterministic overlays must not bind a model identity"
            )
        _bounded(self, "logic guided repair packet")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "packet_id": self.packet_id,
            "admitted_prediction_id": self.admitted_prediction_id,
            "rpr_packet_id": self.rpr_packet_id,
            "rpr_plan_id": self.rpr_plan_id,
            "rpr_plan_step_id": self.rpr_plan_step_id,
            "writer_lease_id": self.writer_lease_id,
            "disposition": self.disposition.value,
            "context_capsule_id": self.context_capsule_id,
            "scope_path_refs": list(self.scope_path_refs),
            "before_hash_refs": list(self.before_hash_refs),
            "permitted_read_paths": list(self.permitted_read_paths),
            "permitted_write_paths": list(self.permitted_write_paths),
            "forbidden_path_refs": list(self.forbidden_path_refs),
            "forbidden_semantic_change_refs": list(self.forbidden_semantic_change_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "validation_refs": list(self.validation_refs),
            "rollback_policy_ref": self.rollback_policy_ref,
            "expansion_handle_refs": list(self.expansion_handle_refs),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "write_authority": False,
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicGuidedRepairPacket":
        fields = (
            "roots",
            "packet_id",
            "admitted_prediction_id",
            "rpr_packet_id",
            "rpr_plan_id",
            "rpr_plan_step_id",
            "writer_lease_id",
            "disposition",
            "context_capsule_id",
            "scope_path_refs",
            "before_hash_refs",
            "permitted_read_paths",
            "permitted_write_paths",
            "forbidden_path_refs",
            "forbidden_semantic_change_refs",
            "postcondition_refs",
            "validation_refs",
            "rollback_policy_ref",
            "expansion_handle_refs",
            "provider_id",
            "model_id",
            "config_id",
            "write_authority",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "logic guided repair packet"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicFixedPointEvidenceAttachment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicFixedPointEvidenceAttachment(CanonicalContract):
    """Extends an existing completion receipt with per-iteration logic evidence.

    Does not replace :class:`~.change_propagation_contracts.PropagationCompletionReceipt`
    or contract-repair completion; it only attaches additional roots and residuals.
    """

    SCHEMA: ClassVar[str] = LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    attachment_id: str
    completion_receipt_id: str
    disposition: FixedPointAttachmentDisposition
    iteration_count: int
    goal_root_ids: tuple[str, ...] = ()
    corpus_root_ids: tuple[str, ...] = ()
    tactician_plan_ids: tuple[str, ...] = ()
    hammer_receipt_ids: tuple[str, ...] = ()
    prediction_receipt_ids: tuple[str, ...] = ()
    original_consumer_coverage_ids: tuple[str, ...] = ()
    second_order_consumer_coverage_ids: tuple[str, ...] = ()
    residual_logic_gap_ids: tuple[str, ...] = ()
    unsupported_logic_gap_ids: tuple[str, ...] = ()
    finalize_receipt_id: str = ""
    compensating_rollback_receipt_id: str = ""
    replaces_completion: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "attachment_id", _identifier(self.attachment_id, "attachment_id")
        )
        object.__setattr__(
            self,
            "completion_receipt_id",
            _identifier(self.completion_receipt_id, "completion_receipt_id"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, FixedPointAttachmentDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "iteration_count",
            _bounded_int(self.iteration_count, "iteration_count", minimum=1),
        )
        object.__setattr__(
            self, "goal_root_ids", _ids(self.goal_root_ids, "goal_root_ids")
        )
        object.__setattr__(
            self, "corpus_root_ids", _ids(self.corpus_root_ids, "corpus_root_ids")
        )
        object.__setattr__(
            self,
            "tactician_plan_ids",
            _ids(self.tactician_plan_ids, "tactician_plan_ids"),
        )
        object.__setattr__(
            self,
            "hammer_receipt_ids",
            _ids(self.hammer_receipt_ids, "hammer_receipt_ids"),
        )
        object.__setattr__(
            self,
            "prediction_receipt_ids",
            _ids(self.prediction_receipt_ids, "prediction_receipt_ids"),
        )
        object.__setattr__(
            self,
            "original_consumer_coverage_ids",
            _ids(
                self.original_consumer_coverage_ids, "original_consumer_coverage_ids"
            ),
        )
        object.__setattr__(
            self,
            "second_order_consumer_coverage_ids",
            _ids(
                self.second_order_consumer_coverage_ids,
                "second_order_consumer_coverage_ids",
            ),
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
            "finalize_receipt_id",
            _text(self.finalize_receipt_id, "finalize_receipt_id"),
        )
        object.__setattr__(
            self,
            "compensating_rollback_receipt_id",
            _text(
                self.compensating_rollback_receipt_id,
                "compensating_rollback_receipt_id",
            ),
        )
        if self.replaces_completion is not False:
            raise ProgramLogicAuthorityError(
                "logic fixed-point evidence attachments extend rather than replace "
                "existing completion receipts"
            )
        object.__setattr__(self, "replaces_completion", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if self.disposition is FixedPointAttachmentDisposition.ATTACHED:
            if self.residual_logic_gap_ids:
                raise ProgramLogicPredictionError(
                    "attached disposition forbids residual logic gaps"
                )
            if not self.finalize_receipt_id:
                raise ProgramLogicPredictionError(
                    "attached disposition requires a finalize receipt"
                )
            if self.compensating_rollback_receipt_id:
                raise ProgramLogicPredictionError(
                    "attached disposition cannot carry a compensating rollback receipt"
                )
        if self.disposition is FixedPointAttachmentDisposition.ROLLED_BACK:
            if not self.compensating_rollback_receipt_id:
                raise ProgramLogicPredictionError(
                    "rolled_back disposition requires a compensating rollback receipt"
                )
            if self.finalize_receipt_id:
                raise ProgramLogicPredictionError(
                    "rolled_back disposition cannot carry a finalize receipt"
                )
        if self.disposition is FixedPointAttachmentDisposition.RESIDUAL and not (
            self.residual_logic_gap_ids or self.unsupported_logic_gap_ids
        ):
            raise ProgramLogicPredictionError(
                "residual disposition requires residual or unsupported logic gaps"
            )
        if (
            self.finalize_receipt_id
            and self.compensating_rollback_receipt_id
        ):
            raise ProgramLogicPredictionError(
                "finalize and compensating rollback receipts are mutually exclusive"
            )
        _bounded(self, "logic fixed point evidence attachment")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREDICTION_VERSION,
            "roots": self.roots.to_dict(),
            "attachment_id": self.attachment_id,
            "completion_receipt_id": self.completion_receipt_id,
            "disposition": self.disposition.value,
            "iteration_count": self.iteration_count,
            "goal_root_ids": list(self.goal_root_ids),
            "corpus_root_ids": list(self.corpus_root_ids),
            "tactician_plan_ids": list(self.tactician_plan_ids),
            "hammer_receipt_ids": list(self.hammer_receipt_ids),
            "prediction_receipt_ids": list(self.prediction_receipt_ids),
            "original_consumer_coverage_ids": list(
                self.original_consumer_coverage_ids
            ),
            "second_order_consumer_coverage_ids": list(
                self.second_order_consumer_coverage_ids
            ),
            "residual_logic_gap_ids": list(self.residual_logic_gap_ids),
            "unsupported_logic_gap_ids": list(self.unsupported_logic_gap_ids),
            "finalize_receipt_id": self.finalize_receipt_id,
            "compensating_rollback_receipt_id": self.compensating_rollback_receipt_id,
            "replaces_completion": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LogicFixedPointEvidenceAttachment":
        fields = (
            "roots",
            "attachment_id",
            "completion_receipt_id",
            "disposition",
            "iteration_count",
            "goal_root_ids",
            "corpus_root_ids",
            "tactician_plan_ids",
            "hammer_receipt_ids",
            "prediction_receipt_ids",
            "original_consumer_coverage_ids",
            "second_order_consumer_coverage_ids",
            "residual_logic_gap_ids",
            "unsupported_logic_gap_ids",
            "finalize_receipt_id",
            "compensating_rollback_receipt_id",
            "replaces_completion",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "logic fixed point evidence attachment"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


__all__ = [
    "PROGRAM_LOGIC_PREDICTION_VERSION",
    "MAX_RECORD_BYTES",
    "MAX_REFERENCE_COUNT",
    "MAX_SUBGOAL_COUNT",
    "PROGRAM_LOGIC_ROOTS_SCHEMA",
    "PROGRAM_LOGIC_GOAL_SCHEMA",
    "LOGIC_GAP_SCHEMA",
    "LOGIC_SUBGOAL_SCHEMA",
    "TACTICIAN_SEARCH_PLAN_SCHEMA",
    "LOGIC_HYPOTHESIS_SCHEMA",
    "LOGIC_PREDICTION_RECEIPT_SCHEMA",
    "SEMANTIC_ROUND_TRIP_RECEIPT_SCHEMA",
    "PROGRAM_LOGIC_NATIVE_GOAL_BINDING_SCHEMA",
    "COUNTERMODEL_VALIDATION_RECEIPT_SCHEMA",
    "LOGIC_GUIDED_REPAIR_PACKET_SCHEMA",
    "LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_SCHEMA",
    "ProgramLogicPredictionError",
    "ProgramLogicPredictionBoundsError",
    "ForgedProgramLogicIdentityError",
    "ProgramLogicAuthorityError",
    "GoalDisposition",
    "GapDisposition",
    "SourceRouteKind",
    "SourceAuthorityClass",
    "ProofStatus",
    "SubgoalDisposition",
    "HypothesisDisposition",
    "PredictionDisposition",
    "NativeGoalDisposition",
    "CountermodelDisposition",
    "ContextOverlayDisposition",
    "FixedPointAttachmentDisposition",
    "GoalFamily",
    "LogicFacetKind",
    "GapMissingClass",
    "ProgramLogicAuthorityRoots",
    "LogicFacetRef",
    "ProgramLogicGoal",
    "LogicGap",
    "LogicSubgoal",
    "TacticianSearchPlan",
    "LogicHypothesis",
    "LogicPredictionReceipt",
    "SemanticRoundTripReceipt",
    "ProgramLogicNativeGoalBinding",
    "CountermodelValidationReceipt",
    "LogicGuidedRepairPacket",
    "LogicFixedPointEvidenceAttachment",
    "content_identity",
]
