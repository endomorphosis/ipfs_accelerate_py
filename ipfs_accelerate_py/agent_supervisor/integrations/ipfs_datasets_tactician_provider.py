"""Supervisor adapter that maps program-repair goals onto the generic datasets
Logic Tactician (``ipfs_datasets_py.logic.tactician@1``).

Design constraints (fail-closed):

* Construction and capability inspection never import the optional package.
* Program source categories are a closed vocabulary ordered by an explicit
  local-first precedence; authoritative routes precede approximate/model routes.
* Actual optional-provider planning executes only through a bounded, referenced
  adapter surface (injected backend or lazy ``LogicTactician``).
* Every query identity, result identity, and exclusion is recorded.
* Stale, cross-root, malformed, free-form-authority, unbounded, and unsupported
  source types are rejected with typed statuses.
* Responses are deterministic and always carry ``semantic_authority=false``.
* An unavailable optional provider yields a typed abstention rather than a
  guessed plan.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicPredictionError,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ..proof.formal_verification_contracts import (
    canonical_json,
    content_identity,
)


# ---------------------------------------------------------------------------
# Versioning / schemas
# ---------------------------------------------------------------------------

IPFS_DATASETS_TACTICIAN_PROVIDER_ID: Final = "ipfs_datasets_py.logic.tactician"
IPFS_DATASETS_TACTICIAN_PROVIDER_VERSION: Final = "1.0.0"
GENERIC_TACTICIAN_INTERFACE: Final = "ipfs_datasets_py.logic.tactician@1"
GENERIC_TACTICIAN_MODULE: Final = "ipfs_datasets_py.logic.tactician"
CODE_TACTICIAN_PLANNER_ID: Final = (
    "ipfs_accelerate_py.agent_supervisor.code-tactician@1"
)

PROVIDER_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-tactician-capability@1"
)
PROVIDER_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-tactician-request@1"
)
PROVIDER_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-tactician-response@1"
)
PROVIDER_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-tactician-policy@1"
)
QUERY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-tactician-query-receipt@1"
)

DEFAULT_MAX_SOURCES: Final = 32
DEFAULT_MAX_ROUTES: Final = 32
DEFAULT_MAX_SUBGOALS: Final = 16
DEFAULT_MAX_QUERY_HINTS: Final = 16
DEFAULT_MAX_REFINEMENT_ROUNDS: Final = 4
DEFAULT_MAX_PREMISES: Final = 64
DEFAULT_MAX_QUERIES: Final = 64
DEFAULT_MAX_GOALS: Final = 32
HARD_MAX_SOURCES: Final = 128
HARD_MAX_ROUTES: Final = 128
HARD_MAX_SUBGOALS: Final = 64
HARD_MAX_PREMISES: Final = 256
HARD_MAX_QUERIES: Final = 256
HARD_MAX_GOALS: Final = 64
HARD_MAX_QUERY_BYTES: Final = 16 * 1024

_IMPORT_LOCK: Final = threading.Lock()


# ---------------------------------------------------------------------------
# Closed program-domain source taxonomy (plan §4.4 + acceptance)
# ---------------------------------------------------------------------------


class CodeSourceType(str, Enum):
    """Closed program-repair source categories for Code Tactician routing."""

    AUTHORITATIVE_CONTRACT = "authoritative_contract"
    TYPE_AND_EFFECT_FACTS = "type_and_effect_facts"
    VALUE_PROVENANCE = "value_provenance"
    PROGRAM_GRAPH = "program_graph"
    SCHEMA_PROTOCOL = "schema_protocol"
    TESTS_AND_SPECS = "tests_and_specs"
    GIT_LINEAGE = "git_lineage"
    THEOREM_CORPUS = "theorem_corpus"
    VECTOR_ANALOGUE = "vector_analogue"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    RUNTIME_WITNESS = "runtime_witness"
    MODEL_HYPOTHESIS = "model_hypothesis"


# Local-first precedence: lower rank is earlier. Authoritative/local exact
# facts precede lexical/vector/history nominations, then model hypotheses.
CODE_SOURCE_PRECEDENCE: Final[tuple[CodeSourceType, ...]] = (
    CodeSourceType.AUTHORITATIVE_CONTRACT,
    CodeSourceType.TYPE_AND_EFFECT_FACTS,
    CodeSourceType.VALUE_PROVENANCE,
    CodeSourceType.PROGRAM_GRAPH,
    CodeSourceType.SCHEMA_PROTOCOL,
    CodeSourceType.TESTS_AND_SPECS,
    CodeSourceType.THEOREM_CORPUS,
    CodeSourceType.GIT_LINEAGE,
    CodeSourceType.VECTOR_ANALOGUE,
    CodeSourceType.KNOWLEDGE_GRAPH,
    CodeSourceType.RUNTIME_WITNESS,
    CodeSourceType.MODEL_HYPOTHESIS,
)

_LOCAL_AUTHORITATIVE_SOURCES: Final[frozenset[CodeSourceType]] = frozenset(
    {
        CodeSourceType.AUTHORITATIVE_CONTRACT,
        CodeSourceType.TYPE_AND_EFFECT_FACTS,
        CodeSourceType.VALUE_PROVENANCE,
        CodeSourceType.PROGRAM_GRAPH,
        CodeSourceType.SCHEMA_PROTOCOL,
        CodeSourceType.TESTS_AND_SPECS,
        CodeSourceType.THEOREM_CORPUS,
    }
)

_APPROXIMATE_MODEL_SOURCES: Final[frozenset[CodeSourceType]] = frozenset(
    {
        CodeSourceType.GIT_LINEAGE,
        CodeSourceType.VECTOR_ANALOGUE,
        CodeSourceType.KNOWLEDGE_GRAPH,
        CodeSourceType.RUNTIME_WITNESS,
        CodeSourceType.MODEL_HYPOTHESIS,
    }
)

_CODE_SOURCE_TO_ROUTE: Final[dict[CodeSourceType, SourceRouteKind]] = {
    CodeSourceType.AUTHORITATIVE_CONTRACT: SourceRouteKind.REVIEWED_CONTRACT,
    CodeSourceType.TYPE_AND_EFFECT_FACTS: SourceRouteKind.LOCAL_STATIC,
    CodeSourceType.VALUE_PROVENANCE: SourceRouteKind.DATAFLOW,
    CodeSourceType.PROGRAM_GRAPH: SourceRouteKind.GRAPH,
    CodeSourceType.SCHEMA_PROTOCOL: SourceRouteKind.LOCAL_STATIC,
    CodeSourceType.TESTS_AND_SPECS: SourceRouteKind.REVIEWED_TEST,
    CodeSourceType.GIT_LINEAGE: SourceRouteKind.HISTORY,
    CodeSourceType.THEOREM_CORPUS: SourceRouteKind.LOCAL_STATIC,
    CodeSourceType.VECTOR_ANALOGUE: SourceRouteKind.VECTOR,
    CodeSourceType.KNOWLEDGE_GRAPH: SourceRouteKind.KNOWLEDGE_GRAPH,
    CodeSourceType.RUNTIME_WITNESS: SourceRouteKind.RUNTIME_WITNESS,
    CodeSourceType.MODEL_HYPOTHESIS: SourceRouteKind.LLM,
}

_CODE_SOURCE_AUTHORITY: Final[dict[CodeSourceType, SourceAuthorityClass]] = {
    CodeSourceType.AUTHORITATIVE_CONTRACT: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.TYPE_AND_EFFECT_FACTS: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.VALUE_PROVENANCE: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.PROGRAM_GRAPH: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.SCHEMA_PROTOCOL: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.TESTS_AND_SPECS: SourceAuthorityClass.CONFORMANCE,
    CodeSourceType.GIT_LINEAGE: SourceAuthorityClass.NOMINATING,
    CodeSourceType.THEOREM_CORPUS: SourceAuthorityClass.AUTHORITATIVE,
    CodeSourceType.VECTOR_ANALOGUE: SourceAuthorityClass.NOMINATING,
    CodeSourceType.KNOWLEDGE_GRAPH: SourceAuthorityClass.NOMINATING,
    CodeSourceType.RUNTIME_WITNESS: SourceAuthorityClass.DIAGNOSTIC,
    CodeSourceType.MODEL_HYPOTHESIS: SourceAuthorityClass.NOMINATING,
}

_PREMISE_TO_CODE_SOURCE: Final[dict[PremiseSourceClass, CodeSourceType]] = {
    PremiseSourceClass.REVIEWED_CONTRACT: CodeSourceType.AUTHORITATIVE_CONTRACT,
    PremiseSourceClass.NORMATIVE_SPEC: CodeSourceType.TESTS_AND_SPECS,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST: CodeSourceType.TESTS_AND_SPECS,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS: CodeSourceType.TYPE_AND_EFFECT_FACTS,
    PremiseSourceClass.VALUE_PROVENANCE: CodeSourceType.VALUE_PROVENANCE,
    PremiseSourceClass.PROGRAM_GRAPH: CodeSourceType.PROGRAM_GRAPH,
    PremiseSourceClass.SCHEMA_PROTOCOL: CodeSourceType.SCHEMA_PROTOCOL,
    PremiseSourceClass.LOCAL_STATIC: CodeSourceType.TYPE_AND_EFFECT_FACTS,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION: CodeSourceType.TYPE_AND_EFFECT_FACTS,
    PremiseSourceClass.COMMENT: CodeSourceType.GIT_LINEAGE,
    PremiseSourceClass.RUNTIME_WITNESS: CodeSourceType.RUNTIME_WITNESS,
    PremiseSourceClass.HISTORY: CodeSourceType.GIT_LINEAGE,
    PremiseSourceClass.VECTOR_ANALOGUE: CodeSourceType.VECTOR_ANALOGUE,
    PremiseSourceClass.KNOWLEDGE_GRAPH: CodeSourceType.KNOWLEDGE_GRAPH,
    PremiseSourceClass.MODEL_HYPOTHESIS: CodeSourceType.MODEL_HYPOTHESIS,
    PremiseSourceClass.THEOREM_CORPUS: CodeSourceType.THEOREM_CORPUS,
    PremiseSourceClass.GIT_LINEAGE: CodeSourceType.GIT_LINEAGE,
}

# Source types that may dispatch a real optional-provider query only through
# a bounded, referenced adapter (never free-form network/body execution).
_QUERYABLE_SOURCE_TYPES: Final[frozenset[CodeSourceType]] = frozenset(
    {
        CodeSourceType.VECTOR_ANALOGUE,
        CodeSourceType.KNOWLEDGE_GRAPH,
        CodeSourceType.GIT_LINEAGE,
        CodeSourceType.THEOREM_CORPUS,
        CodeSourceType.RUNTIME_WITNESS,
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_code",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw",
        "prompt",
        "completion",
        "transcript",
        "embedding",
        "theorem_text",
        "proof_script",
    }
)


class CodeTacticianStatus(str, Enum):
    """Closed adapter outcomes; only PLANNED yields a search plan."""

    PLANNED = "planned"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    MALFORMED = "malformed"
    UNSUPPORTED = "unsupported"
    POLICY_DENIED = "policy_denied"


class CodeTacticianReasonCode(str, Enum):
    """Machine-readable reason for non-planned outcomes."""

    OK = "ok"
    OPTIONAL_MODULE_UNAVAILABLE = "optional_module_unavailable"
    INTERFACE_INCOMPATIBLE = "interface_incompatible"
    CROSS_ROOT = "cross_root"
    STALE_ROOTS = "stale_roots"
    MALFORMED_REQUEST = "malformed_request"
    FREE_FORM_AUTHORITY = "free_form_authority"
    UNBOUNDED_PLAN = "unbounded_plan"
    UNSUPPORTED_SOURCE_TYPE = "unsupported_source_type"
    NO_ADMISSIBLE_SOURCES = "no_admissible_sources"
    EMPTY_GOALS = "empty_goals"
    PLANNER_ABSTAINED = "planner_abstained"
    PLANNER_ERROR = "planner_error"
    QUERY_ADAPTER_MISSING = "query_adapter_missing"
    QUERY_BOUNDS_EXCEEDED = "query_bounds_exceeded"
    QUERY_UNSUPPORTED = "query_unsupported"
    INTERNAL_ERROR = "internal_error"


class CodeTacticianError(ValueError):
    """Raised for local request/policy violations before optional import."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise CodeTacticianError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise CodeTacticianError(f"{field_name} must not be empty")
    return result


def _positive_int(
    value: Any,
    *,
    field_name: str,
    maximum: int,
    default: int | None = None,
) -> int:
    if value is None:
        if default is None:
            raise CodeTacticianError(f"{field_name} is required")
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CodeTacticianError(f"{field_name} must be a positive integer")
    if value > maximum:
        raise CodeTacticianError(
            f"{field_name}={value} exceeds hard maximum {maximum}"
        )
    return value


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CodeTacticianError(f"{field_name} must be a boolean")
    return value


def _assert_body_free(value: Any, *, field_name: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodeTacticianError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise CodeTacticianError(
                    f"{field_name} may not contain free-form body field {key!r}"
                )
            if normalized in {
                "semantic_authority",
                "expectation_authority",
                "proof_authority",
                "write_authority",
            } and item is True:
                raise CodeTacticianError(
                    f"{field_name} rejects free-form authority promotion via {key}"
                )
            _assert_body_free(item, field_name=field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, field_name=field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise CodeTacticianError(f"{field_name} may not contain binary bodies")


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _roots_payload(roots: ProgramLogicAuthorityRoots) -> dict[str, str]:
    return {
        "repository_id": roots.repository_id,
        "objective_id": roots.objective_id,
        "trace_id": roots.trace_id,
        "change_id": roots.change_id,
        "consumer_id": roots.consumer_id,
        "forest_id": roots.forest_id,
        "tree_id": roots.tree_id,
        "overlay_id": roots.overlay_id,
        "graph_id": roots.graph_id,
        "index_id": roots.index_id,
        "corpus_id": roots.corpus_id,
        "model_id": roots.model_id,
        "translator_id": roots.translator_id,
        "toolchain_id": roots.toolchain_id,
        "policy_id": roots.policy_id,
        "environment_id": roots.environment_id,
        "content_id": roots.content_id,
    }


def code_source_rank(source_type: CodeSourceType) -> int:
    try:
        return CODE_SOURCE_PRECEDENCE.index(source_type)
    except ValueError:
        return len(CODE_SOURCE_PRECEDENCE)


def is_local_authoritative(source_type: CodeSourceType) -> bool:
    return source_type in _LOCAL_AUTHORITATIVE_SOURCES


def is_approximate_or_model(source_type: CodeSourceType) -> bool:
    return source_type in _APPROXIMATE_MODEL_SOURCES


def map_premise_source(source_class: PremiseSourceClass | str) -> CodeSourceType:
    if isinstance(source_class, PremiseSourceClass):
        key = source_class
    else:
        try:
            key = PremiseSourceClass(str(source_class))
        except ValueError as exc:
            raise CodeTacticianError(
                f"unsupported premise source class {source_class!r}"
            ) from exc
    try:
        return _PREMISE_TO_CODE_SOURCE[key]
    except KeyError as exc:
        raise CodeTacticianError(
            f"unsupported premise source class {key!r}"
        ) from exc


def map_code_source_to_route(source_type: CodeSourceType) -> SourceRouteKind:
    return _CODE_SOURCE_TO_ROUTE[source_type]


def parse_code_source_type(value: Any) -> CodeSourceType:
    if isinstance(value, CodeSourceType):
        return value
    try:
        return CodeSourceType(str(value).strip())
    except ValueError as exc:
        raise CodeTacticianError(
            f"unsupported source type {value!r}; "
            f"allowed={sorted(item.value for item in CodeSourceType)}"
        ) from exc


# ---------------------------------------------------------------------------
# Policy / request / response records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeTacticianPolicy:
    """Operator-controlled bounds for program-domain Tactician planning.

    Capability flags that would imply network, write, proof, or semantic
    authority remain fixed closed.
    """

    policy_id: str = "code-tactician.policy.default@1"
    source_class_order: tuple[CodeSourceType, ...] = CODE_SOURCE_PRECEDENCE
    max_sources: int = DEFAULT_MAX_SOURCES
    max_routes: int = DEFAULT_MAX_ROUTES
    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    max_query_hints_per_source: int = DEFAULT_MAX_QUERY_HINTS
    max_refinement_rounds: int = DEFAULT_MAX_REFINEMENT_ROUNDS
    max_premises: int = DEFAULT_MAX_PREMISES
    max_queries: int = DEFAULT_MAX_QUERIES
    denied_source_types: tuple[CodeSourceType, ...] = ()
    allow_approximate_routes: bool = True
    allow_model_hypothesis: bool = False
    require_local_before_approximate: bool = True
    network_allowed: bool = False
    write_allowed: bool = False
    proof_execution_allowed: bool = False
    semantic_authority: bool = False
    stop_policy_ref: str = "stop:code-tactician.default@1"
    escalation_policy_ref: str = "escalation:code-tactician.default@1"
    abstention_policy_ref: str = "abstention:code-tactician.default@1"
    resource_policy_ref: str = "resource:code-tactician.default@1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        order = tuple(
            parse_code_source_type(item) for item in self.source_class_order
        )
        if not order:
            raise CodeTacticianError("source_class_order must not be empty")
        if len(order) != len(set(order)):
            raise CodeTacticianError("source_class_order must be unique")
        # Local authoritative routes must precede approximate/model routes.
        if self.require_local_before_approximate:
            last_local = -1
            first_approx = len(order)
            for index, item in enumerate(order):
                if item in _LOCAL_AUTHORITATIVE_SOURCES:
                    last_local = max(last_local, index)
                if item in _APPROXIMATE_MODEL_SOURCES:
                    first_approx = min(first_approx, index)
            if last_local >= 0 and first_approx < len(order) and last_local > first_approx:
                raise CodeTacticianError(
                    "local authoritative routes must precede approximate/model routes"
                )
        object.__setattr__(self, "source_class_order", order)
        object.__setattr__(
            self,
            "max_sources",
            _positive_int(
                self.max_sources, field_name="max_sources", maximum=HARD_MAX_SOURCES
            ),
        )
        object.__setattr__(
            self,
            "max_routes",
            _positive_int(
                self.max_routes, field_name="max_routes", maximum=HARD_MAX_ROUTES
            ),
        )
        object.__setattr__(
            self,
            "max_subgoals",
            _positive_int(
                self.max_subgoals, field_name="max_subgoals", maximum=HARD_MAX_SUBGOALS
            ),
        )
        object.__setattr__(
            self,
            "max_query_hints_per_source",
            _positive_int(
                self.max_query_hints_per_source,
                field_name="max_query_hints_per_source",
                maximum=HARD_MAX_ROUTES,
            ),
        )
        object.__setattr__(
            self,
            "max_refinement_rounds",
            _positive_int(
                self.max_refinement_rounds,
                field_name="max_refinement_rounds",
                maximum=HARD_MAX_ROUTES,
            ),
        )
        object.__setattr__(
            self,
            "max_premises",
            _positive_int(
                self.max_premises, field_name="max_premises", maximum=HARD_MAX_PREMISES
            ),
        )
        object.__setattr__(
            self,
            "max_queries",
            _positive_int(
                self.max_queries, field_name="max_queries", maximum=HARD_MAX_QUERIES
            ),
        )
        denied = tuple(
            parse_code_source_type(item) for item in self.denied_source_types
        )
        object.__setattr__(self, "denied_source_types", denied)
        for flag_name in (
            "allow_approximate_routes",
            "allow_model_hypothesis",
            "require_local_before_approximate",
            "network_allowed",
            "write_allowed",
            "proof_execution_allowed",
            "semantic_authority",
        ):
            object.__setattr__(
                self,
                flag_name,
                _bool(getattr(self, flag_name), field_name=flag_name),
            )
        if self.network_allowed or self.write_allowed or self.proof_execution_allowed:
            raise CodeTacticianError(
                "Code Tactician policy forbids network, write, and proof execution"
            )
        if self.semantic_authority is not False:
            raise CodeTacticianError(
                "Code Tactician policy cannot claim semantic authority"
            )
        for ref_name in (
            "stop_policy_ref",
            "escalation_policy_ref",
            "abstention_policy_ref",
            "resource_policy_ref",
        ):
            object.__setattr__(
                self,
                ref_name,
                _text(getattr(self, ref_name), field_name=ref_name),
            )

    def rank(self, source_type: CodeSourceType) -> int:
        try:
            return self.source_class_order.index(source_type)
        except ValueError:
            return len(self.source_class_order)

    def admits(self, source_type: CodeSourceType) -> bool:
        if source_type in self.denied_source_types:
            return False
        if (
            source_type is CodeSourceType.MODEL_HYPOTHESIS
            and not self.allow_model_hypothesis
        ):
            return False
        if (
            source_type in _APPROXIMATE_MODEL_SOURCES
            and source_type is not CodeSourceType.MODEL_HYPOTHESIS
            and not self.allow_approximate_routes
        ):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "source_class_order": [item.value for item in self.source_class_order],
            "max_sources": self.max_sources,
            "max_routes": self.max_routes,
            "max_subgoals": self.max_subgoals,
            "max_query_hints_per_source": self.max_query_hints_per_source,
            "max_refinement_rounds": self.max_refinement_rounds,
            "max_premises": self.max_premises,
            "max_queries": self.max_queries,
            "denied_source_types": [item.value for item in self.denied_source_types],
            "allow_approximate_routes": self.allow_approximate_routes,
            "allow_model_hypothesis": self.allow_model_hypothesis,
            "require_local_before_approximate": self.require_local_before_approximate,
            "network_allowed": False,
            "write_allowed": False,
            "proof_execution_allowed": False,
            "semantic_authority": False,
            "stop_policy_ref": self.stop_policy_ref,
            "escalation_policy_ref": self.escalation_policy_ref,
            "abstention_policy_ref": self.abstention_policy_ref,
            "resource_policy_ref": self.resource_policy_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeTacticianPolicy":
        if not isinstance(payload, Mapping):
            raise CodeTacticianError("policy must be an object")
        data = dict(payload)
        data.pop("schema", None)
        return cls(**data)


@dataclass(frozen=True)
class CodeTacticianQuerySpec:
    """Bounded, reference-only provider query (no free-form bodies)."""

    query_id: str
    source_type: CodeSourceType
    adapter_ref: str
    target_ref: str
    root_bindings: Mapping[str, str] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", _text(self.query_id, field_name="query_id")
        )
        object.__setattr__(
            self, "source_type", parse_code_source_type(self.source_type)
        )
        object.__setattr__(
            self, "adapter_ref", _text(self.adapter_ref, field_name="adapter_ref")
        )
        object.__setattr__(
            self, "target_ref", _text(self.target_ref, field_name="target_ref")
        )
        roots = {
            _text(key, field_name="root_bindings.key"): _text(
                value, field_name="root_bindings.value"
            )
            for key, value in dict(self.root_bindings).items()
        }
        object.__setattr__(self, "root_bindings", MappingProxyType(roots))
        params = dict(self.parameters)
        _assert_body_free(params, field_name="query.parameters")
        encoded = canonical_json(params).encode("utf-8")
        if len(encoded) > HARD_MAX_QUERY_BYTES:
            raise CodeTacticianError("query parameters exceed byte bound")
        object.__setattr__(self, "parameters", MappingProxyType(params))
        if self.source_type not in _QUERYABLE_SOURCE_TYPES:
            raise CodeTacticianError(
                f"source type {self.source_type.value!r} is not queryable "
                "through bounded adapters"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "source_type": self.source_type.value,
            "adapter_ref": self.adapter_ref,
            "target_ref": self.target_ref,
            "root_bindings": dict(self.root_bindings),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class CodeTacticianQueryResult:
    """Recorded result of one bounded adapter query."""

    query_id: str
    result_id: str
    source_type: CodeSourceType
    adapter_ref: str
    status: str
    hit_refs: tuple[str, ...] = ()
    exclusion_reason: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", _text(self.query_id, field_name="query_id")
        )
        object.__setattr__(
            self, "result_id", _text(self.result_id, field_name="result_id")
        )
        object.__setattr__(
            self, "source_type", parse_code_source_type(self.source_type)
        )
        object.__setattr__(
            self, "adapter_ref", _text(self.adapter_ref, field_name="adapter_ref")
        )
        object.__setattr__(
            self, "status", _text(self.status, field_name="status")
        )
        hits = tuple(
            sorted(
                {
                    _text(item, field_name="hit_refs")
                    for item in (self.hit_refs or ())
                }
            )
        )
        object.__setattr__(self, "hit_refs", hits)
        object.__setattr__(
            self,
            "exclusion_reason",
            _text(self.exclusion_reason, field_name="exclusion_reason", required=False),
        )
        if self.semantic_authority is not False:
            raise CodeTacticianError(
                "query results cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUERY_RECEIPT_SCHEMA,
            "query_id": self.query_id,
            "result_id": self.result_id,
            "source_type": self.source_type.value,
            "adapter_ref": self.adapter_ref,
            "status": self.status,
            "hit_refs": list(self.hit_refs),
            "exclusion_reason": self.exclusion_reason,
            "semantic_authority": False,
        }


@dataclass(frozen=True)
class CodeTacticianExclusion:
    """Why one source/premise/query was excluded from the plan."""

    exclusion_id: str
    subject_ref: str
    source_type: str
    rationale: str
    stage: str = "admission"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exclusion_id",
            _text(self.exclusion_id, field_name="exclusion_id"),
        )
        object.__setattr__(
            self, "subject_ref", _text(self.subject_ref, field_name="subject_ref")
        )
        object.__setattr__(
            self,
            "source_type",
            _text(self.source_type, field_name="source_type", required=False),
        )
        object.__setattr__(
            self, "rationale", _text(self.rationale, field_name="rationale")
        )
        object.__setattr__(self, "stage", _text(self.stage, field_name="stage"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclusion_id": self.exclusion_id,
            "subject_ref": self.subject_ref,
            "source_type": self.source_type,
            "rationale": self.rationale,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class CodeTacticianRequest:
    """Bounded request turning program goals/corpus into a Tactician plan."""

    roots: ProgramLogicAuthorityRoots
    goals: tuple[ProgramLogicGoal, ...]
    corpus: ProgramLogicPremiseCorpus
    policy: CodeTacticianPolicy = field(default_factory=CodeTacticianPolicy)
    information_demands: tuple[CodeSourceType, ...] = ()
    query_specs: tuple[CodeTacticianQuerySpec, ...] = ()
    expected_roots: ProgramLogicAuthorityRoots | None = None
    admitted_tree_id: str = ""
    admitted_corpus_id: str = ""
    logic_family_refs: tuple[str, ...] = ()
    translation_refs: tuple[str, ...] = ()
    model_id: str = ""
    config_id: str = ""
    request_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = self.roots
        if not isinstance(roots, ProgramLogicAuthorityRoots):
            if isinstance(roots, Mapping):
                roots = (
                    ProgramLogicAuthorityRoots.from_dict(roots)
                    if "schema" in roots
                    else ProgramLogicAuthorityRoots(**roots)
                )
            else:
                raise CodeTacticianError("roots must be ProgramLogicAuthorityRoots")
        object.__setattr__(self, "roots", roots)

        goals = self._decode_goals(self.goals)
        if not goals:
            raise CodeTacticianError("goals must not be empty")
        if len(goals) > HARD_MAX_GOALS:
            raise CodeTacticianError("goals exceed hard maximum")
        for goal in goals:
            if goal.roots.content_id != roots.content_id:
                raise CodeTacticianError(
                    f"goal {goal.goal_id!r} roots do not match request roots"
                )
        object.__setattr__(self, "goals", goals)

        corpus = self.corpus
        if not isinstance(corpus, ProgramLogicPremiseCorpus):
            if isinstance(corpus, Mapping):
                corpus = (
                    ProgramLogicPremiseCorpus.from_dict(corpus)
                    if "schema" in corpus
                    else ProgramLogicPremiseCorpus(**corpus)
                )
            else:
                raise CodeTacticianError("corpus must be ProgramLogicPremiseCorpus")
        if corpus.roots.content_id != roots.content_id:
            raise CodeTacticianError("corpus roots do not match request roots")
        object.__setattr__(self, "corpus", corpus)

        policy = self.policy
        if isinstance(policy, Mapping):
            policy = CodeTacticianPolicy.from_dict(policy)
        if not isinstance(policy, CodeTacticianPolicy):
            raise CodeTacticianError("policy must be CodeTacticianPolicy")
        object.__setattr__(self, "policy", policy)

        demands = tuple(
            parse_code_source_type(item) for item in (self.information_demands or ())
        )
        object.__setattr__(self, "information_demands", demands)

        queries = tuple(self._decode_queries(self.query_specs))
        if len(queries) > policy.max_queries:
            raise CodeTacticianError("query_specs exceed policy max_queries")
        object.__setattr__(self, "query_specs", queries)

        expected = self.expected_roots
        if expected is not None and not isinstance(expected, ProgramLogicAuthorityRoots):
            if isinstance(expected, Mapping):
                expected = (
                    ProgramLogicAuthorityRoots.from_dict(expected)
                    if "schema" in expected
                    else ProgramLogicAuthorityRoots(**expected)
                )
            else:
                raise CodeTacticianError(
                    "expected_roots must be ProgramLogicAuthorityRoots"
                )
        object.__setattr__(self, "expected_roots", expected)

        object.__setattr__(
            self,
            "admitted_tree_id",
            _text(self.admitted_tree_id, field_name="admitted_tree_id", required=False),
        )
        object.__setattr__(
            self,
            "admitted_corpus_id",
            _text(
                self.admitted_corpus_id, field_name="admitted_corpus_id", required=False
            ),
        )
        object.__setattr__(
            self,
            "logic_family_refs",
            tuple(
                sorted(
                    {
                        _text(item, field_name="logic_family_refs")
                        for item in (self.logic_family_refs or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "translation_refs",
            tuple(
                sorted(
                    {
                        _text(item, field_name="translation_refs")
                        for item in (self.translation_refs or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, field_name="model_id", required=False),
        )
        config = _text(self.config_id, field_name="config_id", required=False)
        if not config:
            config = policy.policy_id
        object.__setattr__(self, "config_id", config)
        request_id = _text(self.request_id, field_name="request_id", required=False)
        if not request_id:
            request_id = _stable_id(
                "code-tactician-request",
                {
                    "roots": roots.content_id,
                    "goal_ids": [goal.goal_id for goal in goals],
                    "corpus": corpus.content_id,
                    "policy": policy.policy_id,
                    "demands": [item.value for item in demands],
                },
            )
        object.__setattr__(self, "request_id", request_id)
        metadata = dict(self.metadata or {})
        _assert_body_free(metadata, field_name="metadata")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @staticmethod
    def _decode_goals(value: Any) -> tuple[ProgramLogicGoal, ...]:
        if value is None:
            return ()
        if isinstance(value, ProgramLogicGoal):
            return (value,)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise CodeTacticianError("goals must be a sequence of ProgramLogicGoal")
        goals: list[ProgramLogicGoal] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, ProgramLogicGoal):
                goal = item
            elif isinstance(item, Mapping):
                goal = (
                    ProgramLogicGoal.from_dict(item)
                    if "schema" in item
                    else ProgramLogicGoal(**item)
                )
            else:
                raise CodeTacticianError("goals must contain ProgramLogicGoal values")
            if goal.goal_id in seen:
                raise CodeTacticianError(f"duplicate goal_id {goal.goal_id!r}")
            seen.add(goal.goal_id)
            goals.append(goal)
        return tuple(goals)

    @staticmethod
    def _decode_queries(value: Any) -> list[CodeTacticianQuerySpec]:
        if value is None:
            return []
        if isinstance(value, CodeTacticianQuerySpec):
            return [value]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise CodeTacticianError("query_specs must be a sequence")
        out: list[CodeTacticianQuerySpec] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, CodeTacticianQuerySpec):
                query = item
            elif isinstance(item, Mapping):
                query = CodeTacticianQuerySpec(**dict(item))
            else:
                raise CodeTacticianError("query_specs must contain CodeTacticianQuerySpec")
            if query.query_id in seen:
                raise CodeTacticianError(f"duplicate query_id {query.query_id!r}")
            seen.add(query.query_id)
            out.append(query)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "roots": self.roots.to_dict(),
            "goals": [goal.to_dict() for goal in self.goals],
            "corpus": self.corpus.to_dict(),
            "policy": self.policy.to_dict(),
            "information_demands": [item.value for item in self.information_demands],
            "query_specs": [item.to_dict() for item in self.query_specs],
            "expected_roots": (
                self.expected_roots.to_dict() if self.expected_roots else None
            ),
            "admitted_tree_id": self.admitted_tree_id,
            "admitted_corpus_id": self.admitted_corpus_id,
            "logic_family_refs": list(self.logic_family_refs),
            "translation_refs": list(self.translation_refs),
            "model_id": self.model_id,
            "config_id": self.config_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CodeTacticianResponse:
    """Deterministic, non-authoritative adapter response."""

    status: CodeTacticianStatus
    reason_code: CodeTacticianReasonCode
    request_id: str
    plan: TacticianSearchPlan | None = None
    query_results: tuple[CodeTacticianQueryResult, ...] = ()
    exclusions: tuple[CodeTacticianExclusion, ...] = ()
    selected_source_types: tuple[CodeSourceType, ...] = ()
    excluded_source_types: tuple[CodeSourceType, ...] = ()
    generic_plan_ref: str = ""
    generic_receipt_ref: str = ""
    planner_id: str = CODE_TACTICIAN_PLANNER_ID
    provider_id: str = IPFS_DATASETS_TACTICIAN_PROVIDER_ID
    provider_version: str = IPFS_DATASETS_TACTICIAN_PROVIDER_VERSION
    interface_version: str = GENERIC_TACTICIAN_INTERFACE
    semantic_authority: bool = False
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum_status(self.status)
        )
        object.__setattr__(
            self, "reason_code", _enum_reason(self.reason_code)
        )
        object.__setattr__(
            self, "request_id", _text(self.request_id, field_name="request_id")
        )
        if self.plan is not None and not isinstance(self.plan, TacticianSearchPlan):
            raise CodeTacticianError("plan must be TacticianSearchPlan or None")
        if self.plan is not None and self.plan.semantic_authority is not False:
            raise CodeTacticianError("plan cannot claim semantic authority")
        if self.status is CodeTacticianStatus.PLANNED and self.plan is None:
            raise CodeTacticianError("planned response requires a search plan")
        object.__setattr__(
            self,
            "query_results",
            tuple(self.query_results or ()),
        )
        object.__setattr__(
            self,
            "exclusions",
            tuple(self.exclusions or ()),
        )
        object.__setattr__(
            self,
            "selected_source_types",
            tuple(parse_code_source_type(item) for item in self.selected_source_types),
        )
        object.__setattr__(
            self,
            "excluded_source_types",
            tuple(parse_code_source_type(item) for item in self.excluded_source_types),
        )
        if self.semantic_authority is not False:
            raise CodeTacticianError(
                "Code Tactician responses cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "message",
            _text(self.message, field_name="message", required=False),
        )
        details = dict(self.details or {})
        _assert_body_free(details, field_name="details")
        object.__setattr__(self, "details", MappingProxyType(details))
        for name in (
            "generic_plan_ref",
            "generic_receipt_ref",
            "planner_id",
            "provider_id",
            "provider_version",
            "interface_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False)
                if name.endswith("_ref")
                else _text(getattr(self, name), field_name=name),
            )

    @property
    def planned(self) -> bool:
        return self.status is CodeTacticianStatus.PLANNED and self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_RESPONSE_SCHEMA,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "request_id": self.request_id,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "query_results": [item.to_dict() for item in self.query_results],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "selected_source_types": [
                item.value for item in self.selected_source_types
            ],
            "excluded_source_types": [
                item.value for item in self.excluded_source_types
            ],
            "generic_plan_ref": self.generic_plan_ref,
            "generic_receipt_ref": self.generic_receipt_ref,
            "planner_id": self.planner_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "interface_version": self.interface_version,
            "semantic_authority": False,
            "message": self.message,
            "details": dict(self.details),
        }


def _enum_status(value: Any) -> CodeTacticianStatus:
    if isinstance(value, CodeTacticianStatus):
        return value
    return CodeTacticianStatus(str(value))


def _enum_reason(value: Any) -> CodeTacticianReasonCode:
    if isinstance(value, CodeTacticianReasonCode):
        return value
    return CodeTacticianReasonCode(str(value))


@dataclass(frozen=True)
class CodeTacticianCapability:
    """Lazy capability declaration; package presence is not authority."""

    provider_id: str = IPFS_DATASETS_TACTICIAN_PROVIDER_ID
    provider_version: str = IPFS_DATASETS_TACTICIAN_PROVIDER_VERSION
    interface_version: str = GENERIC_TACTICIAN_INTERFACE
    imported: bool = False
    available: bool = False
    health: str = "lazy"
    supported_source_types: tuple[str, ...] = field(
        default_factory=lambda: tuple(item.value for item in CODE_SOURCE_PRECEDENCE)
    )
    semantic_authority: bool = False
    completion_authority: bool = False
    reason_code: str = "lazy"

    def __post_init__(self) -> None:
        if self.semantic_authority or self.completion_authority:
            raise CodeTacticianError(
                "capability declaration cannot claim semantic or completion authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "completion_authority", False)
        object.__setattr__(
            self,
            "supported_source_types",
            tuple(self.supported_source_types),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CAPABILITY_SCHEMA,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "interface_version": self.interface_version,
            "imported": self.imported,
            "available": self.available,
            "health": self.health,
            "supported_source_types": list(self.supported_source_types),
            "semantic_authority": False,
            "completion_authority": False,
            "reason_code": self.reason_code,
        }


# ---------------------------------------------------------------------------
# Source candidate projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceCandidate:
    source_id: str
    source_type: CodeSourceType
    precedence: int
    rationale: str
    premise_ids: tuple[str, ...] = ()
    query_hints: tuple[str, ...] = ()
    source_root: str = ""


def _default_rationale(source_type: CodeSourceType) -> str:
    if source_type in _LOCAL_AUTHORITATIVE_SOURCES:
        return (
            f"Local authoritative route for {source_type.value}; "
            "exact facts precede approximate/model nomination"
        )
    if source_type is CodeSourceType.MODEL_HYPOTHESIS:
        return (
            "Approval-gated model hypothesis route; nominating only, "
            "never semantic authority"
        )
    return (
        f"Approximate/nominating route for {source_type.value}; "
        "cannot establish information sufficiency alone"
    )


def project_source_candidates(
    request: CodeTacticianRequest,
) -> tuple[list[_SourceCandidate], list[CodeTacticianExclusion]]:
    """Project corpus premises + demands into ordered source candidates."""

    policy = request.policy
    exclusions: list[CodeTacticianExclusion] = []
    by_type: dict[CodeSourceType, list[str]] = {
        item: [] for item in CODE_SOURCE_PRECEDENCE
    }

    if len(request.corpus.premises) > policy.max_premises:
        raise CodeTacticianError(
            f"corpus premises {len(request.corpus.premises)} exceed "
            f"policy max_premises {policy.max_premises}"
        )

    for premise in request.corpus.premises:
        source_type = map_premise_source(premise.source_class)
        if premise.semantic_authority is True:
            exclusions.append(
                CodeTacticianExclusion(
                    exclusion_id=_stable_id(
                        "exclusion",
                        {"premise": premise.premise_id, "why": "semantic_authority"},
                    ),
                    subject_ref=premise.premise_id,
                    source_type=source_type.value,
                    rationale="Premise claimed semantic_authority=true",
                    stage="authority_gate",
                )
            )
            continue
        if not policy.admits(source_type):
            exclusions.append(
                CodeTacticianExclusion(
                    exclusion_id=_stable_id(
                        "exclusion",
                        {"premise": premise.premise_id, "why": "policy_denied"},
                    ),
                    subject_ref=premise.premise_id,
                    source_type=source_type.value,
                    rationale=f"Source type {source_type.value} denied by policy",
                    stage="policy",
                )
            )
            continue
        by_type[source_type].append(premise.premise_id)

    # Information demands activate empty route classes as reference-only sources.
    demanded = request.information_demands or tuple(
        item
        for item in policy.source_class_order
        if item in _LOCAL_AUTHORITATIVE_SOURCES
    )
    for demand in demanded:
        if demand not in by_type:
            by_type[demand] = []
        if not policy.admits(demand):
            exclusions.append(
                CodeTacticianExclusion(
                    exclusion_id=_stable_id(
                        "exclusion",
                        {"demand": demand.value, "why": "policy_denied"},
                    ),
                    subject_ref=f"demand:{demand.value}",
                    source_type=demand.value,
                    rationale=f"Information demand {demand.value} denied by policy",
                    stage="policy",
                )
            )

    candidates: list[_SourceCandidate] = []
    for source_type in policy.source_class_order:
        if not policy.admits(source_type):
            if source_type in demanded or by_type.get(source_type):
                continue
            continue
        premises = tuple(sorted(by_type.get(source_type, ())))
        # Skip approximate routes with no premises and no explicit demand.
        if (
            source_type in _APPROXIMATE_MODEL_SOURCES
            and not premises
            and source_type not in demanded
        ):
            continue
        # Skip undemanded local routes with no premises only when demands
        # were explicitly supplied (otherwise keep the full local ladder).
        if (
            request.information_demands
            and source_type not in demanded
            and not premises
        ):
            continue
        hints = tuple(
            sorted(premises)[: policy.max_query_hints_per_source]
        )
        candidates.append(
            _SourceCandidate(
                source_id=f"source:{source_type.value}",
                source_type=source_type,
                precedence=policy.rank(source_type),
                rationale=_default_rationale(source_type),
                premise_ids=premises,
                query_hints=hints,
                source_root=request.roots.corpus_id,
            )
        )

    candidates.sort(key=lambda item: (item.precedence, item.source_id))
    return candidates, exclusions


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


PlannerFactory = Callable[[], Any]
Importer = Callable[[str], Any]
QueryAdapter = Callable[[CodeTacticianQuerySpec], CodeTacticianQueryResult]


class IpfsDatasetsTacticianProvider:
    """Lazy, fail-closed program-repair adapter for the generic Logic Tactician.

    Parameters
    ----------
    policy:
        Default :class:`CodeTacticianPolicy` used when a request omits one.
    importer:
        Optional import callable (defaults to :func:`importlib.import_module`).
    planner_factory:
        Optional factory returning an object with ``plan(goal, sources, policy)``
        matching :class:`LogicTactician`. Used by tests and embedded deployments.
    query_adapters:
        Mapping of ``adapter_ref`` -> callable for bounded provider queries.
    module_name:
        Optional datasets module path override (defaults to the generic package).
    """

    def __init__(
        self,
        policy: CodeTacticianPolicy | None = None,
        *,
        importer: Importer | None = None,
        planner_factory: PlannerFactory | None = None,
        query_adapters: Mapping[str, QueryAdapter] | None = None,
        module_name: str = GENERIC_TACTICIAN_MODULE,
    ) -> None:
        self._default_policy = policy or CodeTacticianPolicy()
        self._importer = importer or importlib.import_module
        self._planner_factory = planner_factory
        self._query_adapters = dict(query_adapters or {})
        self._module_name = module_name
        self._module: Any | None = None
        self._import_error: BaseException | None = None
        self._import_attempted = False
        self._lock = threading.Lock()

    # -- capability / lazy loading -----------------------------------------

    def capabilities(self) -> CodeTacticianCapability:
        """Declare capability without importing the optional package."""

        return CodeTacticianCapability(
            imported=False,
            available=False,
            health="lazy",
            reason_code="lazy",
        )

    capability = capabilities

    def _load_module(self) -> Any | None:
        if self._planner_factory is not None:
            return None
        with self._lock:
            if self._import_attempted:
                return self._module
            self._import_attempted = True
            try:
                self._module = self._importer(self._module_name)
            except Exception as exc:  # noqa: BLE001 - typed abstention path
                self._import_error = exc
                self._module = None
            return self._module

    def _resolve_planner(self) -> tuple[Any | None, CodeTacticianReasonCode | None, str]:
        if self._planner_factory is not None:
            try:
                planner = self._planner_factory()
            except Exception as exc:  # noqa: BLE001
                return None, CodeTacticianReasonCode.PLANNER_ERROR, str(exc)
            return planner, None, ""

        module = self._load_module()
        if module is None:
            detail = (
                f"{type(self._import_error).__name__}: {self._import_error}"
                if self._import_error is not None
                else f"module {self._module_name!r} unavailable"
            )
            return (
                None,
                CodeTacticianReasonCode.OPTIONAL_MODULE_UNAVAILABLE,
                detail,
            )

        logic_cls = getattr(module, "LogicTactician", None)
        if logic_cls is None:
            # Lazy export packages may only expose names via __getattr__.
            try:
                logic_cls = module.LogicTactician  # type: ignore[attr-defined]
            except Exception:
                logic_cls = None
        if logic_cls is None:
            return (
                None,
                CodeTacticianReasonCode.INTERFACE_INCOMPATIBLE,
                "LogicTactician symbol missing",
            )
        try:
            planner = logic_cls()
        except Exception as exc:  # noqa: BLE001
            return None, CodeTacticianReasonCode.PLANNER_ERROR, str(exc)
        return planner, None, ""

    # -- public planning API -----------------------------------------------

    def build_request(self, payload: Mapping[str, Any] | CodeTacticianRequest) -> CodeTacticianRequest:
        """Normalize a mapping or request object into :class:`CodeTacticianRequest`."""

        if isinstance(payload, CodeTacticianRequest):
            return payload
        if not isinstance(payload, Mapping):
            raise CodeTacticianError("request payload must be an object")
        data = dict(payload)
        data.pop("schema", None)
        if "policy" not in data:
            data["policy"] = self._default_policy
        return CodeTacticianRequest(**data)

    def plan(
        self,
        request: Mapping[str, Any] | CodeTacticianRequest,
    ) -> CodeTacticianResponse:
        """Plan program-logic search routes under explicit source precedence."""

        try:
            normalized = self.build_request(request)
        except CodeTacticianError as exc:
            return self._reject(
                request_id=_stable_id("code-tactician-request", {"error": str(exc)}),
                status=CodeTacticianStatus.MALFORMED,
                reason=CodeTacticianReasonCode.MALFORMED_REQUEST,
                message=str(exc),
            )
        except (ProgramLogicPredictionError, TypeError, ValueError) as exc:
            return self._reject(
                request_id=_stable_id("code-tactician-request", {"error": str(exc)}),
                status=CodeTacticianStatus.MALFORMED,
                reason=CodeTacticianReasonCode.MALFORMED_REQUEST,
                message=str(exc),
            )

        admission = self._admit_request(normalized)
        if admission is not None:
            return admission

        try:
            candidates, exclusions = project_source_candidates(normalized)
        except CodeTacticianError as exc:
            reason = (
                CodeTacticianReasonCode.UNBOUNDED_PLAN
                if "max_premises" in str(exc) or "exceed" in str(exc)
                else CodeTacticianReasonCode.MALFORMED_REQUEST
            )
            return self._reject(
                request_id=normalized.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=reason,
                message=str(exc),
            )

        if not candidates:
            return CodeTacticianResponse(
                status=CodeTacticianStatus.ABSTAINED,
                reason_code=CodeTacticianReasonCode.NO_ADMISSIBLE_SOURCES,
                request_id=normalized.request_id,
                exclusions=tuple(exclusions),
                message="no admissible program source routes after policy gates",
                semantic_authority=False,
            )

        # Bounded referenced queries execute only through registered adapters.
        query_results, query_exclusions = self._execute_queries(normalized)
        exclusions.extend(query_exclusions)

        planner, reason, detail = self._resolve_planner()
        if planner is None:
            status = (
                CodeTacticianStatus.UNAVAILABLE
                if reason is CodeTacticianReasonCode.OPTIONAL_MODULE_UNAVAILABLE
                else CodeTacticianStatus.ABSTAINED
            )
            return CodeTacticianResponse(
                status=status,
                reason_code=reason or CodeTacticianReasonCode.OPTIONAL_MODULE_UNAVAILABLE,
                request_id=normalized.request_id,
                query_results=tuple(query_results),
                exclusions=tuple(exclusions),
                excluded_source_types=tuple(
                    item.source_type for item in candidates
                ),
                message=detail or "generic Logic Tactician unavailable",
                semantic_authority=False,
                details={"module": self._module_name},
            )

        try:
            plan, generic_plan_ref, generic_receipt_ref, plan_exclusions = (
                self._plan_with_generic(
                    normalized,
                    candidates,
                    planner,
                    query_results=query_results,
                )
            )
        except CodeTacticianError as exc:
            return CodeTacticianResponse(
                status=CodeTacticianStatus.ABSTAINED,
                reason_code=CodeTacticianReasonCode.PLANNER_ABSTAINED,
                request_id=normalized.request_id,
                query_results=tuple(query_results),
                exclusions=tuple(exclusions),
                message=str(exc),
                semantic_authority=False,
            )
        except Exception as exc:  # noqa: BLE001
            return CodeTacticianResponse(
                status=CodeTacticianStatus.ABSTAINED,
                reason_code=CodeTacticianReasonCode.PLANNER_ERROR,
                request_id=normalized.request_id,
                query_results=tuple(query_results),
                exclusions=tuple(exclusions),
                message=f"{type(exc).__name__}: {exc}",
                semantic_authority=False,
            )

        exclusions.extend(plan_exclusions)
        # Derive selected/excluded source types from candidates and plan routes.
        selected_types = self._selected_types_from_plan(plan, candidates)
        excluded_types = tuple(
            item.source_type
            for item in candidates
            if item.source_type not in selected_types
        )

        return CodeTacticianResponse(
            status=CodeTacticianStatus.PLANNED,
            reason_code=CodeTacticianReasonCode.OK,
            request_id=normalized.request_id,
            plan=plan,
            query_results=tuple(query_results),
            exclusions=tuple(exclusions),
            selected_source_types=selected_types,
            excluded_source_types=excluded_types,
            generic_plan_ref=generic_plan_ref,
            generic_receipt_ref=generic_receipt_ref,
            planner_id=plan.planner_id or CODE_TACTICIAN_PLANNER_ID,
            semantic_authority=False,
            message="planned",
        )

    def _selected_types_from_plan(
        self,
        plan: TacticianSearchPlan,
        candidates: Sequence[_SourceCandidate],
    ) -> tuple[CodeSourceType, ...]:
        route_to_types: dict[SourceRouteKind, list[CodeSourceType]] = {}
        for candidate in candidates:
            route = map_code_source_to_route(candidate.source_type)
            route_to_types.setdefault(route, []).append(candidate.source_type)
        selected: list[CodeSourceType] = []
        seen: set[CodeSourceType] = set()
        for route in plan.ordered_source_routes:
            for source_type in route_to_types.get(route, ()):
                if source_type not in seen:
                    seen.add(source_type)
                    selected.append(source_type)
        # Preserve policy order among selected types.
        order = {item: index for index, item in enumerate(CODE_SOURCE_PRECEDENCE)}
        selected.sort(key=lambda item: order.get(item, 999))
        return tuple(selected)

    # -- admission gates ---------------------------------------------------

    def _admit_request(
        self, request: CodeTacticianRequest
    ) -> CodeTacticianResponse | None:
        if not request.goals:
            return self._reject(
                request_id=request.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=CodeTacticianReasonCode.EMPTY_GOALS,
                message="no goals supplied",
            )

        if request.expected_roots is not None:
            if request.expected_roots.content_id != request.roots.content_id:
                return self._reject(
                    request_id=request.request_id,
                    status=CodeTacticianStatus.REJECTED,
                    reason=CodeTacticianReasonCode.CROSS_ROOT,
                    message="request roots diverge from expected_roots",
                    details={
                        "request_roots": request.roots.content_id,
                        "expected_roots": request.expected_roots.content_id,
                    },
                )

        if request.admitted_tree_id and request.admitted_tree_id != request.roots.tree_id:
            return self._reject(
                request_id=request.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=CodeTacticianReasonCode.STALE_ROOTS,
                message="admitted_tree_id does not match request.roots.tree_id",
                details={
                    "admitted_tree_id": request.admitted_tree_id,
                    "tree_id": request.roots.tree_id,
                },
            )

        if (
            request.admitted_corpus_id
            and request.admitted_corpus_id != request.roots.corpus_id
        ):
            return self._reject(
                request_id=request.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=CodeTacticianReasonCode.STALE_ROOTS,
                message="admitted_corpus_id does not match request.roots.corpus_id",
                details={
                    "admitted_corpus_id": request.admitted_corpus_id,
                    "corpus_id": request.roots.corpus_id,
                },
            )

        # Cross-root among goals already enforced in request construction.
        for goal in request.goals:
            if goal.roots.content_id != request.corpus.roots.content_id:
                return self._reject(
                    request_id=request.request_id,
                    status=CodeTacticianStatus.REJECTED,
                    reason=CodeTacticianReasonCode.CROSS_ROOT,
                    message=f"goal {goal.goal_id} roots diverge from corpus",
                )

        policy = request.policy
        if (
            policy.max_sources > HARD_MAX_SOURCES
            or policy.max_routes > HARD_MAX_ROUTES
            or policy.max_subgoals > HARD_MAX_SUBGOALS
        ):
            return self._reject(
                request_id=request.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=CodeTacticianReasonCode.UNBOUNDED_PLAN,
                message="policy budgets exceed hard maxima",
            )

        for demand in request.information_demands:
            try:
                parse_code_source_type(demand)
            except CodeTacticianError as exc:
                return self._reject(
                    request_id=request.request_id,
                    status=CodeTacticianStatus.REJECTED,
                    reason=CodeTacticianReasonCode.UNSUPPORTED_SOURCE_TYPE,
                    message=str(exc),
                )

        for query in request.query_specs:
            for key, value in query.root_bindings.items():
                expected = getattr(request.roots, key, None)
                if expected is not None and expected != value:
                    return self._reject(
                        request_id=request.request_id,
                        status=CodeTacticianStatus.REJECTED,
                        reason=CodeTacticianReasonCode.CROSS_ROOT,
                        message=(
                            f"query {query.query_id} root binding {key} "
                            "does not match request roots"
                        ),
                    )
            if query.source_type not in _QUERYABLE_SOURCE_TYPES:
                return self._reject(
                    request_id=request.request_id,
                    status=CodeTacticianStatus.REJECTED,
                    reason=CodeTacticianReasonCode.UNSUPPORTED_SOURCE_TYPE,
                    message=(
                        f"query source type {query.source_type.value} is not "
                        "supported for adapter execution"
                    ),
                )

        try:
            _assert_body_free(request.metadata, field_name="metadata")
        except CodeTacticianError as exc:
            return self._reject(
                request_id=request.request_id,
                status=CodeTacticianStatus.REJECTED,
                reason=CodeTacticianReasonCode.FREE_FORM_AUTHORITY,
                message=str(exc),
            )

        return None

    def _reject(
        self,
        *,
        request_id: str,
        status: CodeTacticianStatus,
        reason: CodeTacticianReasonCode,
        message: str,
        details: Mapping[str, Any] | None = None,
        exclusions: Sequence[CodeTacticianExclusion] = (),
    ) -> CodeTacticianResponse:
        return CodeTacticianResponse(
            status=status,
            reason_code=reason,
            request_id=request_id,
            exclusions=tuple(exclusions),
            message=message,
            details=dict(details or {}),
            semantic_authority=False,
        )

    # -- query execution ---------------------------------------------------

    def _execute_queries(
        self, request: CodeTacticianRequest
    ) -> tuple[list[CodeTacticianQueryResult], list[CodeTacticianExclusion]]:
        results: list[CodeTacticianQueryResult] = []
        exclusions: list[CodeTacticianExclusion] = []
        if len(request.query_specs) > request.policy.max_queries:
            exclusions.append(
                CodeTacticianExclusion(
                    exclusion_id=_stable_id(
                        "exclusion", {"why": "max_queries", "n": len(request.query_specs)}
                    ),
                    subject_ref="query_specs",
                    source_type="",
                    rationale="query_specs exceed policy max_queries",
                    stage="query_budget",
                )
            )
            return results, exclusions

        for query in request.query_specs:
            if not request.policy.admits(query.source_type):
                result = CodeTacticianQueryResult(
                    query_id=query.query_id,
                    result_id=_stable_id(
                        "query-result",
                        {"query_id": query.query_id, "status": "excluded"},
                    ),
                    source_type=query.source_type,
                    adapter_ref=query.adapter_ref,
                    status="excluded",
                    exclusion_reason="source type denied by policy",
                )
                results.append(result)
                exclusions.append(
                    CodeTacticianExclusion(
                        exclusion_id=_stable_id(
                            "exclusion",
                            {"query": query.query_id, "why": "policy_denied"},
                        ),
                        subject_ref=query.query_id,
                        source_type=query.source_type.value,
                        rationale="Query source type denied by policy",
                        stage="query_policy",
                    )
                )
                continue

            adapter = self._query_adapters.get(query.adapter_ref)
            if adapter is None:
                result = CodeTacticianQueryResult(
                    query_id=query.query_id,
                    result_id=_stable_id(
                        "query-result",
                        {
                            "query_id": query.query_id,
                            "status": "adapter_missing",
                        },
                    ),
                    source_type=query.source_type,
                    adapter_ref=query.adapter_ref,
                    status="adapter_missing",
                    exclusion_reason="no registered bounded adapter for adapter_ref",
                )
                results.append(result)
                exclusions.append(
                    CodeTacticianExclusion(
                        exclusion_id=_stable_id(
                            "exclusion",
                            {"query": query.query_id, "why": "adapter_missing"},
                        ),
                        subject_ref=query.query_id,
                        source_type=query.source_type.value,
                        rationale=(
                            f"No bounded adapter registered for {query.adapter_ref!r}"
                        ),
                        stage="query_adapter",
                    )
                )
                continue

            try:
                outcome = adapter(query)
            except Exception as exc:  # noqa: BLE001
                result = CodeTacticianQueryResult(
                    query_id=query.query_id,
                    result_id=_stable_id(
                        "query-result",
                        {"query_id": query.query_id, "status": "error"},
                    ),
                    source_type=query.source_type,
                    adapter_ref=query.adapter_ref,
                    status="error",
                    exclusion_reason=f"{type(exc).__name__}: {exc}",
                )
                results.append(result)
                exclusions.append(
                    CodeTacticianExclusion(
                        exclusion_id=_stable_id(
                            "exclusion",
                            {"query": query.query_id, "why": "adapter_error"},
                        ),
                        subject_ref=query.query_id,
                        source_type=query.source_type.value,
                        rationale=f"Adapter error: {type(exc).__name__}",
                        stage="query_adapter",
                    )
                )
                continue

            if not isinstance(outcome, CodeTacticianQueryResult):
                raise CodeTacticianError(
                    "query adapter must return CodeTacticianQueryResult"
                )
            if outcome.query_id != query.query_id:
                raise CodeTacticianError(
                    "query adapter result query_id must match the request"
                )
            if outcome.semantic_authority is not False:
                raise CodeTacticianError(
                    "query adapter result cannot claim semantic authority"
                )
            results.append(outcome)
            if outcome.status not in {"completed", "ok", "hits"}:
                exclusions.append(
                    CodeTacticianExclusion(
                        exclusion_id=_stable_id(
                            "exclusion",
                            {
                                "query": query.query_id,
                                "status": outcome.status,
                            },
                        ),
                        subject_ref=query.query_id,
                        source_type=query.source_type.value,
                        rationale=outcome.exclusion_reason
                        or f"Query status {outcome.status}",
                        stage="query_result",
                    )
                )
        return results, exclusions

    # -- generic planner projection ----------------------------------------

    def _plan_with_generic(
        self,
        request: CodeTacticianRequest,
        candidates: Sequence[_SourceCandidate],
        planner: Any,
        *,
        query_results: Sequence[CodeTacticianQueryResult],
    ) -> tuple[
        TacticianSearchPlan,
        str,
        str,
        list[CodeTacticianExclusion],
    ]:
        module = self._module
        # Prefer symbols from an injected planner's defining module when no
        # optional package was loaded (test fakes).
        symbols = self._resolve_generic_symbols(module, planner)

        policy_obj = self._build_generic_policy(request.policy, symbols)
        exclusions: list[CodeTacticianExclusion] = []

        # Multi-goal plans: plan the primary goal and attach residual goal ids.
        primary = request.goals[0]
        generic_goal = self._build_generic_goal(primary, request, symbols)

        # Generic LogicTactician hard-rejects candidate floods above
        # max_sources * 4. Truncate in local-first order and record exclusions
        # rather than aborting an otherwise valid program-repair plan.
        hard_bound = max(1, request.policy.max_sources * 4)
        admitted_candidates = list(candidates[:hard_bound])
        for candidate in candidates[hard_bound:]:
            exclusions.append(
                CodeTacticianExclusion(
                    exclusion_id=_stable_id(
                        "exclusion",
                        {
                            "source": candidate.source_id,
                            "why": "hard_admission_bound",
                        },
                    ),
                    subject_ref=candidate.source_id,
                    source_type=candidate.source_type.value,
                    rationale=(
                        f"Excluded after hard admission bound "
                        f"(max_sources*4={hard_bound})"
                    ),
                    stage="admission_budget",
                )
            )

        generic_sources = [
            self._build_generic_source(candidate, symbols)
            for candidate in admitted_candidates
        ]

        plan_fn = getattr(planner, "plan", None)
        if not callable(plan_fn):
            raise CodeTacticianError("planner must expose a plan() method")

        generic_plan = plan_fn(generic_goal, generic_sources, policy_obj)
        generic_plan_ref = str(
            getattr(generic_plan, "plan_id", "")
            or _stable_id("generic-plan", {"goal": primary.goal_id})
        )

        # Optional receipt when TacticianReceipt is available.
        generic_receipt_ref = ""
        receipt_cls = symbols.get("TacticianReceipt")
        if receipt_cls is not None and hasattr(receipt_cls, "from_plan"):
            try:
                receipt = receipt_cls.from_plan(generic_plan, policy_obj)
                generic_receipt_ref = str(getattr(receipt, "receipt_id", ""))
            except Exception:
                generic_receipt_ref = ""

        stop_disposition = str(
            getattr(
                getattr(generic_plan, "stop_disposition", None),
                "value",
                getattr(generic_plan, "stop_disposition", "continue"),
            )
        )
        if stop_disposition in {"abstain", "no_admissible_sources", "cycle_detected"}:
            raise CodeTacticianError(
                f"generic planner stop_disposition={stop_disposition}"
            )

        search_plan, map_exclusions = self._project_search_plan(
            request=request,
            candidates=admitted_candidates,
            generic_plan=generic_plan,
            query_results=query_results,
        )
        exclusions.extend(map_exclusions)
        return search_plan, generic_plan_ref, generic_receipt_ref, exclusions

    def _resolve_generic_symbols(
        self, module: Any | None, planner: Any
    ) -> dict[str, Any]:
        symbols: dict[str, Any] = {}
        names = (
            "TacticianGoal",
            "TacticianSource",
            "TacticianPolicy",
            "TacticianReceipt",
            "default_policy",
            "LogicTactician",
        )
        # From loaded module first.
        if module is not None:
            for name in names:
                try:
                    symbols[name] = getattr(module, name)
                except Exception:
                    pass
        # Fall back to planner module attributes (fakes / partial surfaces).
        planner_module = getattr(planner, "__module__", None)
        if planner_module:
            try:
                if importlib.util.find_spec(planner_module) is not None:
                    loaded = importlib.import_module(planner_module)
                    for name in names:
                        if name not in symbols and hasattr(loaded, name):
                            symbols[name] = getattr(loaded, name)
            except Exception:
                pass
        # Allow fakes to hang constructors on the planner instance/class.
        for name in names:
            if name not in symbols and hasattr(planner, name):
                symbols[name] = getattr(planner, name)
            if name not in symbols and hasattr(type(planner), name):
                symbols[name] = getattr(type(planner), name)
        return symbols

    def _build_generic_policy(
        self, policy: CodeTacticianPolicy, symbols: Mapping[str, Any]
    ) -> Any:
        policy_cls = symbols.get("TacticianPolicy")
        payload = {
            "policy_id": policy.policy_id,
            "source_class_order": [item.value for item in policy.source_class_order],
            "max_sources": policy.max_sources,
            "max_routes": policy.max_routes,
            "max_subgoals": policy.max_subgoals,
            "max_query_hints_per_source": policy.max_query_hints_per_source,
            "max_refinement_rounds": policy.max_refinement_rounds,
            "allow_learned_ranking": False,
            "allow_llm_nomination": False,
            "denied_source_classes": [
                item.value for item in policy.denied_source_types
            ],
            "network_allowed": False,
            "write_allowed": False,
            "proof_execution_allowed": False,
            "semantic_authority": False,
        }
        if policy_cls is None:
            # Duck-typed fake policy.
            return MappingProxyType(payload)
        if hasattr(policy_cls, "from_dict"):
            return policy_cls.from_dict(payload)
        return policy_cls(**payload)

    def _build_generic_goal(
        self,
        goal: ProgramLogicGoal,
        request: CodeTacticianRequest,
        symbols: Mapping[str, Any],
    ) -> Any:
        goal_cls = symbols.get("TacticianGoal")
        proof_gaps = [
            ref
            for ref in (
                list(goal.bound_refs)
                + list(goal.invalidation_refs)
                + list(goal.assumption_refs)
            )
            if ref
        ]
        # Prefer explicit information-demand gaps when present.
        if request.information_demands:
            proof_gaps = [
                f"gap:{item.value}" for item in request.information_demands
            ] + proof_gaps
        # De-dupe while preserving order.
        seen: set[str] = set()
        ordered_gaps: list[str] = []
        for gap in proof_gaps:
            if gap not in seen:
                seen.add(gap)
                ordered_gaps.append(gap)
        ordered_gaps = ordered_gaps[: request.policy.max_subgoals]

        payload = {
            "goal_id": goal.goal_id,
            "statement_ref": goal.positive_statement_ref,
            "goal_family": goal.family.value
            if hasattr(goal.family, "value")
            else str(goal.family),
            "goal_root": goal.content_id,
            "corpus_root": request.corpus.content_id,
            "config_root": request.policy.policy_id,
            "authority_roots": _roots_payload(request.roots),
            "proof_gaps": ordered_gaps,
            "assumptions": list(goal.assumption_refs),
            "metadata": {
                "disposition": goal.disposition.value
                if hasattr(goal.disposition, "value")
                else str(goal.disposition),
                "semantic_authority": False,
            },
        }
        if goal_cls is None:
            return MappingProxyType(payload)
        if hasattr(goal_cls, "from_dict"):
            return goal_cls.from_dict(payload)
        return goal_cls(**payload)

    def _build_generic_source(
        self, candidate: _SourceCandidate, symbols: Mapping[str, Any]
    ) -> Any:
        source_cls = symbols.get("TacticianSource")
        payload = {
            "source_id": candidate.source_id,
            "source_class": candidate.source_type.value,
            "precedence": candidate.precedence,
            "rationale": candidate.rationale,
            "query_hints": list(candidate.query_hints),
            "source_root": candidate.source_root,
            "metadata": {
                "premise_ids": list(candidate.premise_ids),
                "semantic_authority": False,
            },
        }
        if source_cls is None:
            return MappingProxyType(payload)
        if hasattr(source_cls, "from_dict"):
            return source_cls.from_dict(payload)
        return source_cls(**payload)

    def _project_search_plan(
        self,
        *,
        request: CodeTacticianRequest,
        candidates: Sequence[_SourceCandidate],
        generic_plan: Any,
        query_results: Sequence[CodeTacticianQueryResult],
    ) -> tuple[TacticianSearchPlan, list[CodeTacticianExclusion]]:
        exclusions: list[CodeTacticianExclusion] = []
        selected_routes_raw = list(getattr(generic_plan, "selected_routes", ()) or ())
        excluded_routes_raw = list(getattr(generic_plan, "excluded_routes", ()) or ())

        # Map generic source classes onto supervisor SourceRouteKind, preserving
        # local-first order and de-duplicating route kinds.
        ordered_routes: list[SourceRouteKind] = []
        selected_premise_ids: list[str] = []
        excluded_premise_ids: list[str] = []
        exclusion_rationale_refs: list[str] = []
        query_refs: list[str] = [item.query_id for item in request.query_specs]
        query_refs.extend(item.result_id for item in query_results)

        candidate_by_id = {item.source_id: item for item in candidates}
        selected_types: list[CodeSourceType] = []

        def _route_source_type(route: Any) -> CodeSourceType | None:
            source_id = str(getattr(route, "source_id", "") or "")
            if source_id in candidate_by_id:
                return candidate_by_id[source_id].source_type
            source_class = str(getattr(route, "source_class", "") or "")
            if source_class:
                try:
                    return parse_code_source_type(source_class)
                except CodeTacticianError:
                    return None
            return None

        for route in selected_routes_raw:
            source_type = _route_source_type(route)
            if source_type is None:
                continue
            selected_types.append(source_type)
            route_kind = map_code_source_to_route(source_type)
            if route_kind not in ordered_routes:
                ordered_routes.append(route_kind)
            candidate = candidate_by_id.get(str(getattr(route, "source_id", "")))
            if candidate is not None:
                selected_premise_ids.extend(candidate.premise_ids)

        for route in excluded_routes_raw:
            source_type = _route_source_type(route)
            rationale = str(getattr(route, "rationale", "") or "excluded")
            subject = str(
                getattr(route, "source_id", "")
                or getattr(route, "route_id", "")
                or "route"
            )
            exclusion = CodeTacticianExclusion(
                exclusion_id=_stable_id(
                    "exclusion",
                    {"subject": subject, "rationale": rationale},
                ),
                subject_ref=subject,
                source_type=source_type.value if source_type else "",
                rationale=rationale,
                stage="generic_plan",
            )
            exclusions.append(exclusion)
            exclusion_rationale_refs.append(exclusion.exclusion_id)
            candidate = candidate_by_id.get(subject)
            if candidate is not None:
                excluded_premise_ids.extend(candidate.premise_ids)

        # If the generic planner returned no selected routes, fall back to the
        # local-first candidate order (still non-authoritative).
        if not ordered_routes:
            for candidate in candidates[: request.policy.max_routes]:
                route_kind = map_code_source_to_route(candidate.source_type)
                if route_kind not in ordered_routes:
                    ordered_routes.append(route_kind)
                selected_premise_ids.extend(candidate.premise_ids)
                selected_types.append(candidate.source_type)

        # Enforce local-before-approximate on the projected route list.
        ordered_routes = self._enforce_local_first_routes(
            ordered_routes, selected_types or [c.source_type for c in candidates]
        )

        subgoals = self._project_subgoals(
            request=request,
            generic_plan=generic_plan,
            selected_types=selected_types or [c.source_type for c in candidates],
        )

        goal_ids = tuple(goal.goal_id for goal in request.goals)
        plan_id = _stable_id(
            "tactician-search-plan",
            {
                "roots": request.roots.content_id,
                "goal_ids": list(goal_ids),
                "routes": [item.value for item in ordered_routes],
                "selected_premises": sorted(set(selected_premise_ids)),
                "excluded_premises": sorted(set(excluded_premise_ids)),
                "queries": query_refs,
                "policy": request.policy.policy_id,
                "generic_plan": str(getattr(generic_plan, "plan_id", "")),
            },
        )

        # De-dupe premise ids while preserving sort for stability.
        selected_unique = tuple(sorted(set(selected_premise_ids)))
        excluded_unique = tuple(
            sorted(set(excluded_premise_ids) - set(selected_unique))
        )
        rationale_unique = tuple(sorted(set(exclusion_rationale_refs)))
        query_unique = tuple(dict.fromkeys(query_refs))  # stable unique

        invalidation_refs = tuple(
            sorted(
                {
                    request.roots.content_id,
                    request.roots.tree_id,
                    request.roots.corpus_id,
                    request.corpus.content_id,
                    *(goal.content_id for goal in request.goals),
                }
            )
        )

        planner_id = str(
            getattr(generic_plan, "planner_id", "") or CODE_TACTICIAN_PLANNER_ID
        )

        plan = TacticianSearchPlan(
            roots=request.roots,
            plan_id=plan_id,
            goal_ids=goal_ids,
            ordered_source_routes=tuple(ordered_routes),
            query_refs=query_unique,
            selected_premise_ids=selected_unique,
            excluded_premise_ids=excluded_unique,
            exclusion_rationale_refs=rationale_unique,
            subgoals=tuple(subgoals),
            planned_logic_family_refs=request.logic_family_refs,
            translation_refs=request.translation_refs,
            stop_policy_ref=request.policy.stop_policy_ref,
            escalation_policy_ref=request.policy.escalation_policy_ref,
            abstention_policy_ref=request.policy.abstention_policy_ref,
            resource_policy_ref=request.policy.resource_policy_ref,
            planner_id=planner_id,
            model_id=request.model_id or request.roots.model_id,
            config_id=request.config_id,
            semantic_authority=False,
            invalidation_refs=invalidation_refs,
        )
        return plan, exclusions

    def _enforce_local_first_routes(
        self,
        routes: Sequence[SourceRouteKind],
        selected_types: Sequence[CodeSourceType],
    ) -> list[SourceRouteKind]:
        """Re-order projected routes so local authoritative kinds come first."""

        local_kinds: list[SourceRouteKind] = []
        approx_kinds: list[SourceRouteKind] = []
        type_order = {item: index for index, item in enumerate(CODE_SOURCE_PRECEDENCE)}
        sorted_types = sorted(
            selected_types, key=lambda item: type_order.get(item, 999)
        )
        # Prefer type-driven order when available.
        if sorted_types:
            for source_type in sorted_types:
                kind = map_code_source_to_route(source_type)
                bucket = (
                    local_kinds
                    if source_type in _LOCAL_AUTHORITATIVE_SOURCES
                    else approx_kinds
                )
                if kind not in bucket and kind not in local_kinds + approx_kinds:
                    bucket.append(kind)
            return local_kinds + approx_kinds

        local_route_set = {
            map_code_source_to_route(item) for item in _LOCAL_AUTHORITATIVE_SOURCES
        }
        for route in routes:
            if route in local_route_set:
                if route not in local_kinds:
                    local_kinds.append(route)
            else:
                if route not in approx_kinds and route not in local_kinds:
                    approx_kinds.append(route)
        return local_kinds + approx_kinds

    def _project_subgoals(
        self,
        *,
        request: CodeTacticianRequest,
        generic_plan: Any,
        selected_types: Sequence[CodeSourceType],
    ) -> list[LogicSubgoal]:
        raw = list(getattr(generic_plan, "subgoals", ()) or ())
        subgoals: list[LogicSubgoal] = []
        primary_goal_id = request.goals[0].goal_id
        if raw:
            for index, item in enumerate(raw[: request.policy.max_subgoals]):
                subgoal_id = str(
                    getattr(item, "subgoal_id", "") or f"subgoal:projected:{index}"
                )
                parent = str(
                    getattr(item, "parent_goal_id", "") or primary_goal_id
                )
                # Bind subgoals to a request goal_id; remap foreign parents.
                goal_id = parent if parent in {g.goal_id for g in request.goals} else primary_goal_id
                depends_on = tuple(
                    str(dep)
                    for dep in list(getattr(item, "depends_on", ()) or ())
                    if str(dep) != subgoal_id
                )
                claim_ref = str(
                    getattr(item, "statement_ref", "")
                    or getattr(item, "claim_ref", "")
                    or f"claim:{subgoal_id}"
                )
                # Assign source route from selected types by index rank.
                if index < len(selected_types):
                    source_type = selected_types[index]
                elif selected_types:
                    source_type = selected_types[0]
                else:
                    source_type = CodeSourceType.TYPE_AND_EFFECT_FACTS
                source_route = map_code_source_to_route(source_type)
                source_authority = _CODE_SOURCE_AUTHORITY[source_type]
                subgoals.append(
                    LogicSubgoal(
                        subgoal_id=subgoal_id,
                        goal_id=goal_id,
                        disposition=SubgoalDisposition.PLANNED,
                        claim_ref=claim_ref,
                        depends_on=depends_on,
                        source_route=source_route,
                        source_authority=source_authority,
                        proof_status=ProofStatus.UNPROVED,
                        score_millipercent=0,
                    )
                )
            return subgoals

        # Deterministic gap-driven subgoals when the generic plan is silent.
        for index, source_type in enumerate(
            selected_types[: request.policy.max_subgoals]
        ):
            subgoal_id = f"subgoal:{source_type.value}"
            depends_on: tuple[str, ...] = ()
            if index > 0:
                prev = selected_types[index - 1]
                depends_on = (f"subgoal:{prev.value}",)
            subgoals.append(
                LogicSubgoal(
                    subgoal_id=subgoal_id,
                    goal_id=primary_goal_id,
                    disposition=SubgoalDisposition.PLANNED,
                    claim_ref=f"claim:{source_type.value}",
                    depends_on=depends_on,
                    source_route=map_code_source_to_route(source_type),
                    source_authority=_CODE_SOURCE_AUTHORITY[source_type],
                    proof_status=ProofStatus.UNPROVED,
                    score_millipercent=0,
                )
            )
        return subgoals


def inspect_code_tactician_capability(
    policy: Mapping[str, Any] | CodeTacticianPolicy | None = None,
) -> CodeTacticianCapability:
    """Pure capability inspection that never imports optional packages."""

    if policy is None:
        CodeTacticianPolicy()
    elif isinstance(policy, CodeTacticianPolicy):
        pass
    else:
        CodeTacticianPolicy.from_dict(policy)
    return CodeTacticianCapability(
        imported=False,
        available=False,
        health="lazy",
        reason_code="lazy",
    )


__all__ = [
    "IPFS_DATASETS_TACTICIAN_PROVIDER_ID",
    "IPFS_DATASETS_TACTICIAN_PROVIDER_VERSION",
    "GENERIC_TACTICIAN_INTERFACE",
    "GENERIC_TACTICIAN_MODULE",
    "CODE_TACTICIAN_PLANNER_ID",
    "CODE_SOURCE_PRECEDENCE",
    "CodeSourceType",
    "CodeTacticianStatus",
    "CodeTacticianReasonCode",
    "CodeTacticianError",
    "CodeTacticianPolicy",
    "CodeTacticianQuerySpec",
    "CodeTacticianQueryResult",
    "CodeTacticianExclusion",
    "CodeTacticianRequest",
    "CodeTacticianResponse",
    "CodeTacticianCapability",
    "IpfsDatasetsTacticianProvider",
    "inspect_code_tactician_capability",
    "code_source_rank",
    "is_local_authoritative",
    "is_approximate_or_model",
    "map_premise_source",
    "map_code_source_to_route",
    "parse_code_source_type",
    "project_source_candidates",
]
