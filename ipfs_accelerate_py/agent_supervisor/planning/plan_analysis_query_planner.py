"""Deterministic reasoning-query planning for create, steer, and diagnosis.

``ReasoningQueryPlan@1`` is the provider-free policy boundary between a
planning/Doctor input and the shared analysis operation and strategy
registries.  The planner always emits the mandatory query portfolio before it
considers caller- or model-nominated optional work.  Query records bind exact
roots and selectors, registry capability declarations, hard resource bounds,
content-addressed cache semantics, and fail-closed behavior.

Model output is nomination-only.  It cannot remove a mandatory query, choose a
provider, endpoint, or credential, or satisfy post-proposal evidence coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..analysis.analysis_operation_registry import (
    AnalysisOperation,
    AnalysisOperationRegistry,
    AnalysisOperationRegistryError,
    LogicFamily,
    default_operation_specs,
    normalize_analysis_operation,
    normalize_logic_family,
)
from ..analysis.analysis_strategy_registry import (
    AnalysisStrategyRegistry,
    PropertyQuestionClass,
    create_default_analysis_strategy_registry,
    property_class_for_operation,
)
from ..analysis.planning_evidence_bundle import (
    CoverageDecision as BundleCoverageDecision,
)
from ..analysis.planning_evidence_bundle import (
    PlanningEvidenceBundle,
)
from .plan_revision_contracts import (
    PlanCreateRequest,
    PlanRequestBudget,
    PlanSteerRequest,
)

REASONING_QUERY_PLAN_INTERFACE: Final[str] = "ReasoningQueryPlan@1"
REASONING_QUERY_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-query-plan@1"
)
REASONING_QUERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-query@1"
)
QUERY_SCOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-query-scope@1"
)
QUERY_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-query-bounds@1"
)
POST_PROPOSAL_COVERAGE_INTERFACE: Final[str] = "EvidenceCoverageReceipt@1"
POST_PROPOSAL_COVERAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/post-proposal-evidence-coverage@1"
)

_MAX_TEXT_BYTES = 8 * 1024
_MAX_SELECTORS = 4_096
_MAX_QUERIES = 256
_SPACE_RE = re.compile(r"\s+")
_SENSITIVE_SUGGESTION_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "base_url",
        "cookie",
        "credential",
        "credentials",
        "endpoint",
        "endpoint_id",
        "host",
        "password",
        "private_key",
        "provider",
        "provider_id",
        "refresh_token",
        "secret",
        "session",
        "token",
        "uri",
        "url",
    }
)


class QueryPlanningError(ValueError):
    """A query plan cannot be compiled without weakening its contract."""


class QueryPlanningBudgetError(QueryPlanningError):
    """The supplied request budget cannot contain the mandatory portfolio."""


class UnsafeModelSuggestionError(QueryPlanningError):
    """A model suggestion attempted provider, endpoint, or secret selection."""


class QueryInputKind(str, Enum):
    CREATE = "create"
    STEER = "steer"
    DIAGNOSIS = "diagnosis"


class QueryEvidenceSlot(str, Enum):
    SYMBOL_IMPACT = "symbol_impact"
    GRAPHRAG_NOMINATION = "graphrag_nomination"
    PREMISES = "premises"
    CONTRADICTIONS = "contradictions"
    LOGIC_TRANSLATION = "logic_translation"
    PROOF = "proof"
    COUNTEREXAMPLE = "counterexample"
    SECURITY = "security"


class QueryRequirement(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class QueryFailureDisposition(str, Enum):
    BLOCK_CANDIDATE_GENERATION = "block_candidate_generation"
    RECORD_UNCERTAINTY_DEBT = "record_uncertainty_debt"


class QueryPlanDecision(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


class ClaimClass(str, Enum):
    CODE = "code"
    POLICY = "policy"
    SECURITY = "security"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    PROOF = "proof"
    COMPLETION = "completion"
    GENERIC = "generic"


class EvidenceAuthority(str, Enum):
    PROMPT = "prompt_nomination"
    MODEL = "model_nomination"
    RETRIEVAL = "retrieval_nomination"
    CURRENT_ROOT_FACT = "current_root_fact"
    REVIEWED_CONTRACT = "reviewed_contract"
    REVIEWED_POLICY = "reviewed_policy"
    SECURITY_ANALYSIS = "security_analysis"
    BOUNDED_OBSERVATION = "bounded_observation"
    PROOF_RECEIPT = "proof_receipt"
    COUNTEREXAMPLE = "counterexample"


class CoverageDisposition(str, Enum):
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    OPTIONAL_MISSING = "optional_missing"


class CoverageDecision(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise QueryPlanningError(f"{name} must be a string")
    result = _SPACE_RE.sub(" ", value).strip()
    if required and not result:
        raise QueryPlanningError(f"{name} must not be empty")
    if "\x00" in result:
        raise QueryPlanningError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise QueryPlanningError(f"{name} exceeds {maximum} UTF-8 bytes")
    return result


def _nonnegative_int(value: Any, name: str, maximum: int = 2**63 - 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise QueryPlanningError(f"{name} must be an integer from 0 through {maximum}")
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QueryPlanningError("canonical values must be finite")
        return format(value, ".17g")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    return str(value)


def _content_id(namespace: str, value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {name: getattr(value, name) for name in fields if hasattr(value, name)}
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _strings(
    value: Any,
    name: str,
    *,
    paths: bool = False,
    maximum: int = _MAX_SELECTORS,
) -> tuple[str, ...]:
    result: set[str] = set()
    for item in _sequence(value):
        raw = getattr(item, "value", item)
        if not isinstance(raw, str):
            continue
        text = _text(raw, name, required=False, maximum=2048)
        if not text:
            continue
        if paths:
            path = PurePosixPath(text)
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise QueryPlanningError(f"{name} contains a non-canonical path")
            text = path.as_posix()
        result.add(text)
    if len(result) > maximum:
        raise QueryPlanningError(f"{name} exceeds {maximum} entries")
    return tuple(sorted(result))


def _nested_values(data: Mapping[str, Any], *names: str) -> tuple[Any, ...]:
    wanted = {name.casefold() for name in names}
    found: list[Any] = []

    def visit(value: Any, depth: int) -> None:
        if depth > 5:
            return
        mapping = _as_mapping(value)
        if mapping:
            for key, item in mapping.items():
                if str(key).casefold() in wanted:
                    found.extend(_sequence(item))
                elif isinstance(item, Mapping):
                    visit(item, depth + 1)

    visit(data, 0)
    return tuple(found)


def _assert_safe_model_suggestions(value: Any, path: str = "model_suggestions") -> None:
    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if name in _SENSITIVE_SUGGESTION_NAMES or any(
                name.endswith("_" + marker)
                for marker in ("credential", "endpoint", "password", "secret", "token")
            ):
                raise UnsafeModelSuggestionError(
                    f"{path} may not select credentials, endpoints, or providers"
                )
            _assert_safe_model_suggestions(item, f"{path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_safe_model_suggestions(item, f"{path}[{index}]")


@dataclass(frozen=True)
class QueryScope:
    """Exact authority roots and selectors for a reasoning query."""

    repository_id: str
    tree_id: str
    objective_revision: str
    policy_id: str
    capability_catalog_root: str
    provider_catalog_root: str
    security_ir_root: str
    intent_ir_root: str
    paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    contracts: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    evidence_references: tuple[str, ...] = ()
    open_frontiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_revision",
            "policy_id",
            "capability_catalog_root",
            "provider_catalog_root",
            "security_ir_root",
            "intent_ir_root",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=2048)
            )
        for name in ("paths", "changed_paths"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, paths=True)
            )
        for name in (
            "symbols",
            "contracts",
            "obligation_ids",
            "evidence_references",
            "open_frontiers",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))

    @property
    def scope_id(self) -> str:
        return _content_id("reasoning-query-scope", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUERY_SCOPE_SCHEMA,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "capability_catalog_root": self.capability_catalog_root,
            "provider_catalog_root": self.provider_catalog_root,
            "security_ir_root": self.security_ir_root,
            "intent_ir_root": self.intent_ir_root,
            "paths": list(self.paths),
            "changed_paths": list(self.changed_paths),
            "symbols": list(self.symbols),
            "contracts": list(self.contracts),
            "obligation_ids": list(self.obligation_ids),
            "evidence_references": list(self.evidence_references),
            "open_frontiers": list(self.open_frontiers),
        }


@dataclass(frozen=True)
class QueryBounds:
    """Hard, integer-unit ceiling for one query."""

    max_input_bytes: int
    max_output_bytes: int
    max_items: int
    timeout_ms: int
    max_cost_micros: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))

    @property
    def executable(self) -> bool:
        return (
            self.max_input_bytes > 0
            and self.max_output_bytes > 0
            and self.max_items > 0
            and self.timeout_ms > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUERY_BOUNDS_SCHEMA,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_items": self.max_items,
            "timeout_ms": self.timeout_ms,
            "max_cost_micros": self.max_cost_micros,
        }


@dataclass(frozen=True)
class QueryCachePolicy:
    cacheable: bool
    scope: str
    key_dimensions: tuple[str, ...]
    allow_stale: bool
    reuse_requires_equivalent_provenance: bool
    semantic_cache_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _text(self.scope, "cache scope"))
        object.__setattr__(
            self,
            "key_dimensions",
            _strings(self.key_dimensions, "cache key dimensions"),
        )
        object.__setattr__(
            self,
            "semantic_cache_key",
            _text(self.semantic_cache_key, "semantic_cache_key"),
        )
        if self.allow_stale:
            raise QueryPlanningError("reasoning queries may not reuse stale evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cacheable": self.cacheable,
            "scope": self.scope,
            "key_dimensions": list(self.key_dimensions),
            "allow_stale": False,
            "reuse_requires_equivalent_provenance": (
                self.reuse_requires_equivalent_provenance
            ),
            "semantic_cache_key": self.semantic_cache_key,
            "cache_miss_is_refutation": False,
        }


@dataclass(frozen=True)
class QueryFailureSemantics:
    unavailable: QueryFailureDisposition
    timeout: QueryFailureDisposition
    malformed_result: QueryFailureDisposition
    stale_result: QueryFailureDisposition
    deterministic_fallback: str
    explicit_receipt_required: bool = True

    def __post_init__(self) -> None:
        for name in ("unavailable", "timeout", "malformed_result", "stale_result"):
            object.__setattr__(self, name, QueryFailureDisposition(getattr(self, name)))
        object.__setattr__(
            self,
            "deterministic_fallback",
            _text(self.deterministic_fallback, "deterministic_fallback"),
        )
        if not self.explicit_receipt_required:
            raise QueryPlanningError("query failures require an explicit receipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unavailable": self.unavailable.value,
            "timeout": self.timeout.value,
            "malformed_result": self.malformed_result.value,
            "stale_result": self.stale_result.value,
            "deterministic_fallback": self.deterministic_fallback,
            "explicit_receipt_required": True,
        }


@dataclass(frozen=True)
class ReasoningQuery:
    """One registry-routed, bounded reasoning question."""

    slot: QueryEvidenceSlot
    operation: AnalysisOperation
    requirement: QueryRequirement
    why: str
    question: str
    scope: QueryScope
    logic_family: LogicFamily | None
    operation_spec_id: str
    strategy_ids: tuple[str, ...]
    provider_capabilities: tuple[str, ...]
    bounds: QueryBounds
    cache: QueryCachePolicy
    failure: QueryFailureSemantics
    nomination_only: bool = False
    suggestion_source: str = "fixed_rule"

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", QueryEvidenceSlot(self.slot))
        object.__setattr__(
            self, "operation", normalize_analysis_operation(self.operation)
        )
        object.__setattr__(self, "requirement", QueryRequirement(self.requirement))
        object.__setattr__(self, "why", _text(self.why, "why"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        if not isinstance(self.scope, QueryScope):
            raise QueryPlanningError("query scope must be QueryScope")
        if self.logic_family is not None:
            object.__setattr__(
                self, "logic_family", normalize_logic_family(self.logic_family)
            )
        object.__setattr__(
            self,
            "operation_spec_id",
            _text(self.operation_spec_id, "operation_spec_id"),
        )
        object.__setattr__(
            self, "strategy_ids", _strings(self.strategy_ids, "strategy_ids")
        )
        object.__setattr__(
            self,
            "provider_capabilities",
            _strings(self.provider_capabilities, "provider_capabilities"),
        )
        if not self.provider_capabilities:
            raise QueryPlanningError("query must bind provider capabilities")
        if not isinstance(self.bounds, QueryBounds):
            raise QueryPlanningError("query bounds must be QueryBounds")
        if not isinstance(self.cache, QueryCachePolicy):
            raise QueryPlanningError("query cache must be QueryCachePolicy")
        if not isinstance(self.failure, QueryFailureSemantics):
            raise QueryPlanningError("query failure must be QueryFailureSemantics")
        object.__setattr__(
            self,
            "suggestion_source",
            _text(self.suggestion_source, "suggestion_source", maximum=128),
        )
        if (
            self.slot is QueryEvidenceSlot.GRAPHRAG_NOMINATION
            and not self.nomination_only
        ):
            raise QueryPlanningError("GraphRAG must remain nomination-only")
        if (
            self.requirement is QueryRequirement.REQUIRED
            and self.failure.unavailable
            is not QueryFailureDisposition.BLOCK_CANDIDATE_GENERATION
        ):
            raise QueryPlanningError("required query absence must block generation")

    @property
    def query_id(self) -> str:
        return _content_id("reasoning-query", self.to_dict(include_query_id=False))

    @property
    def required(self) -> bool:
        return self.requirement is QueryRequirement.REQUIRED

    @property
    def evidence_slot(self) -> str:
        return self.slot.value

    @property
    def capability_requirements(self) -> tuple[str, ...]:
        return self.provider_capabilities

    @property
    def max_bytes(self) -> int:
        return self.bounds.max_output_bytes

    @property
    def max_items(self) -> int:
        return self.bounds.max_items

    @property
    def timeout_ms(self) -> int:
        return self.bounds.timeout_ms

    @property
    def max_cost_micros(self) -> int:
        return self.bounds.max_cost_micros

    @property
    def cache_key(self) -> str:
        return self.cache.semantic_cache_key

    def to_dict(self, *, include_query_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REASONING_QUERY_SCHEMA,
            "slot": self.slot.value,
            "operation": self.operation.value,
            "requirement": self.requirement.value,
            "why": self.why,
            "question": self.question,
            "scope": self.scope.to_dict(),
            "logic_family": self.logic_family.value if self.logic_family else "",
            "operation_spec_id": self.operation_spec_id,
            "strategy_ids": list(self.strategy_ids),
            "provider_capabilities": list(self.provider_capabilities),
            "provider_selection": "registry_at_execution",
            "endpoint_selection": False,
            "credential_selection": False,
            "bounds": self.bounds.to_dict(),
            "cache": self.cache.to_dict(),
            "failure": self.failure.to_dict(),
            "nomination_only": self.nomination_only,
            "suggestion_source": self.suggestion_source,
            "authority": {
                "proposal_only": True,
                "proof_authority": False,
                "completion_authority": False,
            },
        }
        if include_query_id:
            payload["query_id"] = self.query_id
        return payload

    def to_analysis_request(self, registry: AnalysisOperationRegistry) -> Any:
        """Build the only executable request form, through the registry."""

        references = tuple(
            {"reference_id": item, "kind": "evidence"}
            for item in self.scope.evidence_references
        )
        return registry.build_request(
            self.operation,
            self.question,
            artifact_references=references,
            logic_family=self.logic_family,
            repository_id=self.scope.repository_id,
            tree_id=self.scope.tree_id,
            objective_revision=self.scope.objective_revision,
            policy_id=self.scope.policy_id,
            timeout_ms=self.bounds.timeout_ms,
            request_id=self.query_id,
        )


@dataclass(frozen=True)
class ReasoningQueryPlan:
    input_kind: QueryInputKind
    request_id: str
    scope: QueryScope
    queries: tuple[ReasoningQuery, ...]
    operation_registry_id: str
    strategy_registry_id: str
    decision: QueryPlanDecision = QueryPlanDecision.READY
    blockers: tuple[str, ...] = ()
    interface: str = REASONING_QUERY_PLAN_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_kind", QueryInputKind(self.input_kind))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        if not isinstance(self.scope, QueryScope):
            raise QueryPlanningError("plan scope must be QueryScope")
        if len(self.queries) > _MAX_QUERIES:
            raise QueryPlanningError(f"query plan exceeds {_MAX_QUERIES} queries")
        ordered = tuple(
            sorted(
                self.queries,
                key=lambda query: (
                    query.requirement is QueryRequirement.OPTIONAL,
                    _MANDATORY_SLOT_ORDER.index(query.slot)
                    if query.slot in _MANDATORY_SLOT_ORDER
                    else len(_MANDATORY_SLOT_ORDER),
                    query.operation.value,
                    query.query_id,
                ),
            )
        )
        if len({item.query_id for item in ordered}) != len(ordered):
            raise QueryPlanningError("query plan contains duplicate queries")
        object.__setattr__(self, "queries", ordered)
        object.__setattr__(
            self,
            "operation_registry_id",
            _text(self.operation_registry_id, "operation_registry_id"),
        )
        object.__setattr__(
            self,
            "strategy_registry_id",
            _text(self.strategy_registry_id, "strategy_registry_id"),
        )
        object.__setattr__(self, "decision", QueryPlanDecision(self.decision))
        object.__setattr__(self, "blockers", _strings(self.blockers, "blockers"))
        missing = set(_MANDATORY_SLOT_ORDER) - {
            item.slot
            for item in ordered
            if item.requirement is QueryRequirement.REQUIRED
        }
        if missing:
            raise QueryPlanningError(
                "mandatory query slots are missing: "
                + ", ".join(sorted(item.value for item in missing))
            )
        expected = (
            QueryPlanDecision.BLOCKED if self.blockers else QueryPlanDecision.READY
        )
        if self.decision is not expected:
            raise QueryPlanningError("query plan decision does not match blockers")

    @property
    def plan_id(self) -> str:
        return _content_id("reasoning-query-plan", self.to_dict(include_plan_id=False))

    @property
    def query_plan_id(self) -> str:
        return self.plan_id

    @property
    def exact_scope(self) -> QueryScope:
        return self.scope

    @property
    def required_queries(self) -> tuple[ReasoningQuery, ...]:
        return tuple(
            item
            for item in self.queries
            if item.requirement is QueryRequirement.REQUIRED
        )

    @property
    def optional_queries(self) -> tuple[ReasoningQuery, ...]:
        return tuple(
            item
            for item in self.queries
            if item.requirement is QueryRequirement.OPTIONAL
        )

    @property
    def ready(self) -> bool:
        return self.decision is QueryPlanDecision.READY

    def query_for_slot(self, slot: QueryEvidenceSlot | str) -> ReasoningQuery:
        wanted = QueryEvidenceSlot(slot)
        for query in self.queries:
            if query.slot is wanted and query.requirement is QueryRequirement.REQUIRED:
                return query
        raise KeyError(wanted.value)

    def queries_for_slot(
        self, slot: QueryEvidenceSlot | str
    ) -> tuple[ReasoningQuery, ...]:
        wanted = QueryEvidenceSlot(slot)
        return tuple(item for item in self.queries if item.slot is wanted)

    def to_dict(self, *, include_plan_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REASONING_QUERY_PLAN_SCHEMA,
            "interface": self.interface,
            "input_kind": self.input_kind.value,
            "request_id": self.request_id,
            "scope": self.scope.to_dict(),
            "operation_registry_id": self.operation_registry_id,
            "strategy_registry_id": self.strategy_registry_id,
            "decision": self.decision.value,
            "blockers": list(self.blockers),
            "queries": [item.to_dict() for item in self.queries],
            "required_query_ids": [item.query_id for item in self.required_queries],
            "optional_query_ids": [item.query_id for item in self.optional_queries],
            "model_suggestions_are_nomination_only": True,
            "provider_selection_deferred_to_registry": True,
        }
        if include_plan_id:
            payload["plan_id"] = self.plan_id
        return payload


@dataclass(frozen=True)
class ProposalClaim:
    claim_id: str
    claim_class: ClaimClass
    evidence_authorities: tuple[EvidenceAuthority, ...] = ()
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_class", ClaimClass(self.claim_class))
        object.__setattr__(
            self,
            "evidence_authorities",
            tuple(
                sorted(
                    {EvidenceAuthority(item) for item in self.evidence_authorities},
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _strings(self.evidence_references, "evidence_references"),
        )

    @classmethod
    def from_value(cls, value: ProposalClaim | Mapping[str, Any]) -> ProposalClaim:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise QueryPlanningError("proposal claim must be a mapping")
        return cls(
            claim_id=str(value.get("claim_id") or value.get("id") or ""),
            claim_class=value.get("claim_class") or value.get("kind") or "generic",
            evidence_authorities=tuple(
                value.get("evidence_authorities") or value.get("authorities") or ()
            ),
            evidence_references=tuple(
                value.get("evidence_references") or value.get("evidence_refs") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_class": self.claim_class.value,
            "evidence_authorities": [item.value for item in self.evidence_authorities],
            "evidence_references": list(self.evidence_references),
        }


@dataclass(frozen=True)
class CoverageSlot:
    slot: str
    required: bool
    disposition: CoverageDisposition
    evidence_references: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _text(self.slot, "coverage slot"))
        object.__setattr__(self, "disposition", CoverageDisposition(self.disposition))
        object.__setattr__(
            self,
            "evidence_references",
            _strings(self.evidence_references, "evidence_references"),
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "required": self.required,
            "disposition": self.disposition.value,
            "evidence_references": list(self.evidence_references),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PostProposalCoverageReceipt:
    """A fresh coverage decision over the generated proposal and query plan."""

    query_plan_id: str
    proposal_id: str
    current_tree_id: str
    query_slots: tuple[CoverageSlot, ...]
    claim_slots: tuple[CoverageSlot, ...]
    decision: CoverageDecision
    blockers: tuple[str, ...] = ()
    phase: str = "post_proposal"
    interface: str = POST_PROPOSAL_COVERAGE_INTERFACE

    def __post_init__(self) -> None:
        for name in ("query_plan_id", "proposal_id", "current_tree_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.phase != "post_proposal":
            raise QueryPlanningError("coverage must rerun after proposal")
        object.__setattr__(self, "decision", CoverageDecision(self.decision))
        object.__setattr__(self, "blockers", _strings(self.blockers, "blockers"))
        blocked = any(
            item.required and item.disposition is CoverageDisposition.BLOCKED
            for item in (*self.query_slots, *self.claim_slots)
        )
        expected = CoverageDecision.BLOCKED if blocked else CoverageDecision.READY
        if self.decision is not expected:
            raise QueryPlanningError("coverage decision does not match slot coverage")
        if blocked and not self.blockers:
            raise QueryPlanningError("blocked coverage needs reason-coded blockers")

    @property
    def receipt_id(self) -> str:
        return _content_id(
            "post-proposal-evidence-coverage",
            self.to_dict(include_receipt_id=False),
        )

    @property
    def ready(self) -> bool:
        return self.decision is CoverageDecision.READY

    @property
    def planning_blocked(self) -> bool:
        return not self.ready

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": POST_PROPOSAL_COVERAGE_SCHEMA,
            "interface": self.interface,
            "phase": "post_proposal",
            "query_plan_id": self.query_plan_id,
            "proposal_id": self.proposal_id,
            "current_tree_id": self.current_tree_id,
            "query_slots": [item.to_dict() for item in self.query_slots],
            "claim_slots": [item.to_dict() for item in self.claim_slots],
            "decision": self.decision.value,
            "planning_blocked": self.planning_blocked,
            "blockers": list(self.blockers),
            "prompt_or_model_evidence_is_never_authoritative": True,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


_MANDATORY_SLOT_ORDER: Final[tuple[QueryEvidenceSlot, ...]] = (
    QueryEvidenceSlot.SYMBOL_IMPACT,
    QueryEvidenceSlot.GRAPHRAG_NOMINATION,
    QueryEvidenceSlot.PREMISES,
    QueryEvidenceSlot.CONTRADICTIONS,
    QueryEvidenceSlot.LOGIC_TRANSLATION,
    QueryEvidenceSlot.PROOF,
    QueryEvidenceSlot.COUNTEREXAMPLE,
    QueryEvidenceSlot.SECURITY,
)

_MANDATORY_RULES: Final[
    Mapping[
        QueryEvidenceSlot,
        tuple[AnalysisOperation, str, str, LogicFamily | None, bool],
    ]
] = MappingProxyType(
    {
        QueryEvidenceSlot.SYMBOL_IMPACT: (
            AnalysisOperation.SYMBOL_IMPACT,
            "Identify exact definitions, callers, consumers, and reverse dependencies in scope.",
            "Resolve symbols and the complete bounded impact frontier for the requested change or diagnosis.",
            None,
            False,
        ),
        QueryEvidenceSlot.GRAPHRAG_NOMINATION: (
            AnalysisOperation.GRAPH_RAG_RETRIEVAL,
            "Nominate related code, contracts, tests, and history without granting semantic authority.",
            "Nominate relevant graph neighborhoods for the exact scope; return references only.",
            None,
            True,
        ),
        QueryEvidenceSlot.PREMISES: (
            AnalysisOperation.PREMISE_SELECTION,
            "Select explicit premises and assumptions needed by the obligations.",
            "Select current, provenance-bound premises for the requested behavior and affected scope.",
            LogicFamily.TDFOL,
            False,
        ),
        QueryEvidenceSlot.CONTRADICTIONS: (
            AnalysisOperation.CONTRADICTION_SEARCH,
            "Find inconsistent requirements, assumptions, contracts, and observed facts.",
            "Search for contradictions among requested behavior, current contracts, and observed facts.",
            LogicFamily.TDFOL,
            False,
        ),
        QueryEvidenceSlot.LOGIC_TRANSLATION: (
            AnalysisOperation.LOGIC_TRANSLATION,
            "Translate prose and contract concepts into a checked logic-family representation.",
            "Translate the requested predicates, constraints, and invalidators for independent checking.",
            LogicFamily.TDFOL,
            False,
        ),
        QueryEvidenceSlot.PROOF: (
            AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
            "Nominate proof obligations and proof strategies; candidates do not prove themselves.",
            "Derive proof candidates for desired behavior and changed-scope obligations.",
            LogicFamily.TDFOL,
            False,
        ),
        QueryEvidenceSlot.COUNTEREXAMPLE: (
            AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
            "Search for bounded counterexamples and failed assumptions before candidate generation.",
            "Search for counterexample candidates against the requested behavior and current premises.",
            LogicFamily.TDFOL,
            False,
        ),
        QueryEvidenceSlot.SECURITY: (
            AnalysisOperation.CONTRADICTION_SEARCH,
            "Check policy, authorization, information-flow, protocol, and SecurityIR conflicts independently.",
            "Search for security, authorization, policy, and protocol contradictions in the exact affected scope.",
            LogicFamily.DEONTIC,
            False,
        ),
    }
)

_REQUIRED_CLAIM_AUTHORITIES: Final[
    Mapping[ClaimClass, frozenset[EvidenceAuthority]]
] = MappingProxyType(
    {
        ClaimClass.CODE: frozenset(
            {
                EvidenceAuthority.CURRENT_ROOT_FACT,
                EvidenceAuthority.BOUNDED_OBSERVATION,
                EvidenceAuthority.PROOF_RECEIPT,
            }
        ),
        ClaimClass.POLICY: frozenset(
            {
                EvidenceAuthority.REVIEWED_POLICY,
                EvidenceAuthority.REVIEWED_CONTRACT,
            }
        ),
        ClaimClass.SECURITY: frozenset(
            {
                EvidenceAuthority.SECURITY_ANALYSIS,
                EvidenceAuthority.PROOF_RECEIPT,
            }
        ),
        ClaimClass.AUTHORIZATION: frozenset(
            {
                EvidenceAuthority.REVIEWED_POLICY,
                EvidenceAuthority.SECURITY_ANALYSIS,
                EvidenceAuthority.PROOF_RECEIPT,
            }
        ),
        ClaimClass.RESOURCE: frozenset(
            {
                EvidenceAuthority.CURRENT_ROOT_FACT,
                EvidenceAuthority.BOUNDED_OBSERVATION,
            }
        ),
        ClaimClass.PROOF: frozenset({EvidenceAuthority.PROOF_RECEIPT}),
        ClaimClass.COMPLETION: frozenset(
            {
                EvidenceAuthority.BOUNDED_OBSERVATION,
                EvidenceAuthority.PROOF_RECEIPT,
            }
        ),
        ClaimClass.GENERIC: frozenset(
            {
                EvidenceAuthority.CURRENT_ROOT_FACT,
                EvidenceAuthority.REVIEWED_CONTRACT,
                EvidenceAuthority.BOUNDED_OBSERVATION,
                EvidenceAuthority.PROOF_RECEIPT,
            }
        ),
    }
)


class PlanAnalysisQueryPlanner:
    """Compile fixed mandatory queries and independently recheck coverage."""

    def __init__(
        self,
        *,
        operation_registry: AnalysisOperationRegistry | None = None,
        strategy_registry: AnalysisStrategyRegistry | None = None,
    ) -> None:
        if operation_registry is None:
            operation_registry = AnalysisOperationRegistry()
            for spec in default_operation_specs():
                operation_registry.register_operation(spec)
        if strategy_registry is None:
            strategy_registry = create_default_analysis_strategy_registry(
                include_provider_adapters=False
            )
        if not isinstance(operation_registry, AnalysisOperationRegistry):
            raise QueryPlanningError(
                "operation_registry must be AnalysisOperationRegistry"
            )
        if not isinstance(strategy_registry, AnalysisStrategyRegistry):
            raise QueryPlanningError(
                "strategy_registry must be AnalysisStrategyRegistry"
            )
        self.operation_registry = operation_registry
        self.strategy_registry = strategy_registry

    def compile(
        self,
        request: PlanCreateRequest | PlanSteerRequest | Mapping[str, Any] | Any,
        *,
        context: Mapping[str, Any] | None = None,
        model_suggestions: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> ReasoningQueryPlan:
        kind, request_id, data, budget = self._normalize_input(request)
        merged = dict(data)
        if context:
            merged["query_context"] = dict(context)
        suggestions = model_suggestions or {}
        _assert_safe_model_suggestions(suggestions)

        scope = self._scope(kind, request, merged)
        concepts = _strings(
            _nested_values(merged, "concepts", "directive_concepts", "keywords"),
            "concepts",
        )
        logic_family = self._logic_family(merged)
        required_extra, optional_extra = self._requested_operations(
            kind, request, merged, suggestions
        )
        mandatory_count = len(_MANDATORY_SLOT_ORDER) + len(required_extra)
        if budget.max_analysis_operations < mandatory_count:
            raise QueryPlanningBudgetError(
                "max_analysis_operations cannot contain all required queries"
            )
        per_query = self._allocate_bounds(budget, mandatory_count)

        queries: list[ReasoningQuery] = []
        blockers: list[str] = []
        for index, slot in enumerate(_MANDATORY_SLOT_ORDER):
            operation, why, question, family, nomination_only = _MANDATORY_RULES[slot]
            if family is LogicFamily.TDFOL:
                family = logic_family
            bounds = per_query[index]
            query = self._build_query(
                slot=slot,
                operation=operation,
                requirement=QueryRequirement.REQUIRED,
                why=why,
                question=self._question(question, scope, concepts),
                scope=scope,
                logic_family=family,
                bounds=bounds,
                nomination_only=nomination_only,
                source="fixed_rule",
            )
            queries.append(query)
            if not bounds.executable:
                blockers.append(f"budget_exhausted:{slot.value}")

        for offset, operation in enumerate(required_extra, start=len(queries)):
            slot = self._slot_for_operation(operation)
            bounds = per_query[offset]
            queries.append(
                self._build_query(
                    slot=slot,
                    operation=operation,
                    requirement=QueryRequirement.REQUIRED,
                    why="The typed request explicitly requires this registered analysis operation.",
                    question=self._question(
                        f"Run required {operation.value} analysis for the exact request scope.",
                        scope,
                        concepts,
                    ),
                    scope=scope,
                    logic_family=(
                        logic_family
                        if self.operation_registry.operation(operation).logic_families
                        else None
                    ),
                    bounds=bounds,
                    nomination_only=operation is AnalysisOperation.GRAPH_RAG_RETRIEVAL,
                    source="typed_request",
                )
            )

        remaining = budget.max_analysis_operations - len(queries)
        for operation, source in optional_extra[:remaining]:
            optional_bounds = self._optional_bounds(budget, len(queries) + 1)
            queries.append(
                self._build_query(
                    slot=self._slot_for_operation(operation),
                    operation=operation,
                    requirement=QueryRequirement.OPTIONAL,
                    why="Optional registered analysis may reduce uncertainty debt.",
                    question=self._question(
                        f"Optionally run {operation.value} analysis for the exact request scope.",
                        scope,
                        concepts,
                    ),
                    scope=scope,
                    logic_family=(
                        logic_family
                        if self.operation_registry.operation(operation).logic_families
                        else None
                    ),
                    bounds=optional_bounds,
                    nomination_only=operation is AnalysisOperation.GRAPH_RAG_RETRIEVAL,
                    source=source,
                )
            )

        return ReasoningQueryPlan(
            input_kind=kind,
            request_id=request_id,
            scope=scope,
            queries=tuple(queries),
            operation_registry_id=self.operation_registry.registry_id,
            strategy_registry_id=self.strategy_registry.registry_id,
            decision=(
                QueryPlanDecision.BLOCKED if blockers else QueryPlanDecision.READY
            ),
            blockers=tuple(blockers),
        )

    compile_create = compile
    compile_steer = compile
    compile_diagnosis = compile
    plan = compile

    def rerun_coverage_after_proposal(
        self,
        query_plan: ReasoningQueryPlan,
        proposal: Mapping[str, Any] | Any,
        *,
        query_evidence: Mapping[str, Any] | None = None,
        evidence_bundle: PlanningEvidenceBundle | None = None,
    ) -> PostProposalCoverageReceipt:
        """Re-evaluate every required query and proposal claim from scratch."""

        if not isinstance(query_plan, ReasoningQueryPlan):
            raise QueryPlanningError("query_plan must be ReasoningQueryPlan")
        payload = _as_mapping(proposal)
        proposal_id = str(
            payload.get("proposal_id")
            or payload.get("plan_id")
            or payload.get("content_id")
            or _content_id("proposal", payload)
        )
        supplied = query_evidence or payload.get("query_evidence") or {}
        if not isinstance(supplied, Mapping):
            raise QueryPlanningError("query_evidence must be a mapping")

        bundle_rejected = (
            evidence_bundle is not None
            and evidence_bundle.coverage.decision is not BundleCoverageDecision.READY
        )
        bundle_stale = (
            evidence_bundle is not None
            and evidence_bundle.current_root_id != query_plan.scope.tree_id
        )
        query_slots: list[CoverageSlot] = []
        blockers: list[str] = []
        for query in query_plan.queries:
            raw_refs = supplied.get(query.query_id, supplied.get(query.slot.value, ()))
            authority: EvidenceAuthority | None = None
            evidence_tree_id = ""
            evidence_query_id = ""
            if isinstance(raw_refs, Mapping):
                authority_value = raw_refs.get("authority")
                if authority_value:
                    try:
                        authority = EvidenceAuthority(authority_value)
                    except ValueError as exc:
                        raise QueryPlanningError(
                            "query evidence has an unknown authority label"
                        ) from exc
                evidence_tree_id = str(
                    raw_refs.get("tree_id") or raw_refs.get("current_tree_id") or ""
                )
                evidence_query_id = str(raw_refs.get("query_id") or "")
                raw_refs = (
                    raw_refs.get("evidence_references")
                    or raw_refs.get("evidence_refs")
                    or raw_refs.get("references")
                    or ()
                )
            refs = _strings(raw_refs, "query evidence references")
            required = query.requirement is QueryRequirement.REQUIRED
            nomination_authority = (
                authority
                in {
                    EvidenceAuthority.PROMPT,
                    EvidenceAuthority.MODEL,
                }
                or (
                    authority is EvidenceAuthority.RETRIEVAL
                    and query.slot is not QueryEvidenceSlot.GRAPHRAG_NOMINATION
                )
                or any(
                    item.casefold().startswith(("prompt:", "model:")) for item in refs
                )
            )
            stale = bool(evidence_tree_id) and evidence_tree_id != query.scope.tree_id
            wrong_query = (
                bool(evidence_query_id) and evidence_query_id != query.query_id
            )
            satisfied = bool(refs) and not any(
                (
                    bundle_rejected,
                    bundle_stale,
                    nomination_authority,
                    stale,
                    wrong_query,
                )
            )
            disposition = (
                CoverageDisposition.SATISFIED
                if satisfied
                else CoverageDisposition.BLOCKED
                if required
                else CoverageDisposition.OPTIONAL_MISSING
            )
            reason_list: list[str] = []
            if not refs:
                reason_list.append(
                    "required_query_evidence_missing"
                    if required
                    else "optional_query_evidence_missing"
                )
            if bundle_rejected:
                reason_list.append("planning_evidence_bundle_not_ready")
            if bundle_stale or stale:
                reason_list.append("stale_or_cross_root_evidence")
            if wrong_query:
                reason_list.append("wrong_query_evidence")
            if nomination_authority:
                reason_list.append("prompt_or_model_evidence_not_authoritative")
            reasons = tuple(reason_list)
            if required and not satisfied:
                blockers.append(
                    f"query:{query.slot.value}:"
                    f"{reasons[0] if reasons else 'query_evidence_rejected'}"
                )
            query_slots.append(
                CoverageSlot(
                    slot=query.slot.value,
                    required=required,
                    disposition=disposition,
                    evidence_references=refs,
                    reason_codes=reasons,
                )
            )

        raw_claims = payload.get("claims") or ()
        claims = tuple(ProposalClaim.from_value(item) for item in _sequence(raw_claims))
        claim_slots: list[CoverageSlot] = []
        for claim in claims:
            required_authorities = _REQUIRED_CLAIM_AUTHORITIES[claim.claim_class]
            authoritative = required_authorities.intersection(
                claim.evidence_authorities
            )
            prompt_only = bool(claim.evidence_authorities) and set(
                claim.evidence_authorities
            ).issubset(
                {
                    EvidenceAuthority.PROMPT,
                    EvidenceAuthority.MODEL,
                    EvidenceAuthority.RETRIEVAL,
                }
            )
            satisfied = bool(
                authoritative and claim.evidence_references and not prompt_only
            )
            reasons: list[str] = []
            if prompt_only:
                reasons.append("prompt_model_or_retrieval_evidence_not_authoritative")
            if not authoritative:
                reasons.append("required_authority_missing")
            if not claim.evidence_references:
                reasons.append("evidence_reference_missing")
            disposition = (
                CoverageDisposition.SATISFIED
                if satisfied
                else CoverageDisposition.BLOCKED
            )
            if not satisfied:
                blockers.append(f"claim:{claim.claim_id}:{reasons[0]}")
            claim_slots.append(
                CoverageSlot(
                    slot=f"claim:{claim.claim_id}:{claim.claim_class.value}",
                    required=True,
                    disposition=disposition,
                    evidence_references=claim.evidence_references,
                    reason_codes=tuple(reasons),
                )
            )

        return PostProposalCoverageReceipt(
            query_plan_id=query_plan.plan_id,
            proposal_id=proposal_id,
            current_tree_id=query_plan.scope.tree_id,
            query_slots=tuple(query_slots),
            claim_slots=tuple(claim_slots),
            decision=(CoverageDecision.BLOCKED if blockers else CoverageDecision.READY),
            blockers=tuple(blockers),
        )

    recheck_coverage = rerun_coverage_after_proposal
    compile_coverage = rerun_coverage_after_proposal

    def _normalize_input(
        self, request: Any
    ) -> tuple[QueryInputKind, str, dict[str, Any], PlanRequestBudget]:
        if isinstance(request, PlanCreateRequest):
            return (
                QueryInputKind.CREATE,
                request.request_cid,
                request.to_dict(),
                request.budget,
            )
        if isinstance(request, PlanSteerRequest):
            return (
                QueryInputKind.STEER,
                request.request_cid,
                request.to_dict(),
                request.budget,
            )
        data = _as_mapping(request)
        if not data:
            raise QueryPlanningError(
                "diagnosis input must be a mapping or body-free typed record"
            )
        request_id = str(
            data.get("finding_cid")
            or data.get("issue_cid")
            or data.get("request_id")
            or data.get("content_id")
            or _content_id("diagnosis-input", data)
        )
        budget_value = data.get("budget")
        if isinstance(budget_value, PlanRequestBudget):
            budget = budget_value
        elif isinstance(budget_value, Mapping):
            allowed = {
                name
                for name in PlanRequestBudget.__dataclass_fields__
                if name != "SCHEMA"
            }
            budget = PlanRequestBudget(
                **{
                    name: value
                    for name, value in budget_value.items()
                    if name in allowed
                }
            )
        else:
            budget = PlanRequestBudget()
        return QueryInputKind.DIAGNOSIS, request_id, data, budget

    def _scope(
        self, kind: QueryInputKind, request: Any, data: Mapping[str, Any]
    ) -> QueryScope:
        roots = getattr(request, "roots", None)
        root_data = _as_mapping(roots)
        if not root_data:
            candidates = (
                data.get("roots"),
                data.get("authority_roots"),
                data.get("snapshot_roots"),
            )
            for candidate in candidates:
                root_data = _as_mapping(candidate)
                if root_data:
                    break
        repository_id = str(
            getattr(request, "repository_id", "")
            or root_data.get("repository_id")
            or data.get("repository_id")
            or ""
        )
        tree_id = str(
            root_data.get("dirty_worktree_root")
            or root_data.get("tree_id")
            or root_data.get("forest_id")
            or root_data.get("repository_root_cid")
            or data.get("tree_id")
            or data.get("current_root_id")
            or ""
        )
        objective_revision = str(
            root_data.get("task_source_revision")
            or data.get("objective_revision")
            or data.get("plan_revision")
            or data.get("finding_cid")
            or ""
        )
        policy_id = str(
            root_data.get("policy_root")
            or root_data.get("policy_id")
            or data.get("policy_id")
            or ""
        )
        required_roots = {
            "repository_id": repository_id,
            "tree_id": tree_id,
            "objective_revision": objective_revision,
            "policy_id": policy_id,
            "capability_catalog_root": str(
                root_data.get("capability_catalog_root")
                or data.get("capability_catalog_root")
                or ""
            ),
            "provider_catalog_root": str(
                root_data.get("provider_catalog_root")
                or data.get("provider_catalog_root")
                or ""
            ),
            "security_ir_root": str(
                root_data.get("security_ir_root") or data.get("security_ir_root") or ""
            ),
            "intent_ir_root": str(
                root_data.get("intent_ir_root") or data.get("intent_ir_root") or ""
            ),
        }
        missing = [name for name, value in required_roots.items() if not value]
        if missing:
            raise QueryPlanningError(
                f"{kind.value} input lacks exact roots: " + ", ".join(missing)
            )
        scope_paths = getattr(request, "scope_paths", ())
        return QueryScope(
            **required_roots,
            paths=_strings(
                (
                    *_sequence(scope_paths),
                    *_nested_values(data, "scope_paths", "paths"),
                ),
                "scope paths",
                paths=True,
            ),
            changed_paths=_strings(
                _nested_values(
                    data,
                    "changed_paths",
                    "affected_paths",
                    "target_paths",
                    "path",
                ),
                "changed paths",
                paths=True,
            ),
            symbols=_strings(
                _nested_values(
                    data, "symbols", "changed_symbols", "affected_symbols", "symbol"
                ),
                "symbols",
            ),
            contracts=_strings(
                _nested_values(data, "contracts", "contract_ids", "contract_refs"),
                "contracts",
            ),
            obligation_ids=_strings(
                _nested_values(data, "obligation_ids", "proof_obligations"),
                "obligation_ids",
            ),
            evidence_references=_strings(
                _nested_values(
                    data,
                    "evidence_refs",
                    "evidence_references",
                    "observation_refs",
                    "counterexample_refs",
                    "causal_slice_refs",
                ),
                "evidence references",
            ),
            open_frontiers=_strings(
                _nested_values(
                    data, "open_frontiers", "open_frontier_refs", "impact_frontiers"
                ),
                "open frontiers",
            ),
        )

    def _logic_family(self, data: Mapping[str, Any]) -> LogicFamily:
        candidates = _nested_values(
            data,
            "required_logic_families",
            "logic_families",
            "logic_family",
        )
        normalized: list[LogicFamily] = []
        for candidate in candidates:
            try:
                normalized.append(normalize_logic_family(candidate))
            except AnalysisOperationRegistryError:
                continue
        return (
            min(set(normalized), key=lambda item: item.value)
            if normalized
            else LogicFamily.TDFOL
        )

    def _requested_operations(
        self,
        kind: QueryInputKind,
        request: Any,
        data: Mapping[str, Any],
        suggestions: Any,
    ) -> tuple[
        tuple[AnalysisOperation, ...], tuple[tuple[AnalysisOperation, str], ...]
    ]:
        required_raw = (
            getattr(request, "required_analysis_operations", ())
            if kind is QueryInputKind.CREATE
            else _nested_values(data, "required_analysis_operations")
        )
        optional_raw = (
            getattr(request, "optional_analysis_operations", ())
            if kind is QueryInputKind.CREATE
            else _nested_values(data, "optional_analysis_operations")
        )
        suggestion_data = _as_mapping(suggestions)
        suggested_raw = _nested_values(
            suggestion_data,
            "operations",
            "analysis_operations",
            "optional_operations",
            "suggested_operations",
        )
        mandatory_operations = {rule[0] for rule in _MANDATORY_RULES.values()}

        def normalize(
            values: Iterable[Any], *, required: bool
        ) -> list[AnalysisOperation]:
            result: list[AnalysisOperation] = []
            for value in values:
                try:
                    operation = normalize_analysis_operation(value)
                    self.operation_registry.operation(operation)
                except AnalysisOperationRegistryError:
                    if required:
                        raise QueryPlanningError(
                            f"required analysis operation is unavailable: {value}"
                        )
                    continue
                if operation not in mandatory_operations and operation not in result:
                    result.append(operation)
            return sorted(result, key=lambda item: item.value)

        required = tuple(normalize(required_raw, required=True))
        optional: list[tuple[AnalysisOperation, str]] = [
            (item, "typed_request")
            for item in normalize(optional_raw, required=False)
            if item not in required
        ]
        optional.extend(
            (item, "model_nomination")
            for item in normalize(suggested_raw, required=False)
            if item not in required and all(item != known[0] for known in optional)
        )
        return required, tuple(optional)

    def _allocate_bounds(
        self, budget: PlanRequestBudget, count: int
    ) -> tuple[QueryBounds, ...]:
        if count <= 0:
            return ()

        def shares(total: int) -> list[int]:
            quotient, remainder = divmod(total, count)
            return [
                quotient + (1 if index < remainder else 0) for index in range(count)
            ]

        inputs = shares(budget.max_scan_bytes)
        outputs = shares(budget.max_scan_bytes)
        items = shares(budget.max_evidence_items)
        times = shares(budget.max_latency_ms)
        costs = shares(budget.max_cost_micros)
        return tuple(
            QueryBounds(
                max_input_bytes=inputs[index],
                max_output_bytes=outputs[index],
                max_items=items[index],
                timeout_ms=times[index],
                max_cost_micros=costs[index],
            )
            for index in range(count)
        )

    def _optional_bounds(self, budget: PlanRequestBudget, count: int) -> QueryBounds:
        return QueryBounds(
            max_input_bytes=budget.max_scan_bytes // max(1, count),
            max_output_bytes=budget.max_scan_bytes // max(1, count),
            max_items=budget.max_evidence_items // max(1, count),
            timeout_ms=budget.max_latency_ms // max(1, count),
            max_cost_micros=budget.max_cost_micros // max(1, count),
        )

    def _question(
        self, fixed: str, scope: QueryScope, concepts: tuple[str, ...]
    ) -> str:
        selectors = []
        if scope.changed_paths:
            selectors.append("changed paths=" + ",".join(scope.changed_paths[:16]))
        elif scope.paths:
            selectors.append("paths=" + ",".join(scope.paths[:16]))
        if scope.symbols:
            selectors.append("symbols=" + ",".join(scope.symbols[:16]))
        if scope.contracts:
            selectors.append("contracts=" + ",".join(scope.contracts[:16]))
        if concepts:
            selectors.append("concepts=" + ",".join(concepts[:16]))
        return fixed + (" Selectors: " + "; ".join(selectors) if selectors else "")

    def _build_query(
        self,
        *,
        slot: QueryEvidenceSlot,
        operation: AnalysisOperation,
        requirement: QueryRequirement,
        why: str,
        question: str,
        scope: QueryScope,
        logic_family: LogicFamily | None,
        bounds: QueryBounds,
        nomination_only: bool,
        source: str,
    ) -> ReasoningQuery:
        spec = self.operation_registry.operation(operation)
        bounds = QueryBounds(
            max_input_bytes=min(
                bounds.max_input_bytes,
                spec.bounds.max_question_bytes
                + spec.bounds.max_artifact_references * spec.bounds.max_reference_bytes,
            ),
            max_output_bytes=min(
                bounds.max_output_bytes,
                spec.bounds.max_evidence_references * spec.bounds.max_reference_bytes,
            ),
            max_items=min(bounds.max_items, spec.bounds.max_evidence_references),
            timeout_ms=min(bounds.timeout_ms, spec.bounds.timeout_ms),
            max_cost_micros=bounds.max_cost_micros,
        )
        strategies = list(self.strategy_registry.strategies_for_operation(operation))
        for property_class in property_class_for_operation(operation):
            strategy = self.strategy_registry.strategy(property_class)
            if all(item.strategy_id != strategy.strategy_id for item in strategies):
                strategies.append(strategy)
        if slot is QueryEvidenceSlot.SECURITY:
            security = self.strategy_registry.strategy(
                PropertyQuestionClass.PROTOCOL_SECURITY
            )
            if all(item.strategy_id != security.strategy_id for item in strategies):
                strategies.append(security)
        capabilities = set(spec.capability_requirements)
        for strategy in strategies:
            for method in strategy.required_methods():
                capabilities.update(method.provider_capabilities)
        preimage = {
            "operation": operation.value,
            "slot": slot.value,
            "scope": scope.to_dict(),
            "question_digest": _content_id("reasoning-question", question),
            "operation_spec_id": spec.spec_id,
            "strategy_ids": sorted(item.strategy_id for item in strategies),
            "capabilities": sorted(capabilities),
            "bounds": bounds.to_dict(),
        }
        cache = QueryCachePolicy(
            cacheable=spec.cache.cacheable,
            scope=spec.cache.scope.value,
            key_dimensions=spec.cache.key_dimensions,
            allow_stale=False,
            reuse_requires_equivalent_provenance=(
                spec.cache.reuse_requires_equivalent_provenance
            ),
            semantic_cache_key=_content_id("reasoning-query-cache-key", preimage),
        )
        disposition = (
            QueryFailureDisposition.BLOCK_CANDIDATE_GENERATION
            if requirement is QueryRequirement.REQUIRED
            else QueryFailureDisposition.RECORD_UNCERTAINTY_DEBT
        )
        return ReasoningQuery(
            slot=slot,
            operation=operation,
            requirement=requirement,
            why=why,
            question=question,
            scope=scope,
            logic_family=logic_family,
            operation_spec_id=spec.spec_id,
            strategy_ids=tuple(item.strategy_id for item in strategies),
            provider_capabilities=tuple(sorted(capabilities)),
            bounds=bounds,
            cache=cache,
            failure=QueryFailureSemantics(
                unavailable=disposition,
                timeout=disposition,
                malformed_result=disposition,
                stale_result=disposition,
                deterministic_fallback=spec.fallback.strategy,
            ),
            nomination_only=nomination_only,
            suggestion_source=source,
        )

    @staticmethod
    def _slot_for_operation(operation: AnalysisOperation) -> QueryEvidenceSlot:
        return {
            AnalysisOperation.SYMBOL_IMPACT: QueryEvidenceSlot.SYMBOL_IMPACT,
            AnalysisOperation.GRAPH_RAG_RETRIEVAL: QueryEvidenceSlot.GRAPHRAG_NOMINATION,
            AnalysisOperation.PREMISE_SELECTION: QueryEvidenceSlot.PREMISES,
            AnalysisOperation.CONTRADICTION_SEARCH: QueryEvidenceSlot.CONTRADICTIONS,
            AnalysisOperation.LOGIC_TRANSLATION: QueryEvidenceSlot.LOGIC_TRANSLATION,
            AnalysisOperation.PROOF_CANDIDATE_ANALYSIS: QueryEvidenceSlot.PROOF,
            AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS: QueryEvidenceSlot.COUNTEREXAMPLE,
        }[operation]


def compile_reasoning_query_plan(
    request: PlanCreateRequest | PlanSteerRequest | Mapping[str, Any] | Any,
    *,
    context: Mapping[str, Any] | None = None,
    model_suggestions: Mapping[str, Any] | Sequence[Any] | None = None,
    operation_registry: AnalysisOperationRegistry | None = None,
    strategy_registry: AnalysisStrategyRegistry | None = None,
) -> ReasoningQueryPlan:
    return PlanAnalysisQueryPlanner(
        operation_registry=operation_registry,
        strategy_registry=strategy_registry,
    ).compile(
        request,
        context=context,
        model_suggestions=model_suggestions,
    )


compile_plan_analysis_queries = compile_reasoning_query_plan
compile_analysis_query_plan = compile_reasoning_query_plan
compile_query_plan = compile_reasoning_query_plan


def rerun_evidence_coverage_after_proposal(
    query_plan: ReasoningQueryPlan,
    proposal: Mapping[str, Any] | Any,
    *,
    query_evidence: Mapping[str, Any] | None = None,
    evidence_bundle: PlanningEvidenceBundle | None = None,
) -> PostProposalCoverageReceipt:
    return PlanAnalysisQueryPlanner().rerun_coverage_after_proposal(
        query_plan,
        proposal,
        query_evidence=query_evidence,
        evidence_bundle=evidence_bundle,
    )


AnalysisQueryPlanner = PlanAnalysisQueryPlanner
DeterministicReasoningQueryPlanner = PlanAnalysisQueryPlanner
EvidenceCoverageReceipt = PostProposalCoverageReceipt
ReasoningQueryBudget = QueryBounds
RequiredEvidenceSlot = QueryEvidenceSlot
rerun_evidence_coverage = rerun_evidence_coverage_after_proposal


__all__ = [
    "REASONING_QUERY_PLAN_INTERFACE",
    "AnalysisQueryPlanner",
    "ClaimClass",
    "CoverageDecision",
    "CoverageDisposition",
    "CoverageSlot",
    "DeterministicReasoningQueryPlanner",
    "EvidenceAuthority",
    "EvidenceCoverageReceipt",
    "PlanAnalysisQueryPlanner",
    "PostProposalCoverageReceipt",
    "ProposalClaim",
    "QueryBounds",
    "QueryCachePolicy",
    "QueryEvidenceSlot",
    "QueryFailureDisposition",
    "QueryFailureSemantics",
    "QueryInputKind",
    "QueryPlanDecision",
    "QueryPlanningBudgetError",
    "QueryPlanningError",
    "QueryRequirement",
    "QueryScope",
    "ReasoningQuery",
    "ReasoningQueryBudget",
    "ReasoningQueryPlan",
    "RequiredEvidenceSlot",
    "UnsafeModelSuggestionError",
    "compile_analysis_query_plan",
    "compile_plan_analysis_queries",
    "compile_query_plan",
    "compile_reasoning_query_plan",
    "rerun_evidence_coverage",
    "rerun_evidence_coverage_after_proposal",
]
