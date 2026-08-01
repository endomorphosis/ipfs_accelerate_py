"""Deterministic analysis and formal-method strategy registry (PDR-012).

Interfaces: ``AnalysisStrategyRegistry@1``, ``AnalysisCapabilityReceipt@1``

Closed property/question classes map to bounded strategies with required
assurance, provider capability identifiers, input/output schemas, cache rules,
budgets, and fallback/abstention behaviour.  The registry is the policy
boundary that selects the least-cost sufficient deterministic method without
importing optional providers or inferring support from importability.

Authority contracts (fail-closed):

* retrieval and learned ranking remain ``nomination_only`` and can never
  satisfy a proof or completion slot;
* required unavailable methods produce a typed abstention;
* optional unavailable methods add uncertainty debt without inventing truth;
* discovery is cold/lazy — importing this module probes nothing;
* every routing result binds provider health, version, and config digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .analysis_operation_registry import (
    AnalysisOperation,
    CacheScope,
    normalize_analysis_operation,
)
from .analysis_transport import AnalysisProviderHealth


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

ANALYSIS_STRATEGY_REGISTRY_INTERFACE: Final[str] = "AnalysisStrategyRegistry@1"
ANALYSIS_STRATEGY_REGISTRY_VERSION: Final[int] = 1
ANALYSIS_STRATEGY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy@1"
)
ANALYSIS_STRATEGY_METHOD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy-method@1"
)
ANALYSIS_CAPABILITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-capability-receipt@1"
)
ANALYSIS_STRATEGY_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy-selection@1"
)
ANALYSIS_STRATEGY_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy-registry@1"
)

ANALYSIS_STRATEGY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy-input@1"
)
ANALYSIS_STRATEGY_OUTPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-strategy-output@1"
)

_MAX_TEXT_BYTES: Final[int] = 8 * 1024
_MAX_IDENTIFIER: Final[int] = 256
_MAX_METHODS_PER_STRATEGY: Final[int] = 64
_MAX_CAPABILITIES: Final[int] = 128
_MAX_BUDGET_VALUE: Final[int] = 10 * 60 * 1000


class AnalysisStrategyRegistryError(ValueError):
    """A strategy declaration, capability receipt, or routing request is invalid."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PropertyQuestionClass(str, Enum):
    """Closed property/question classes from the Planner/Doctor strategy table.

    These are analysis *routing* classes (what kind of question is being asked),
    not formal end-goal property kinds.
    """

    SYNTAX_STRUCTURE = "syntax_structure"
    CONTROL_DATA_FLOW = "control_data_flow"
    ALIASING_STATE = "aliasing_state"
    VALUES_SECURITY_FLOW = "values_security_flow"
    CONTRACTS = "contracts"
    HEAP_NATIVE_SAFETY = "heap_native_safety"
    RECURSIVE_INVARIANTS = "recursive_invariants"
    CONSTRAINT_SOLVING = "constraint_solving"
    STATE_CONCURRENCY = "state_concurrency"
    PROTOCOL_SECURITY = "protocol_security"
    BEHAVIORAL_TESTS = "behavioral_tests"
    RUNTIME_CONTRACTS = "runtime_contracts"
    REWRITE_SYNTHESIS = "rewrite_synthesis"
    SUPPLY_CHAIN = "supply_chain"
    RETRIEVAL = "retrieval"
    FORMAL_KERNELS = "formal_kernels"
    CRYPTOGRAPHIC_LINEAGE = "cryptographic_lineage"


# Stable aliases for callers that prefer the shorter name.
PropertyClass = PropertyQuestionClass


class StrategyMethod(str, Enum):
    """Closed set of deterministic analysis and formal-method techniques."""

    # Syntax / structure
    PYTHON_AST = "python_ast"
    TREE_SITTER = "tree_sitter"
    SYMBOL_INDEX = "symbol_index"
    CALL_GRAPH = "call_graph"
    # Control / data flow
    CFG = "cfg"
    SSA = "ssa"
    PDG = "pdg"
    REACHING_DEFINITIONS = "reaching_definitions"
    DEF_USE = "def_use"
    INTERPROCEDURAL_SUMMARY = "interprocedural_summary"
    # Aliasing / state
    POINTS_TO = "points_to"
    TYPESTATE = "typestate"
    OWNERSHIP = "ownership"
    ESCAPE_EFFECT = "escape_effect"
    # Values / security flow
    TAINT = "taint"
    INFORMATION_FLOW = "information_flow"
    PROVENANCE = "provenance"
    ABSTRACT_INTERPRETATION = "abstract_interpretation"
    # Contracts
    INTERFACE_DIFF = "interface_diff"
    PRE_POSTCONDITIONS = "pre_postconditions"
    INVARIANTS = "invariants"
    WEAKEST_PRECONDITION = "weakest_precondition"
    HOARE = "hoare"
    # Heap / native
    SEPARATION_LOGIC = "separation_logic"
    LIFETIME_OWNERSHIP = "lifetime_ownership"
    CBMC = "cbmc"
    KLEE = "klee"
    ANGR = "angr"
    # Recursive invariants / solvers
    CHC = "chc"
    DATALOG = "datalog"
    CEGAR = "cegar"
    CEGIS = "cegis"
    PDR = "pdr"
    # Constraint solving
    SAT = "sat"
    SMT = "smt"
    MAXSAT = "maxsat"
    Z3 = "z3"
    CVC5 = "cvc5"
    TACTICIAN_HAMMER = "tactician_hammer"
    # State / concurrency
    TLA_PLUS = "tla_plus"
    TLC = "tlc"
    APALACHE = "apalache"
    TEMPORAL_LOGIC = "temporal_logic"
    RACE_DEADLOCK = "race_deadlock"
    # Protocol / security
    TAMARIN = "tamarin"
    PROVERIF = "proverif"
    SECURITY_IR = "security_ir"
    AUTHORIZATION_DATALOG = "authorization_datalog"
    HYPERPROPERTY = "hyperproperty"
    # Behavioral tests
    PROPERTY_BASED_TEST = "property_based_test"
    FUZZ = "fuzz"
    CONCOLIC = "concolic"
    MUTATION_TEST = "mutation_test"
    DIFFERENTIAL_TEST = "differential_test"
    METAMORPHIC_TEST = "metamorphic_test"
    SANITIZER = "sanitizer"
    # Runtime
    TEMPORAL_MONITOR = "temporal_monitor"
    INVARIANT_MINING = "invariant_mining"
    TRACE_COMPARISON = "trace_comparison"
    DELTA_DEBUGGING = "delta_debugging"
    # Rewrite / synthesis
    REVIEWED_TEMPLATE = "reviewed_template"
    SEMANTIC_PATCH = "semantic_patch"
    EGRAPH = "egraph"
    ENUMERATIVE_SYNTHESIS = "enumerative_synthesis"
    # Supply chain
    SBOM = "sbom"
    LOCKFILE_REPRO = "lockfile_repro"
    OSV_SCAN = "osv_scan"
    # Retrieval (nomination only)
    BM25 = "bm25"
    VECTOR_RETRIEVAL = "vector_retrieval"
    EMBEDDING_RETRIEVAL = "embedding_retrieval"
    GRAPH_RAG = "graph_rag"
    KG_NEIGHBORHOOD = "kg_neighborhood"
    LEARNED_RANKING = "learned_ranking"
    # Formal kernels
    LEAN_KERNEL = "lean_kernel"
    ROCQ_KERNEL = "rocq_kernel"
    ISABELLE_KERNEL = "isabelle_kernel"
    # Cryptographic lineage
    CID_MERKLE = "cid_merkle"
    SIGNATURE = "signature"
    ZKP_ATTESTATION = "zkp_attestation"


class MethodRole(str, Enum):
    """Whether absence of a method abstains or only adds debt."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class AuthorityUse(str, Enum):
    """How strategy output may be used; never grants completion authority."""

    STRUCTURAL_EVIDENCE = "structural_evidence"
    OPEN_FRONTIER = "open_frontier"
    CANDIDATE = "candidate"
    CHECKED_FACT = "checked_fact"
    STATIC_FINDING = "static_finding"
    OBLIGATION = "obligation"
    BOUNDED_OBSERVATION = "bounded_observation"
    DIAGNOSTIC_CANDIDATE = "diagnostic_candidate"
    NOMINATION_ONLY = "nomination_only"
    KERNEL_ASSURANCE = "kernel_assurance"
    INTEGRITY_ATTESTATION = "integrity_attestation"
    SECURITY_EVIDENCE = "security_evidence"


class StrategyAssurance(str, Enum):
    """Assurance lattice for analysis strategy routing (body-free).

    Mirrors the formal-verification lattice names where useful, but remains
    independent so this module stays cold-importable without the proof package.
    """

    UNVERIFIED = "unverified"
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    BOUNDED_CHECKED = "bounded_checked"
    SOLVER_CHECKED = "solver_checked"
    KERNEL_VERIFIED = "kernel_verified"
    ATTESTED = "attested"

    @property
    def rank(self) -> int:
        return {
            StrategyAssurance.UNVERIFIED: 0,
            StrategyAssurance.OBSERVED: 1,
            StrategyAssurance.CANDIDATE: 2,
            StrategyAssurance.BOUNDED_CHECKED: 3,
            StrategyAssurance.SOLVER_CHECKED: 4,
            StrategyAssurance.KERNEL_VERIFIED: 5,
            StrategyAssurance.ATTESTED: 6,
        }[self]

    def satisfies(self, required: "StrategyAssurance | str") -> bool:
        other = (
            required
            if isinstance(required, StrategyAssurance)
            else StrategyAssurance(str(required).strip().lower())
        )
        return self.rank >= other.rank


class FallbackBehavior(str, Enum):
    """Closed fallback / abstention behaviours for a strategy or method."""

    DETERMINISTIC_LOCAL = "deterministic_local"
    ABSTAIN_REQUIRED = "abstain_required"
    DEBT_OPTIONAL = "debt_optional"
    OPEN_FRONTIER = "open_frontier"
    NOMINATION_ONLY = "nomination_only"
    FAIL_CLOSED = "fail_closed"


class SelectionOutcome(str, Enum):
    """Terminal outcome of one property-class routing decision."""

    SELECTED = "selected"
    PARTIAL = "partial"
    ABSTAIN = "abstain"
    DEBT_ONLY = "debt_only"


class CapabilityAdmission(str, Enum):
    """Admission outcome bound into a capability receipt."""

    AVAILABLE = "available"
    LAZY = "lazy"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    DEGRADED = "degraded"
    UNPROBED = "unprobed"


class UncertaintyDebtKind(str, Enum):
    """Typed debt when an optional method is unavailable."""

    OPTIONAL_METHOD_UNAVAILABLE = "optional_method_unavailable"
    PROVIDER_DEGRADED = "provider_degraded"
    BUDGET_TRUNCATION = "budget_truncation"
    OPEN_FRONTIER = "open_frontier"
    NOMINATION_GAP = "nomination_gap"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise AnalysisStrategyRegistryError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AnalysisStrategyRegistryError(f"{name} must not be empty")
    if "\x00" in result:
        raise AnalysisStrategyRegistryError(f"{name} must not contain NUL bytes")
    if len(result.encode("utf-8")) > maximum:
        raise AnalysisStrategyRegistryError(
            f"{name} exceeds {maximum} UTF-8 bytes"
        )
    return result


def _positive_int(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise AnalysisStrategyRegistryError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


def _nonneg_int(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise AnalysisStrategyRegistryError(
            f"{name} must be an integer from 0 through {maximum}"
        )
    return value


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(getattr(value, "value", value)).strip().lower())
    except (TypeError, ValueError) as exc:
        raise AnalysisStrategyRegistryError(
            f"unknown {name}: {value!r}"
        ) from exc


def _canonical(value: Any, *, name: str = "value", depth: int = 0) -> Any:
    if depth > 12:
        raise AnalysisStrategyRegistryError(f"{name} exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisStrategyRegistryError(f"{name} must be finite")
        return format(value, ".17g")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AnalysisStrategyRegistryError(f"{name} keys must be strings")
        return {
            key: _canonical(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical(item, name=name, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict(), name=name, depth=depth + 1)
    raise AnalysisStrategyRegistryError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def _content_id(namespace: str, value: Any) -> str:
    encoded = json.dumps(
        _canonical(value, name=namespace),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _string_tuple(
    values: Sequence[Any] | None,
    name: str,
    *,
    maximum_items: int = _MAX_CAPABILITIES,
    maximum_item_bytes: int = _MAX_IDENTIFIER,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise AnalysisStrategyRegistryError(f"{name} must be a sequence")
    items = tuple(
        sorted(
            {
                _text(item, name, maximum=maximum_item_bytes)
                for item in values
            }
        )
    )
    if len(items) > maximum_items:
        raise AnalysisStrategyRegistryError(
            f"{name} exceeds {maximum_items} entries"
        )
    return items


def normalize_property_class(value: Any) -> PropertyQuestionClass:
    if isinstance(value, PropertyQuestionClass):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    aliases = {
        "syntax": PropertyQuestionClass.SYNTAX_STRUCTURE,
        "structure": PropertyQuestionClass.SYNTAX_STRUCTURE,
        "symbols": PropertyQuestionClass.SYNTAX_STRUCTURE,
        "control_flow": PropertyQuestionClass.CONTROL_DATA_FLOW,
        "data_flow": PropertyQuestionClass.CONTROL_DATA_FLOW,
        "cfg": PropertyQuestionClass.CONTROL_DATA_FLOW,
        "aliasing": PropertyQuestionClass.ALIASING_STATE,
        "taint": PropertyQuestionClass.VALUES_SECURITY_FLOW,
        "security_flow": PropertyQuestionClass.VALUES_SECURITY_FLOW,
        "contract": PropertyQuestionClass.CONTRACTS,
        "heap": PropertyQuestionClass.HEAP_NATIVE_SAFETY,
        "native_safety": PropertyQuestionClass.HEAP_NATIVE_SAFETY,
        "invariants": PropertyQuestionClass.RECURSIVE_INVARIANTS,
        "constraints": PropertyQuestionClass.CONSTRAINT_SOLVING,
        "smt": PropertyQuestionClass.CONSTRAINT_SOLVING,
        "concurrency": PropertyQuestionClass.STATE_CONCURRENCY,
        "protocol": PropertyQuestionClass.PROTOCOL_SECURITY,
        "tests": PropertyQuestionClass.BEHAVIORAL_TESTS,
        "runtime": PropertyQuestionClass.RUNTIME_CONTRACTS,
        "synthesis": PropertyQuestionClass.REWRITE_SYNTHESIS,
        "supplychain": PropertyQuestionClass.SUPPLY_CHAIN,
        "retrieval_nomination": PropertyQuestionClass.RETRIEVAL,
        "kernel": PropertyQuestionClass.FORMAL_KERNELS,
        "formal_kernel": PropertyQuestionClass.FORMAL_KERNELS,
        "crypto_lineage": PropertyQuestionClass.CRYPTOGRAPHIC_LINEAGE,
        "lineage": PropertyQuestionClass.CRYPTOGRAPHIC_LINEAGE,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return PropertyQuestionClass(raw)
    except ValueError as exc:
        raise AnalysisStrategyRegistryError(
            f"unknown property class: {value!r}"
        ) from exc


def normalize_strategy_method(value: Any) -> StrategyMethod:
    if isinstance(value, StrategyMethod):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    aliases = {
        "ast": StrategyMethod.PYTHON_AST,
        "tree_sitter_parse": StrategyMethod.TREE_SITTER,
        "points_to_alias": StrategyMethod.POINTS_TO,
        "wp": StrategyMethod.WEAKEST_PRECONDITION,
        "weakest_preconditions": StrategyMethod.WEAKEST_PRECONDITION,
        "hoare_triple": StrategyMethod.HOARE,
        "bm25_retrieval": StrategyMethod.BM25,
        "vector": StrategyMethod.VECTOR_RETRIEVAL,
        "embedding": StrategyMethod.EMBEDDING_RETRIEVAL,
        "graphrag": StrategyMethod.GRAPH_RAG,
        "learned_rank": StrategyMethod.LEARNED_RANKING,
        "lean": StrategyMethod.LEAN_KERNEL,
        "coq": StrategyMethod.ROCQ_KERNEL,
        "rocq": StrategyMethod.ROCQ_KERNEL,
        "isabelle": StrategyMethod.ISABELLE_KERNEL,
        "zkp": StrategyMethod.ZKP_ATTESTATION,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return StrategyMethod(raw)
    except ValueError as exc:
        raise AnalysisStrategyRegistryError(
            f"unknown strategy method: {value!r}"
        ) from exc


def normalize_strategy_assurance(value: Any) -> StrategyAssurance:
    if isinstance(value, StrategyAssurance):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    aliases = {
        "none": StrategyAssurance.UNVERIFIED,
        "structural": StrategyAssurance.OBSERVED,
        "observation": StrategyAssurance.OBSERVED,
        "bounded": StrategyAssurance.BOUNDED_CHECKED,
        "solver": StrategyAssurance.SOLVER_CHECKED,
        "solver_verified": StrategyAssurance.SOLVER_CHECKED,
        "kernel": StrategyAssurance.KERNEL_VERIFIED,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return StrategyAssurance(raw)
    except ValueError as exc:
        raise AnalysisStrategyRegistryError(
            f"unknown strategy assurance: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Budget / cache / method / strategy declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyBudget:
    """Hard bounds for one strategy execution attempt."""

    timeout_ms: int = 30_000
    max_input_bytes: int = 64 * 1024
    max_output_bytes: int = 128 * 1024
    max_scope_paths: int = 4_096
    max_solver_fuel: int = 10_000
    max_memory_mib: int = 1_024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeout_ms",
            _positive_int(self.timeout_ms, "timeout_ms", _MAX_BUDGET_VALUE),
        )
        object.__setattr__(
            self,
            "max_input_bytes",
            _positive_int(self.max_input_bytes, "max_input_bytes", 16 * 1024 * 1024),
        )
        object.__setattr__(
            self,
            "max_output_bytes",
            _positive_int(
                self.max_output_bytes, "max_output_bytes", 16 * 1024 * 1024
            ),
        )
        object.__setattr__(
            self,
            "max_scope_paths",
            _positive_int(self.max_scope_paths, "max_scope_paths", 1_000_000),
        )
        object.__setattr__(
            self,
            "max_solver_fuel",
            _positive_int(self.max_solver_fuel, "max_solver_fuel", 10_000_000),
        )
        object.__setattr__(
            self,
            "max_memory_mib",
            _positive_int(self.max_memory_mib, "max_memory_mib", 65_536),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "timeout_ms": self.timeout_ms,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_scope_paths": self.max_scope_paths,
            "max_solver_fuel": self.max_solver_fuel,
            "max_memory_mib": self.max_memory_mib,
        }

    @classmethod
    def from_value(
        cls, value: "StrategyBudget | Mapping[str, Any] | None"
    ) -> "StrategyBudget":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisStrategyRegistryError("budget must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisStrategyRegistryError(
                "unknown budget fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class StrategyCacheRules:
    """Cache identity rules for strategy results (memoization only)."""

    cacheable: bool = True
    content_addressed: bool = True
    scope: CacheScope = CacheScope.TREE
    key_dimensions: tuple[str, ...] = (
        "strategy_id",
        "property_class",
        "method",
        "repository_id",
        "tree_id",
        "scope",
        "premises",
        "assumptions",
        "toolchain",
        "provider_capability",
        "policy_id",
        "required_assurance",
        "bounds",
    )
    allow_stale: bool = False
    rederive_assurance_on_hit: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _enum(self.scope, CacheScope, "scope"))
        dimensions = tuple(
            sorted(
                {
                    _text(item, "cache key dimension", maximum=128)
                    for item in self.key_dimensions
                }
            )
        )
        if self.cacheable and not dimensions:
            raise AnalysisStrategyRegistryError(
                "cacheable strategies require key dimensions"
            )
        if self.allow_stale:
            raise AnalysisStrategyRegistryError(
                "strategy cache cannot reuse stale evidence"
            )
        if not self.rederive_assurance_on_hit:
            raise AnalysisStrategyRegistryError(
                "strategy cache hits must re-derive assurance"
            )
        object.__setattr__(self, "key_dimensions", dimensions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cacheable": self.cacheable,
            "content_addressed": self.content_addressed,
            "scope": self.scope.value,
            "key_dimensions": list(self.key_dimensions),
            "allow_stale": False,
            "rederive_assurance_on_hit": True,
        }

    @classmethod
    def from_value(
        cls, value: "StrategyCacheRules | Mapping[str, Any] | None"
    ) -> "StrategyCacheRules":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisStrategyRegistryError("cache rules must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisStrategyRegistryError(
                "unknown cache rule fields: " + ", ".join(sorted(unknown))
            )
        fields = dict(value)
        if "key_dimensions" in fields:
            fields["key_dimensions"] = tuple(fields["key_dimensions"])
        return cls(**fields)


@dataclass(frozen=True)
class StrategyMethodBinding:
    """One method inside a property-class strategy, ordered by cost."""

    method: StrategyMethod
    role: MethodRole = MethodRole.OPTIONAL
    cost_rank: int = 100
    max_assurance: StrategyAssurance = StrategyAssurance.CANDIDATE
    authority_use: AuthorityUse = AuthorityUse.CANDIDATE
    provider_capabilities: tuple[str, ...] = ()
    analysis_operations: tuple[str, ...] = ()
    fallback: FallbackBehavior = FallbackBehavior.DEBT_OPTIONAL
    nomination_only: bool = False
    learned_ranking: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "method", normalize_strategy_method(self.method)
        )
        object.__setattr__(self, "role", _enum(self.role, MethodRole, "role"))
        object.__setattr__(
            self,
            "cost_rank",
            _nonneg_int(self.cost_rank, "cost_rank", 10_000),
        )
        object.__setattr__(
            self,
            "max_assurance",
            normalize_strategy_assurance(self.max_assurance),
        )
        object.__setattr__(
            self,
            "authority_use",
            _enum(self.authority_use, AuthorityUse, "authority_use"),
        )
        object.__setattr__(
            self,
            "provider_capabilities",
            _string_tuple(self.provider_capabilities, "provider_capabilities"),
        )
        operations = tuple(
            sorted(
                {
                    normalize_analysis_operation(item).value
                    for item in self.analysis_operations
                }
            )
        )
        object.__setattr__(self, "analysis_operations", operations)
        object.__setattr__(
            self,
            "fallback",
            _enum(self.fallback, FallbackBehavior, "fallback"),
        )
        if self.nomination_only or self.learned_ranking:
            object.__setattr__(self, "nomination_only", True)
            object.__setattr__(
                self, "authority_use", AuthorityUse.NOMINATION_ONLY
            )
            if self.max_assurance.rank > StrategyAssurance.CANDIDATE.rank:
                raise AnalysisStrategyRegistryError(
                    f"{self.method.value} is nomination-only and cannot claim "
                    f"{self.max_assurance.value} assurance"
                )
            object.__setattr__(
                self, "fallback", FallbackBehavior.NOMINATION_ONLY
            )
        if self.role is MethodRole.REQUIRED:
            if self.fallback not in {
                FallbackBehavior.ABSTAIN_REQUIRED,
                FallbackBehavior.FAIL_CLOSED,
                FallbackBehavior.DETERMINISTIC_LOCAL,
            }:
                object.__setattr__(
                    self, "fallback", FallbackBehavior.ABSTAIN_REQUIRED
                )
        elif self.fallback is FallbackBehavior.ABSTAIN_REQUIRED:
            raise AnalysisStrategyRegistryError(
                f"optional method {self.method.value} cannot use abstain_required"
            )

    @property
    def binding_id(self) -> str:
        return _content_id("analysis-strategy-method", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_STRATEGY_METHOD_SCHEMA,
            "method": self.method.value,
            "role": self.role.value,
            "cost_rank": self.cost_rank,
            "max_assurance": self.max_assurance.value,
            "authority_use": self.authority_use.value,
            "provider_capabilities": list(self.provider_capabilities),
            "analysis_operations": list(self.analysis_operations),
            "fallback": self.fallback.value,
            "nomination_only": self.nomination_only,
            "learned_ranking": self.learned_ranking,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"binding_id": self.binding_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyMethodBinding":
        if not isinstance(value, Mapping):
            raise AnalysisStrategyRegistryError(
                "method binding must be an object"
            )
        allowed = {
            "schema",
            "binding_id",
            "method",
            "role",
            "cost_rank",
            "max_assurance",
            "authority_use",
            "provider_capabilities",
            "analysis_operations",
            "fallback",
            "nomination_only",
            "learned_ranking",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisStrategyRegistryError(
                "unknown method binding fields: " + ", ".join(sorted(unknown))
            )
        result = cls(
            method=value.get("method", ""),
            role=value.get("role", MethodRole.OPTIONAL),
            cost_rank=value.get("cost_rank", 100),
            max_assurance=value.get("max_assurance", StrategyAssurance.CANDIDATE),
            authority_use=value.get("authority_use", AuthorityUse.CANDIDATE),
            provider_capabilities=tuple(
                value.get("provider_capabilities") or ()
            ),
            analysis_operations=tuple(value.get("analysis_operations") or ()),
            fallback=value.get("fallback", FallbackBehavior.DEBT_OPTIONAL),
            nomination_only=bool(value.get("nomination_only", False)),
            learned_ranking=bool(value.get("learned_ranking", False)),
        )
        claimed = value.get("binding_id")
        if claimed is not None and claimed != result.binding_id:
            raise AnalysisStrategyRegistryError(
                "method binding identity does not match"
            )
        return result


@dataclass(frozen=True)
class AnalysisStrategySpec:
    """Complete declaration for one property-class strategy."""

    property_class: PropertyQuestionClass
    methods: tuple[StrategyMethodBinding, ...]
    required_assurance: StrategyAssurance = StrategyAssurance.OBSERVED
    input_schema: str = ANALYSIS_STRATEGY_INPUT_SCHEMA
    output_schema: str = ANALYSIS_STRATEGY_OUTPUT_SCHEMA
    cache: StrategyCacheRules = field(default_factory=StrategyCacheRules)
    budget: StrategyBudget = field(default_factory=StrategyBudget)
    fallback: FallbackBehavior = FallbackBehavior.ABSTAIN_REQUIRED
    description: str = ""
    analysis_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_class",
            normalize_property_class(self.property_class),
        )
        if (
            isinstance(self.methods, (str, bytes, bytearray))
            or not isinstance(self.methods, Sequence)
            or not self.methods
        ):
            raise AnalysisStrategyRegistryError(
                "strategy requires a non-empty methods sequence"
            )
        bindings: list[StrategyMethodBinding] = []
        seen: set[StrategyMethod] = set()
        for raw in self.methods:
            if isinstance(raw, StrategyMethodBinding):
                binding = raw
            elif isinstance(raw, Mapping):
                binding = StrategyMethodBinding.from_dict(raw)
            else:
                raise AnalysisStrategyRegistryError(
                    "methods entries must be StrategyMethodBinding or objects"
                )
            if binding.method in seen:
                raise AnalysisStrategyRegistryError(
                    f"duplicate method in strategy: {binding.method.value}"
                )
            seen.add(binding.method)
            bindings.append(binding)
        if len(bindings) > _MAX_METHODS_PER_STRATEGY:
            raise AnalysisStrategyRegistryError(
                f"strategy exceeds {_MAX_METHODS_PER_STRATEGY} methods"
            )
        # Stable order: cost rank, then method id.
        bindings.sort(key=lambda item: (item.cost_rank, item.method.value))
        object.__setattr__(self, "methods", tuple(bindings))
        object.__setattr__(
            self,
            "required_assurance",
            normalize_strategy_assurance(self.required_assurance),
        )
        object.__setattr__(
            self,
            "input_schema",
            _text(self.input_schema, "input_schema", maximum=512),
        )
        object.__setattr__(
            self,
            "output_schema",
            _text(self.output_schema, "output_schema", maximum=512),
        )
        if not isinstance(self.cache, StrategyCacheRules):
            object.__setattr__(
                self, "cache", StrategyCacheRules.from_value(self.cache)
            )
        if not isinstance(self.budget, StrategyBudget):
            object.__setattr__(
                self, "budget", StrategyBudget.from_value(self.budget)
            )
        object.__setattr__(
            self,
            "fallback",
            _enum(self.fallback, FallbackBehavior, "fallback"),
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description", required=False, maximum=1024),
        )
        op_ids = {
            normalize_analysis_operation(item).value
            for item in self.analysis_operations
        }
        for binding in self.methods:
            op_ids.update(binding.analysis_operations)
        object.__setattr__(
            self,
            "analysis_operations",
            tuple(sorted(op_ids)),
        )
        # Retrieval strategies are always nomination-only end-to-end.
        if self.property_class is PropertyQuestionClass.RETRIEVAL:
            if any(not item.nomination_only for item in self.methods):
                raise AnalysisStrategyRegistryError(
                    "retrieval strategy methods must remain nomination_only"
                )
            if self.required_assurance.rank > StrategyAssurance.CANDIDATE.rank:
                raise AnalysisStrategyRegistryError(
                    "retrieval strategy cannot require above candidate assurance"
                )
            object.__setattr__(
                self, "fallback", FallbackBehavior.NOMINATION_ONLY
            )
        # Formal kernels require kernel assurance path.
        if self.property_class is PropertyQuestionClass.FORMAL_KERNELS:
            if self.required_assurance.rank < StrategyAssurance.KERNEL_VERIFIED.rank:
                object.__setattr__(
                    self,
                    "required_assurance",
                    StrategyAssurance.KERNEL_VERIFIED,
                )

    @property
    def strategy_id(self) -> str:
        return _content_id("analysis-strategy", self._payload())

    @property
    def property_class_id(self) -> str:
        return self.property_class.value

    def required_methods(self) -> tuple[StrategyMethodBinding, ...]:
        return tuple(
            item for item in self.methods if item.role is MethodRole.REQUIRED
        )

    def optional_methods(self) -> tuple[StrategyMethodBinding, ...]:
        return tuple(
            item for item in self.methods if item.role is MethodRole.OPTIONAL
        )

    def methods_meeting_assurance(
        self, required: StrategyAssurance | str
    ) -> tuple[StrategyMethodBinding, ...]:
        target = normalize_strategy_assurance(required)
        return tuple(
            item
            for item in self.methods
            if item.max_assurance.satisfies(target) and not item.nomination_only
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_STRATEGY_SCHEMA,
            "registry_version": ANALYSIS_STRATEGY_REGISTRY_VERSION,
            "property_class": self.property_class.value,
            "methods": [item.to_dict() for item in self.methods],
            "required_assurance": self.required_assurance.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cache": self.cache.to_dict(),
            "budget": self.budget.to_dict(),
            "fallback": self.fallback.value,
            "description": self.description,
            "analysis_operations": list(self.analysis_operations),
            "authority": {
                "repository_mutation": False,
                "validation_omission_selection": False,
                "candidate_promotion": False,
                "proof_authority": False,
                "completion_authority": False,
                "retrieval_is_nomination_only": (
                    self.property_class is PropertyQuestionClass.RETRIEVAL
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisStrategySpec":
        if not isinstance(value, Mapping):
            raise AnalysisStrategyRegistryError(
                "strategy declaration must be an object"
            )
        allowed = {
            "schema",
            "registry_version",
            "strategy_id",
            "property_class",
            "methods",
            "required_assurance",
            "input_schema",
            "output_schema",
            "cache",
            "budget",
            "fallback",
            "description",
            "analysis_operations",
            "authority",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisStrategyRegistryError(
                "unknown strategy declaration fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema", ANALYSIS_STRATEGY_SCHEMA) != ANALYSIS_STRATEGY_SCHEMA:
            raise AnalysisStrategyRegistryError("unsupported strategy schema")
        if value.get(
            "registry_version", ANALYSIS_STRATEGY_REGISTRY_VERSION
        ) != ANALYSIS_STRATEGY_REGISTRY_VERSION:
            raise AnalysisStrategyRegistryError(
                "unsupported strategy registry version"
            )
        authority = value.get("authority")
        if isinstance(authority, Mapping):
            for forbidden in (
                "repository_mutation",
                "validation_omission_selection",
                "candidate_promotion",
                "proof_authority",
                "completion_authority",
            ):
                if authority.get(forbidden, False) is not False:
                    raise AnalysisStrategyRegistryError(
                        f"strategy declaration claims forbidden authority: {forbidden}"
                    )
        methods_raw = value.get("methods") or ()
        methods = tuple(
            item
            if isinstance(item, StrategyMethodBinding)
            else StrategyMethodBinding.from_dict(item)
            for item in methods_raw
        )
        result = cls(
            property_class=value.get("property_class", ""),
            methods=methods,
            required_assurance=value.get(
                "required_assurance", StrategyAssurance.OBSERVED
            ),
            input_schema=value.get("input_schema", ANALYSIS_STRATEGY_INPUT_SCHEMA),
            output_schema=value.get(
                "output_schema", ANALYSIS_STRATEGY_OUTPUT_SCHEMA
            ),
            cache=StrategyCacheRules.from_value(value.get("cache")),
            budget=StrategyBudget.from_value(value.get("budget")),
            fallback=value.get("fallback", FallbackBehavior.ABSTAIN_REQUIRED),
            description=value.get("description") or "",
            analysis_operations=tuple(value.get("analysis_operations") or ()),
        )
        claimed = value.get("strategy_id")
        if claimed is not None and claimed != result.strategy_id:
            raise AnalysisStrategyRegistryError(
                "strategy declaration identity does not match"
            )
        return result


# ---------------------------------------------------------------------------
# Capability receipts and selection outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisCapabilityReceipt:
    """Provider health/version/config binding for one strategy method result.

    Interface: ``AnalysisCapabilityReceipt@1``
    """

    capability_id: str
    provider_id: str
    method: StrategyMethod
    property_class: PropertyQuestionClass
    admission: CapabilityAdmission = CapabilityAdmission.UNPROBED
    health: AnalysisProviderHealth = AnalysisProviderHealth.LAZY
    provider_version: str = "unknown"
    config_digest: str = ""
    capability_revision: str = ""
    reason_code: str = ""
    nomination_only: bool = False
    strategy_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            _text(self.capability_id, "capability_id", maximum=_MAX_IDENTIFIER),
        )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", maximum=_MAX_IDENTIFIER),
        )
        object.__setattr__(
            self, "method", normalize_strategy_method(self.method)
        )
        object.__setattr__(
            self,
            "property_class",
            normalize_property_class(self.property_class),
        )
        object.__setattr__(
            self,
            "admission",
            _enum(self.admission, CapabilityAdmission, "admission"),
        )
        object.__setattr__(
            self,
            "health",
            _enum(self.health, AnalysisProviderHealth, "health"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version,
                "provider_version",
                required=False,
                maximum=_MAX_IDENTIFIER,
            )
            or "unknown",
        )
        object.__setattr__(
            self,
            "config_digest",
            _text(
                self.config_digest,
                "config_digest",
                required=False,
                maximum=512,
            ),
        )
        object.__setattr__(
            self,
            "capability_revision",
            _text(
                self.capability_revision,
                "capability_revision",
                required=False,
                maximum=512,
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(
                self.reason_code, "reason_code", required=False, maximum=256
            ),
        )
        object.__setattr__(
            self,
            "strategy_id",
            _text(
                self.strategy_id, "strategy_id", required=False, maximum=512
            ),
        )
        if not isinstance(self.details, Mapping):
            raise AnalysisStrategyRegistryError("details must be an object")
        object.__setattr__(
            self, "details", MappingProxyType(dict(self.details))
        )
        # Usable admissions must not claim health that contradicts usability.
        if (
            self.admission is CapabilityAdmission.AVAILABLE
            and not self.health.usable
        ):
            raise AnalysisStrategyRegistryError(
                "available capability receipt requires usable health"
            )

    @property
    def receipt_id(self) -> str:
        return _content_id("analysis-capability-receipt", self._payload())

    @property
    def usable(self) -> bool:
        return self.admission in {
            CapabilityAdmission.AVAILABLE,
            CapabilityAdmission.LAZY,
        } and self.health.usable

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CAPABILITY_RECEIPT_SCHEMA,
            "interface": "AnalysisCapabilityReceipt@1",
            "capability_id": self.capability_id,
            "provider_id": self.provider_id,
            "method": self.method.value,
            "property_class": self.property_class.value,
            "admission": self.admission.value,
            "health": self.health.value,
            "provider_version": self.provider_version,
            "config_digest": self.config_digest,
            "capability_revision": self.capability_revision,
            "reason_code": self.reason_code,
            "nomination_only": self.nomination_only,
            "strategy_id": self.strategy_id,
            "details": dict(self.details),
            "authority": {
                "completion_authority": False,
                "proof_authority": False,
                "candidate_promotion": False,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"receipt_id": self.receipt_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisCapabilityReceipt":
        if not isinstance(value, Mapping):
            raise AnalysisStrategyRegistryError(
                "capability receipt must be an object"
            )
        allowed = {
            "schema",
            "interface",
            "receipt_id",
            "capability_id",
            "provider_id",
            "method",
            "property_class",
            "admission",
            "health",
            "provider_version",
            "config_digest",
            "capability_revision",
            "reason_code",
            "nomination_only",
            "strategy_id",
            "details",
            "authority",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisStrategyRegistryError(
                "unknown capability receipt fields: "
                + ", ".join(sorted(unknown))
            )
        if (
            value.get("schema", ANALYSIS_CAPABILITY_RECEIPT_SCHEMA)
            != ANALYSIS_CAPABILITY_RECEIPT_SCHEMA
        ):
            raise AnalysisStrategyRegistryError(
                "unsupported capability receipt schema"
            )
        authority = value.get("authority")
        if isinstance(authority, Mapping):
            for forbidden in (
                "completion_authority",
                "proof_authority",
                "candidate_promotion",
            ):
                if authority.get(forbidden, False) is not False:
                    raise AnalysisStrategyRegistryError(
                        f"capability receipt claims forbidden authority: {forbidden}"
                    )
        result = cls(
            capability_id=value.get("capability_id", ""),
            provider_id=value.get("provider_id", ""),
            method=value.get("method", ""),
            property_class=value.get("property_class", ""),
            admission=value.get("admission", CapabilityAdmission.UNPROBED),
            health=value.get("health", AnalysisProviderHealth.LAZY),
            provider_version=value.get("provider_version", "unknown"),
            config_digest=value.get("config_digest") or "",
            capability_revision=value.get("capability_revision") or "",
            reason_code=value.get("reason_code") or "",
            nomination_only=bool(value.get("nomination_only", False)),
            strategy_id=value.get("strategy_id") or "",
            details=dict(value.get("details") or {}),
        )
        claimed = value.get("receipt_id")
        if claimed is not None and claimed != result.receipt_id:
            raise AnalysisStrategyRegistryError(
                "capability receipt identity does not match"
            )
        return result


@dataclass(frozen=True)
class UncertaintyDebt:
    """Explicit uncertainty when an optional path is missing or degraded."""

    kind: UncertaintyDebtKind
    method: StrategyMethod | None
    property_class: PropertyQuestionClass
    message: str
    reason_code: str = ""
    receipt: AnalysisCapabilityReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, UncertaintyDebtKind, "kind")
        )
        if self.method is not None:
            object.__setattr__(
                self, "method", normalize_strategy_method(self.method)
            )
        object.__setattr__(
            self,
            "property_class",
            normalize_property_class(self.property_class),
        )
        object.__setattr__(
            self,
            "message",
            _text(self.message, "message", maximum=1024),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(
                self.reason_code, "reason_code", required=False, maximum=256
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "method": self.method.value if self.method is not None else "",
            "property_class": self.property_class.value,
            "message": self.message,
            "reason_code": self.reason_code,
            "receipt_id": (
                self.receipt.receipt_id if self.receipt is not None else ""
            ),
        }


@dataclass(frozen=True)
class StrategySelection:
    """Least-cost sufficient routing decision for one property class."""

    property_class: PropertyQuestionClass
    strategy_id: str
    outcome: SelectionOutcome
    selected_methods: tuple[StrategyMethodBinding, ...] = ()
    receipts: tuple[AnalysisCapabilityReceipt, ...] = ()
    debt: tuple[UncertaintyDebt, ...] = ()
    required_assurance: StrategyAssurance = StrategyAssurance.OBSERVED
    achieved_assurance: StrategyAssurance = StrategyAssurance.UNVERIFIED
    abstention_reason: str = ""
    nomination_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_class",
            normalize_property_class(self.property_class),
        )
        object.__setattr__(
            self,
            "strategy_id",
            _text(self.strategy_id, "strategy_id", maximum=512),
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, SelectionOutcome, "outcome")
        )
        object.__setattr__(
            self,
            "required_assurance",
            normalize_strategy_assurance(self.required_assurance),
        )
        object.__setattr__(
            self,
            "achieved_assurance",
            normalize_strategy_assurance(self.achieved_assurance),
        )
        object.__setattr__(
            self,
            "abstention_reason",
            _text(
                self.abstention_reason,
                "abstention_reason",
                required=False,
                maximum=1024,
            ),
        )
        object.__setattr__(self, "selected_methods", tuple(self.selected_methods))
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(self, "debt", tuple(self.debt))
        if self.outcome is SelectionOutcome.ABSTAIN and not self.abstention_reason:
            raise AnalysisStrategyRegistryError(
                "abstain selection requires abstention_reason"
            )
        if self.outcome is SelectionOutcome.SELECTED and not self.selected_methods:
            raise AnalysisStrategyRegistryError(
                "selected outcome requires at least one method"
            )

    @property
    def selection_id(self) -> str:
        return _content_id("analysis-strategy-selection", self._payload())

    @property
    def abstained(self) -> bool:
        return self.outcome is SelectionOutcome.ABSTAIN

    @property
    def has_debt(self) -> bool:
        return bool(self.debt)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_STRATEGY_SELECTION_SCHEMA,
            "property_class": self.property_class.value,
            "strategy_id": self.strategy_id,
            "outcome": self.outcome.value,
            "selected_methods": [item.to_dict() for item in self.selected_methods],
            "receipts": [item.to_dict() for item in self.receipts],
            "debt": [item.to_dict() for item in self.debt],
            "required_assurance": self.required_assurance.value,
            "achieved_assurance": self.achieved_assurance.value,
            "abstention_reason": self.abstention_reason,
            "nomination_only": self.nomination_only,
            "authority": {
                "completion_authority": False,
                "proof_authority": False,
                "retrieval_is_nomination_only": self.nomination_only
                or self.property_class is PropertyQuestionClass.RETRIEVAL,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"selection_id": self.selection_id, **self._payload()}


# ---------------------------------------------------------------------------
# Lazy provider binding (cold discovery)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LazyProviderAdapter:
    """Declaration-only provider adapter; factory runs only on explicit probe."""

    provider_id: str
    capability_ids: tuple[str, ...]
    methods: tuple[StrategyMethod, ...]
    provider_version: str = "unknown"
    config_digest: str = ""
    health: AnalysisProviderHealth = AnalysisProviderHealth.LAZY
    factory: Callable[[], Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", maximum=_MAX_IDENTIFIER),
        )
        object.__setattr__(
            self,
            "capability_ids",
            _string_tuple(self.capability_ids, "capability_ids"),
        )
        if not self.capability_ids:
            raise AnalysisStrategyRegistryError(
                "lazy provider adapter requires capability_ids"
            )
        methods = tuple(
            sorted(
                {normalize_strategy_method(item) for item in self.methods},
                key=lambda item: item.value,
            )
        )
        if not methods:
            raise AnalysisStrategyRegistryError(
                "lazy provider adapter requires methods"
            )
        object.__setattr__(self, "methods", methods)
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version,
                "provider_version",
                required=False,
                maximum=_MAX_IDENTIFIER,
            )
            or "unknown",
        )
        object.__setattr__(
            self,
            "config_digest",
            _text(
                self.config_digest,
                "config_digest",
                required=False,
                maximum=512,
            ),
        )
        object.__setattr__(
            self,
            "health",
            _enum(self.health, AnalysisProviderHealth, "health"),
        )
        if self.factory is not None and not callable(self.factory):
            raise AnalysisStrategyRegistryError(
                "provider factory must be callable when supplied"
            )

    def declaration_receipt(
        self,
        *,
        method: StrategyMethod,
        property_class: PropertyQuestionClass,
        strategy_id: str = "",
        nomination_only: bool = False,
    ) -> AnalysisCapabilityReceipt:
        """Cold receipt: LAZY admission without invoking the factory."""

        capability_id = self.capability_ids[0]
        for candidate in self.capability_ids:
            # Prefer an id that mentions the method when present.
            if method.value in candidate:
                capability_id = candidate
                break
        return AnalysisCapabilityReceipt(
            capability_id=capability_id,
            provider_id=self.provider_id,
            method=method,
            property_class=property_class,
            admission=CapabilityAdmission.LAZY,
            health=self.health,
            provider_version=self.provider_version,
            config_digest=self.config_digest,
            capability_revision=_content_id(
                "lazy-provider",
                {
                    "provider_id": self.provider_id,
                    "capability_ids": list(self.capability_ids),
                    "version": self.provider_version,
                    "config": self.config_digest,
                },
            ),
            reason_code="cold_lazy_declaration",
            nomination_only=nomination_only,
            strategy_id=strategy_id,
            details={"probed": False, "import_inferred": False},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "capability_ids": list(self.capability_ids),
            "methods": [item.value for item in self.methods],
            "provider_version": self.provider_version,
            "config_digest": self.config_digest,
            "health": self.health.value,
            "has_factory": self.factory is not None,
            "probed_on_import": False,
        }


# ---------------------------------------------------------------------------
# Default strategy portfolio (plan §5.1)
# ---------------------------------------------------------------------------


def _method(
    method: StrategyMethod,
    *,
    role: MethodRole = MethodRole.OPTIONAL,
    cost: int,
    assurance: StrategyAssurance,
    authority: AuthorityUse,
    capabilities: Sequence[str] = (),
    operations: Sequence[str] = (),
    nomination_only: bool = False,
    learned_ranking: bool = False,
    fallback: FallbackBehavior | None = None,
) -> StrategyMethodBinding:
    if fallback is None:
        if nomination_only or learned_ranking:
            fallback = FallbackBehavior.NOMINATION_ONLY
        elif role is MethodRole.REQUIRED:
            fallback = FallbackBehavior.ABSTAIN_REQUIRED
        else:
            fallback = FallbackBehavior.DEBT_OPTIONAL
    return StrategyMethodBinding(
        method=method,
        role=role,
        cost_rank=cost,
        max_assurance=assurance,
        authority_use=authority,
        provider_capabilities=tuple(capabilities),
        analysis_operations=tuple(operations),
        fallback=fallback,
        nomination_only=nomination_only,
        learned_ranking=learned_ranking,
    )


def default_strategy_specs() -> tuple[AnalysisStrategySpec, ...]:
    """Return the closed property-class → strategy portfolio."""

    symbol_ops = (AnalysisOperation.SYMBOL_IMPACT.value,)
    graph_ops = (AnalysisOperation.GRAPH_RAG_RETRIEVAL.value,)
    proof_ops = (
        AnalysisOperation.PROOF_CANDIDATE_ANALYSIS.value,
        AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS.value,
    )
    premise_ops = (AnalysisOperation.PREMISE_SELECTION.value,)
    logic_ops = (AnalysisOperation.LOGIC_TRANSLATION.value,)

    return (
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.SYNTAX_STRUCTURE,
            description=(
                "Syntax, symbols, imports, and calls via AST/Tree-sitter indexes."
            ),
            required_assurance=StrategyAssurance.OBSERVED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            analysis_operations=symbol_ops,
            methods=(
                _method(
                    StrategyMethod.PYTHON_AST,
                    role=MethodRole.REQUIRED,
                    cost=10,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("ast_index_read", "python_ast"),
                    operations=symbol_ops,
                ),
                _method(
                    StrategyMethod.TREE_SITTER,
                    role=MethodRole.OPTIONAL,
                    cost=20,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("tree_sitter", "polyglot_ast"),
                    operations=symbol_ops,
                ),
                _method(
                    StrategyMethod.SYMBOL_INDEX,
                    role=MethodRole.REQUIRED,
                    cost=15,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("symbol_impact", "ast_index_read"),
                    operations=symbol_ops,
                ),
                _method(
                    StrategyMethod.CALL_GRAPH,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("call_graph",),
                    operations=symbol_ops,
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.CONTROL_DATA_FLOW,
            description="CFG, SSA, PDG, reaching definitions, def-use, summaries.",
            required_assurance=StrategyAssurance.OBSERVED,
            fallback=FallbackBehavior.OPEN_FRONTIER,
            analysis_operations=symbol_ops,
            methods=(
                _method(
                    StrategyMethod.CFG,
                    role=MethodRole.REQUIRED,
                    cost=20,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("cfg",),
                ),
                _method(
                    StrategyMethod.SSA,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("ssa",),
                ),
                _method(
                    StrategyMethod.PDG,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("pdg",),
                ),
                _method(
                    StrategyMethod.REACHING_DEFINITIONS,
                    role=MethodRole.OPTIONAL,
                    cost=35,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("reaching_definitions",),
                ),
                _method(
                    StrategyMethod.DEF_USE,
                    role=MethodRole.OPTIONAL,
                    cost=35,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("def_use",),
                ),
                _method(
                    StrategyMethod.INTERPROCEDURAL_SUMMARY,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.OPEN_FRONTIER,
                    capabilities=("interprocedural_summary",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.ALIASING_STATE,
            description="Points-to, typestate, ownership, escape/effect analysis.",
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.POINTS_TO,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("points_to",),
                ),
                _method(
                    StrategyMethod.TYPESTATE,
                    role=MethodRole.OPTIONAL,
                    cost=45,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.CHECKED_FACT,
                    capabilities=("typestate",),
                ),
                _method(
                    StrategyMethod.OWNERSHIP,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("ownership",),
                ),
                _method(
                    StrategyMethod.ESCAPE_EFFECT,
                    role=MethodRole.OPTIONAL,
                    cost=55,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("escape_effect",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.VALUES_SECURITY_FLOW,
            description="Taint, information-flow, provenance, abstract domains.",
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.TAINT,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("taint",),
                ),
                _method(
                    StrategyMethod.INFORMATION_FLOW,
                    role=MethodRole.OPTIONAL,
                    cost=55,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("information_flow",),
                ),
                _method(
                    StrategyMethod.PROVENANCE,
                    role=MethodRole.OPTIONAL,
                    cost=35,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.STRUCTURAL_EVIDENCE,
                    capabilities=("value_provenance",),
                ),
                _method(
                    StrategyMethod.ABSTRACT_INTERPRETATION,
                    role=MethodRole.OPTIONAL,
                    cost=60,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("abstract_interpretation",),
                ),
                _method(
                    StrategyMethod.HYPERPROPERTY,
                    role=MethodRole.OPTIONAL,
                    cost=90,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("hyperproperty",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.CONTRACTS,
            description=(
                "Interface/schema diffs, pre/post, invariants, WP, Hoare triples."
            ),
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            analysis_operations=logic_ops + proof_ops,
            methods=(
                _method(
                    StrategyMethod.INTERFACE_DIFF,
                    role=MethodRole.REQUIRED,
                    cost=15,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("interface_diff", "contract_analysis"),
                ),
                _method(
                    StrategyMethod.PRE_POSTCONDITIONS,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("pre_postconditions",),
                    operations=logic_ops,
                ),
                _method(
                    StrategyMethod.INVARIANTS,
                    role=MethodRole.OPTIONAL,
                    cost=45,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("invariants",),
                ),
                _method(
                    StrategyMethod.WEAKEST_PRECONDITION,
                    role=MethodRole.OPTIONAL,
                    cost=60,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("weakest_precondition",),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.HOARE,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("hoare",),
                    operations=proof_ops,
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.HEAP_NATIVE_SAFETY,
            description="Separation logic, lifetime/ownership, CBMC/KLEE/angr.",
            required_assurance=StrategyAssurance.BOUNDED_CHECKED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            methods=(
                _method(
                    StrategyMethod.SEPARATION_LOGIC,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("separation_logic",),
                ),
                _method(
                    StrategyMethod.LIFETIME_OWNERSHIP,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("lifetime_ownership",),
                ),
                _method(
                    StrategyMethod.CBMC,
                    role=MethodRole.OPTIONAL,
                    cost=80,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("cbmc",),
                ),
                _method(
                    StrategyMethod.KLEE,
                    role=MethodRole.OPTIONAL,
                    cost=85,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("klee",),
                ),
                _method(
                    StrategyMethod.ANGR,
                    role=MethodRole.OPTIONAL,
                    cost=90,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("angr",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.RECURSIVE_INVARIANTS,
            description="Abstract interpretation, CHC, Datalog, CEGAR/PDR.",
            required_assurance=StrategyAssurance.BOUNDED_CHECKED,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.ABSTRACT_INTERPRETATION,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("abstract_interpretation",),
                ),
                _method(
                    StrategyMethod.CHC,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("chc",),
                ),
                _method(
                    StrategyMethod.DATALOG,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CHECKED_FACT,
                    capabilities=("datalog",),
                ),
                _method(
                    StrategyMethod.CEGAR,
                    role=MethodRole.OPTIONAL,
                    cost=80,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("cegar",),
                ),
                _method(
                    StrategyMethod.PDR,
                    role=MethodRole.OPTIONAL,
                    cost=75,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("pdr",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.CONSTRAINT_SOLVING,
            description="SAT/SMT/MaxSAT, Z3/CVC5, Tactician/Hammer, CEGIS.",
            required_assurance=StrategyAssurance.SOLVER_CHECKED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            analysis_operations=proof_ops + premise_ops,
            methods=(
                _method(
                    StrategyMethod.SAT,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("sat",),
                ),
                _method(
                    StrategyMethod.SMT,
                    role=MethodRole.REQUIRED,
                    cost=50,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("smt", "logic_family_routing"),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.Z3,
                    role=MethodRole.OPTIONAL,
                    cost=55,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("z3",),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.CVC5,
                    role=MethodRole.OPTIONAL,
                    cost=55,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("cvc5",),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.MAXSAT,
                    role=MethodRole.OPTIONAL,
                    cost=60,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("maxsat",),
                ),
                _method(
                    StrategyMethod.TACTICIAN_HAMMER,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.CANDIDATE,
                    capabilities=("tactician_hammer",),
                    operations=premise_ops + proof_ops,
                ),
                _method(
                    StrategyMethod.CEGIS,
                    role=MethodRole.OPTIONAL,
                    cost=80,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("cegis",),
                ),
            ),
            budget=StrategyBudget(timeout_ms=60_000, max_solver_fuel=50_000),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.STATE_CONCURRENCY,
            description="TLA+/TLC/Apalache, temporal logic, race/deadlock.",
            required_assurance=StrategyAssurance.BOUNDED_CHECKED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            methods=(
                _method(
                    StrategyMethod.TEMPORAL_LOGIC,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("temporal_logic",),
                ),
                _method(
                    StrategyMethod.TLA_PLUS,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("tla_plus",),
                ),
                _method(
                    StrategyMethod.TLC,
                    role=MethodRole.OPTIONAL,
                    cost=75,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("tlc",),
                ),
                _method(
                    StrategyMethod.APALACHE,
                    role=MethodRole.OPTIONAL,
                    cost=80,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("apalache",),
                ),
                _method(
                    StrategyMethod.RACE_DEADLOCK,
                    role=MethodRole.OPTIONAL,
                    cost=45,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.STATIC_FINDING,
                    capabilities=("race_deadlock",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.PROTOCOL_SECURITY,
            description=(
                "Tamarin/ProVerif, SecurityIR, authorization Datalog, hyperproperties."
            ),
            required_assurance=StrategyAssurance.SOLVER_CHECKED,
            fallback=FallbackBehavior.FAIL_CLOSED,
            methods=(
                _method(
                    StrategyMethod.SECURITY_IR,
                    role=MethodRole.REQUIRED,
                    cost=20,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("security_ir",),
                ),
                _method(
                    StrategyMethod.AUTHORIZATION_DATALOG,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("authorization_datalog",),
                ),
                _method(
                    StrategyMethod.TAMARIN,
                    role=MethodRole.OPTIONAL,
                    cost=90,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("tamarin",),
                ),
                _method(
                    StrategyMethod.PROVERIF,
                    role=MethodRole.OPTIONAL,
                    cost=90,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("proverif",),
                ),
                _method(
                    StrategyMethod.HYPERPROPERTY,
                    role=MethodRole.OPTIONAL,
                    cost=95,
                    assurance=StrategyAssurance.SOLVER_CHECKED,
                    authority=AuthorityUse.OBLIGATION,
                    capabilities=("hyperproperty",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.BEHAVIORAL_TESTS,
            description=(
                "Property-based, fuzz, concolic, mutation, differential, "
                "metamorphic; never theorem substitution."
            ),
            required_assurance=StrategyAssurance.BOUNDED_CHECKED,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.PROPERTY_BASED_TEST,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("property_based_test",),
                ),
                _method(
                    StrategyMethod.FUZZ,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("fuzz",),
                ),
                _method(
                    StrategyMethod.CONCOLIC,
                    role=MethodRole.OPTIONAL,
                    cost=60,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("concolic",),
                ),
                _method(
                    StrategyMethod.MUTATION_TEST,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("mutation_test",),
                ),
                _method(
                    StrategyMethod.DIFFERENTIAL_TEST,
                    role=MethodRole.OPTIONAL,
                    cost=45,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("differential_test",),
                ),
                _method(
                    StrategyMethod.METAMORPHIC_TEST,
                    role=MethodRole.OPTIONAL,
                    cost=45,
                    assurance=StrategyAssurance.BOUNDED_CHECKED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("metamorphic_test",),
                ),
                _method(
                    StrategyMethod.SANITIZER,
                    role=MethodRole.OPTIONAL,
                    cost=25,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.BOUNDED_OBSERVATION,
                    capabilities=("sanitizer",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.RUNTIME_CONTRACTS,
            description=(
                "Temporal monitors, invariant mining, trace comparison, delta debugging."
            ),
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.TEMPORAL_MONITOR,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("temporal_monitor",),
                ),
                _method(
                    StrategyMethod.INVARIANT_MINING,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("invariant_mining",),
                ),
                _method(
                    StrategyMethod.TRACE_COMPARISON,
                    role=MethodRole.OPTIONAL,
                    cost=35,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("trace_comparison",),
                ),
                _method(
                    StrategyMethod.DELTA_DEBUGGING,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("delta_debugging",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.REWRITE_SYNTHESIS,
            description=(
                "Reviewed templates, semantic patches, e-graphs, enumerative synthesis; "
                "candidate code only until gates pass."
            ),
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.REVIEWED_TEMPLATE,
                    role=MethodRole.OPTIONAL,
                    cost=20,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("reviewed_template",),
                ),
                _method(
                    StrategyMethod.SEMANTIC_PATCH,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("semantic_patch",),
                ),
                _method(
                    StrategyMethod.EGRAPH,
                    role=MethodRole.OPTIONAL,
                    cost=50,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("egraph",),
                ),
                _method(
                    StrategyMethod.ENUMERATIVE_SYNTHESIS,
                    role=MethodRole.OPTIONAL,
                    cost=70,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.DIAGNOSTIC_CANDIDATE,
                    capabilities=("enumerative_synthesis",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.SUPPLY_CHAIN,
            description="SBOM, lockfile/repro/SLSA, OSV/pip-audit scanners.",
            required_assurance=StrategyAssurance.OBSERVED,
            fallback=FallbackBehavior.DEBT_OPTIONAL,
            methods=(
                _method(
                    StrategyMethod.SBOM,
                    role=MethodRole.OPTIONAL,
                    cost=20,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("sbom",),
                ),
                _method(
                    StrategyMethod.LOCKFILE_REPRO,
                    role=MethodRole.OPTIONAL,
                    cost=25,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("lockfile_repro",),
                ),
                _method(
                    StrategyMethod.OSV_SCAN,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.OBSERVED,
                    authority=AuthorityUse.SECURITY_EVIDENCE,
                    capabilities=("osv_scan",),
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.RETRIEVAL,
            description=(
                "BM25, vectors, embeddings, GraphRAG, KG neighborhoods, learned ranking; "
                "nomination/ranking only."
            ),
            required_assurance=StrategyAssurance.CANDIDATE,
            fallback=FallbackBehavior.NOMINATION_ONLY,
            analysis_operations=graph_ops + premise_ops,
            methods=(
                _method(
                    StrategyMethod.BM25,
                    role=MethodRole.OPTIONAL,
                    cost=10,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("bm25", "graph_read"),
                    operations=graph_ops,
                    nomination_only=True,
                ),
                _method(
                    StrategyMethod.VECTOR_RETRIEVAL,
                    role=MethodRole.OPTIONAL,
                    cost=20,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("vector_index",),
                    operations=graph_ops,
                    nomination_only=True,
                ),
                _method(
                    StrategyMethod.EMBEDDING_RETRIEVAL,
                    role=MethodRole.OPTIONAL,
                    cost=25,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("embedding_retrieval",),
                    nomination_only=True,
                ),
                _method(
                    StrategyMethod.GRAPH_RAG,
                    role=MethodRole.OPTIONAL,
                    cost=30,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("graphrag_retrieval", "graph_read"),
                    operations=graph_ops,
                    nomination_only=True,
                ),
                _method(
                    StrategyMethod.KG_NEIGHBORHOOD,
                    role=MethodRole.OPTIONAL,
                    cost=35,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("kg_neighborhood",),
                    nomination_only=True,
                ),
                _method(
                    StrategyMethod.LEARNED_RANKING,
                    role=MethodRole.OPTIONAL,
                    cost=40,
                    assurance=StrategyAssurance.CANDIDATE,
                    authority=AuthorityUse.NOMINATION_ONLY,
                    capabilities=("learned_ranking",),
                    nomination_only=True,
                    learned_ranking=True,
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.FORMAL_KERNELS,
            description=(
                "Lean, Rocq/Coq, Isabelle kernels; assurance only after exact replay."
            ),
            required_assurance=StrategyAssurance.KERNEL_VERIFIED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            analysis_operations=proof_ops,
            methods=(
                _method(
                    StrategyMethod.LEAN_KERNEL,
                    role=MethodRole.OPTIONAL,
                    cost=100,
                    assurance=StrategyAssurance.KERNEL_VERIFIED,
                    authority=AuthorityUse.KERNEL_ASSURANCE,
                    capabilities=("lean", "kernel_replay"),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.ROCQ_KERNEL,
                    role=MethodRole.OPTIONAL,
                    cost=100,
                    assurance=StrategyAssurance.KERNEL_VERIFIED,
                    authority=AuthorityUse.KERNEL_ASSURANCE,
                    capabilities=("rocq", "kernel_replay"),
                    operations=proof_ops,
                ),
                _method(
                    StrategyMethod.ISABELLE_KERNEL,
                    role=MethodRole.OPTIONAL,
                    cost=100,
                    assurance=StrategyAssurance.KERNEL_VERIFIED,
                    authority=AuthorityUse.KERNEL_ASSURANCE,
                    capabilities=("isabelle", "kernel_replay"),
                    operations=proof_ops,
                ),
            ),
        ),
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.CRYPTOGRAPHIC_LINEAGE,
            description=(
                "CIDs/Merkle proofs, signatures, optional ZKP; integrity/privacy, "
                "not arbitrary code correctness."
            ),
            required_assurance=StrategyAssurance.ATTESTED,
            fallback=FallbackBehavior.ABSTAIN_REQUIRED,
            methods=(
                _method(
                    StrategyMethod.CID_MERKLE,
                    role=MethodRole.REQUIRED,
                    cost=10,
                    assurance=StrategyAssurance.ATTESTED,
                    authority=AuthorityUse.INTEGRITY_ATTESTATION,
                    capabilities=("cid_merkle", "content_identity"),
                ),
                _method(
                    StrategyMethod.SIGNATURE,
                    role=MethodRole.OPTIONAL,
                    cost=20,
                    assurance=StrategyAssurance.ATTESTED,
                    authority=AuthorityUse.INTEGRITY_ATTESTATION,
                    capabilities=("signature",),
                ),
                _method(
                    StrategyMethod.ZKP_ATTESTATION,
                    role=MethodRole.OPTIONAL,
                    cost=80,
                    assurance=StrategyAssurance.ATTESTED,
                    authority=AuthorityUse.INTEGRITY_ATTESTATION,
                    capabilities=("zkp_attestation",),
                ),
            ),
        ),
    )


def default_lazy_provider_adapters() -> tuple[LazyProviderAdapter, ...]:
    """Cold declarations for local and optional providers (no import/probe)."""

    return (
        LazyProviderAdapter(
            provider_id="supervisor-local-analysis",
            capability_ids=(
                "ast_index_read",
                "python_ast",
                "symbol_impact",
                "graph_read",
                "graphrag_retrieval",
                "bm25",
                "interface_diff",
                "contract_analysis",
                "cid_merkle",
                "content_identity",
                "security_ir",
            ),
            methods=(
                StrategyMethod.PYTHON_AST,
                StrategyMethod.SYMBOL_INDEX,
                StrategyMethod.CALL_GRAPH,
                StrategyMethod.CFG,
                StrategyMethod.BM25,
                StrategyMethod.GRAPH_RAG,
                StrategyMethod.INTERFACE_DIFF,
                StrategyMethod.CID_MERKLE,
                StrategyMethod.SECURITY_IR,
            ),
            provider_version="1.0.0",
            config_digest="config:local-deterministic@1",
            health=AnalysisProviderHealth.LAZY,
            factory=None,
        ),
        LazyProviderAdapter(
            provider_id="ipfs-datasets-analysis",
            capability_ids=(
                "tree_sitter",
                "polyglot_ast",
                "vector_index",
                "embedding_retrieval",
                "smt",
                "z3",
                "cvc5",
                "logic_family_routing",
                "tactician_hammer",
                "premise_selection",
                "learned_ranking",
            ),
            methods=(
                StrategyMethod.TREE_SITTER,
                StrategyMethod.VECTOR_RETRIEVAL,
                StrategyMethod.EMBEDDING_RETRIEVAL,
                StrategyMethod.SMT,
                StrategyMethod.Z3,
                StrategyMethod.CVC5,
                StrategyMethod.TACTICIAN_HAMMER,
                StrategyMethod.LEARNED_RANKING,
            ),
            provider_version="unknown",
            config_digest="config:optional-datasets@lazy",
            health=AnalysisProviderHealth.LAZY,
            factory=None,
        ),
        LazyProviderAdapter(
            provider_id="formal-kernel-matrix",
            capability_ids=("lean", "rocq", "isabelle", "kernel_replay"),
            methods=(
                StrategyMethod.LEAN_KERNEL,
                StrategyMethod.ROCQ_KERNEL,
                StrategyMethod.ISABELLE_KERNEL,
            ),
            provider_version="unknown",
            config_digest="config:prover-matrix@lazy",
            health=AnalysisProviderHealth.LAZY,
            factory=None,
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AnalysisStrategyRegistry:
    """Closed property-class → strategy router with capability receipts.

    Interface: ``AnalysisStrategyRegistry@1``
    """

    def __init__(self) -> None:
        self._strategies: dict[PropertyQuestionClass, AnalysisStrategySpec] = {}
        self._adapters: dict[str, LazyProviderAdapter] = {}
        self._adapter_order: list[str] = []
        self._capability_index: dict[str, list[str]] = {}
        # Maps capability_id -> provider_ids that declare it (lazy only).
        self._probed: dict[str, AnalysisCapabilityReceipt] = {}

    def register_strategy(
        self,
        spec: AnalysisStrategySpec,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(spec, AnalysisStrategySpec):
            raise AnalysisStrategyRegistryError(
                "strategy must be an AnalysisStrategySpec"
            )
        if spec.property_class in self._strategies and not replace_existing:
            raise AnalysisStrategyRegistryError(
                f"strategy already registered: {spec.property_class.value}"
            )
        self._strategies[spec.property_class] = spec

    def register_provider_adapter(
        self,
        adapter: LazyProviderAdapter,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(adapter, LazyProviderAdapter):
            raise AnalysisStrategyRegistryError(
                "adapter must be a LazyProviderAdapter"
            )
        if adapter.provider_id in self._adapters and not replace_existing:
            raise AnalysisStrategyRegistryError(
                f"provider adapter already registered: {adapter.provider_id}"
            )
        if adapter.provider_id not in self._adapters:
            self._adapter_order.append(adapter.provider_id)
        # Drop prior capability index entries for this provider.
        for capability_id, providers in list(self._capability_index.items()):
            self._capability_index[capability_id] = [
                item for item in providers if item != adapter.provider_id
            ]
            if not self._capability_index[capability_id]:
                del self._capability_index[capability_id]
        self._adapters[adapter.provider_id] = adapter
        for capability_id in adapter.capability_ids:
            self._capability_index.setdefault(capability_id, []).append(
                adapter.provider_id
            )

    def strategy(self, property_class: Any) -> AnalysisStrategySpec:
        key = normalize_property_class(property_class)
        try:
            return self._strategies[key]
        except KeyError as exc:
            raise AnalysisStrategyRegistryError(
                f"no strategy registered for property class: {key.value}"
            ) from exc

    get = strategy

    def strategies(self) -> tuple[AnalysisStrategySpec, ...]:
        return tuple(
            self._strategies[key]
            for key in sorted(self._strategies, key=lambda item: item.value)
        )

    list_strategies = strategies

    def provider_adapters(self) -> tuple[LazyProviderAdapter, ...]:
        return tuple(
            self._adapters[provider_id] for provider_id in self._adapter_order
        )

    def discover_provider_declarations(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        """Cold discovery: declarations only; factories are not invoked."""

        return tuple(item.to_dict() for item in self.provider_adapters())

    def property_classes(self) -> tuple[PropertyQuestionClass, ...]:
        return tuple(
            sorted(self._strategies, key=lambda item: item.value)
        )

    @property
    def registry_id(self) -> str:
        return _content_id(
            "analysis-strategy-registry",
            {
                "version": ANALYSIS_STRATEGY_REGISTRY_VERSION,
                "interface": ANALYSIS_STRATEGY_REGISTRY_INTERFACE,
                "strategies": [item.to_dict() for item in self.strategies()],
                "providers": [item.to_dict() for item in self.provider_adapters()],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_STRATEGY_REGISTRY_SCHEMA,
            "interface": ANALYSIS_STRATEGY_REGISTRY_INTERFACE,
            "registry_version": ANALYSIS_STRATEGY_REGISTRY_VERSION,
            "registry_id": self.registry_id,
            "strategies": [item.to_dict() for item in self.strategies()],
            "providers": [item.to_dict() for item in self.provider_adapters()],
            "authority": {
                "repository_mutation": False,
                "validation_omission_selection": False,
                "candidate_promotion": False,
                "proof_authority": False,
                "completion_authority": False,
                "retrieval_is_nomination_only": True,
                "import_does_not_imply_support": True,
            },
        }

    def bind_capability_receipt(
        self,
        *,
        method: Any,
        property_class: Any,
        capability_id: str,
        provider_id: str,
        admission: CapabilityAdmission | str = CapabilityAdmission.LAZY,
        health: AnalysisProviderHealth | str = AnalysisProviderHealth.LAZY,
        provider_version: str = "unknown",
        config_digest: str = "",
        capability_revision: str = "",
        reason_code: str = "",
        nomination_only: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> AnalysisCapabilityReceipt:
        """Build and index a receipt that binds health/version/config."""

        prop = normalize_property_class(property_class)
        strategy_id = ""
        if prop in self._strategies:
            strategy_id = self._strategies[prop].strategy_id
        receipt = AnalysisCapabilityReceipt(
            capability_id=capability_id,
            provider_id=provider_id,
            method=method,
            property_class=prop,
            admission=admission,
            health=health,
            provider_version=provider_version,
            config_digest=config_digest,
            capability_revision=capability_revision,
            reason_code=reason_code,
            nomination_only=nomination_only,
            strategy_id=strategy_id,
            details=dict(details or {}),
        )
        key = f"{receipt.provider_id}:{receipt.capability_id}:{receipt.method.value}"
        self._probed[key] = receipt
        return receipt

    def _receipts_for_method(
        self,
        binding: StrategyMethodBinding,
        *,
        property_class: PropertyQuestionClass,
        strategy_id: str,
        available_capabilities: Mapping[str, AnalysisCapabilityReceipt] | None,
    ) -> tuple[AnalysisCapabilityReceipt, ...]:
        """Resolve cold declarations and optional explicit availability map."""

        receipts: list[AnalysisCapabilityReceipt] = []
        if available_capabilities is not None:
            # Explicit map is authoritative: never fall back to cold adapters
            # and never infer support from importability.
            for capability_id in binding.provider_capabilities:
                receipt = available_capabilities.get(capability_id)
                if receipt is not None:
                    receipts.append(receipt)
            return tuple(receipts)

        for capability_id in binding.provider_capabilities:
            provider_ids = self._capability_index.get(capability_id, ())
            for provider_id in provider_ids:
                adapter = self._adapters[provider_id]
                if binding.method not in adapter.methods and capability_id not in (
                    adapter.capability_ids
                ):
                    continue
                probed_key = (
                    f"{provider_id}:{capability_id}:{binding.method.value}"
                )
                if probed_key in self._probed:
                    receipts.append(self._probed[probed_key])
                    continue
                receipts.append(
                    adapter.declaration_receipt(
                        method=binding.method,
                        property_class=property_class,
                        strategy_id=strategy_id,
                        nomination_only=binding.nomination_only,
                    )
                )
        # Method with no declared capability still yields an unprobed receipt.
        if not receipts and not binding.provider_capabilities:
            receipts.append(
                AnalysisCapabilityReceipt(
                    capability_id=f"method:{binding.method.value}",
                    provider_id="undeclared",
                    method=binding.method,
                    property_class=property_class,
                    admission=CapabilityAdmission.UNPROBED,
                    health=AnalysisProviderHealth.UNAVAILABLE,
                    reason_code="no_provider_capability_declared",
                    nomination_only=binding.nomination_only,
                    strategy_id=strategy_id,
                )
            )
        return tuple(receipts)

    def _method_usable(
        self,
        binding: StrategyMethodBinding,
        receipts: Sequence[AnalysisCapabilityReceipt],
        *,
        available_capabilities: Mapping[str, AnalysisCapabilityReceipt] | None,
    ) -> bool:
        if not binding.provider_capabilities:
            # Methods with no external capability binding are locally admissible
            # without probing; callers still cannot promote their verdicts.
            return True
        if available_capabilities is not None:
            # Explicit map is authoritative: every declared capability must be
            # present and usable. A missing key is unavailable (fail closed).
            for capability_id in binding.provider_capabilities:
                receipt = available_capabilities.get(capability_id)
                if receipt is None or not receipt.usable:
                    return False
            return True
        # Cold path: LAZY declarations count as provisionally usable so routes
        # can be nominated without probing. Importability is never consulted.
        if not receipts:
            return False
        return any(item.usable for item in receipts)

    def select(
        self,
        property_class: Any,
        *,
        required_assurance: StrategyAssurance | str | None = None,
        available_capabilities: Mapping[str, AnalysisCapabilityReceipt]
        | Mapping[str, bool]
        | None = None,
        prefer_nomination: bool = False,
    ) -> StrategySelection:
        """Select least-cost sufficient methods for a property class.

        ``available_capabilities`` may map capability_id → receipt or bool.
        When omitted, cold LAZY declarations are treated as provisionally
        usable; required missing capabilities still abstain once the map is
        explicit and marks them unavailable.
        """

        prop = normalize_property_class(property_class)
        spec = self.strategy(prop)
        target = normalize_strategy_assurance(
            required_assurance
            if required_assurance is not None
            else spec.required_assurance
        )
        normalized_caps = self._normalize_capability_map(
            available_capabilities, property_class=prop, strategy_id=spec.strategy_id
        )

        selected: list[StrategyMethodBinding] = []
        all_receipts: list[AnalysisCapabilityReceipt] = []
        debt: list[UncertaintyDebt] = []
        nomination_only = prop is PropertyQuestionClass.RETRIEVAL

        # Retrieval / learned ranking never satisfy proof-level assurance.
        if nomination_only and target.rank > StrategyAssurance.CANDIDATE.rank:
            return StrategySelection(
                property_class=prop,
                strategy_id=spec.strategy_id,
                outcome=SelectionOutcome.ABSTAIN,
                required_assurance=target,
                achieved_assurance=StrategyAssurance.UNVERIFIED,
                abstention_reason=(
                    "retrieval and learned ranking remain nomination-only and "
                    f"cannot satisfy {target.value} assurance"
                ),
                nomination_only=True,
            )

        # Evaluate methods in cost order (already sorted on the spec).
        for binding in spec.methods:
            receipts = self._receipts_for_method(
                binding,
                property_class=prop,
                strategy_id=spec.strategy_id,
                available_capabilities=normalized_caps,
            )
            all_receipts.extend(receipts)
            usable = self._method_usable(
                binding,
                receipts,
                available_capabilities=normalized_caps,
            )
            meets = (
                binding.nomination_only
                or binding.max_assurance.satisfies(target)
                or prefer_nomination
            )
            if binding.nomination_only:
                # Nomination methods may always be selected when usable, but
                # never raise achieved assurance above candidate.
                meets = usable
            if not usable:
                if binding.role is MethodRole.REQUIRED:
                    return StrategySelection(
                        property_class=prop,
                        strategy_id=spec.strategy_id,
                        outcome=SelectionOutcome.ABSTAIN,
                        receipts=tuple(all_receipts),
                        debt=tuple(debt),
                        required_assurance=target,
                        achieved_assurance=StrategyAssurance.UNVERIFIED,
                        abstention_reason=(
                            f"required method {binding.method.value} is unavailable"
                        ),
                        nomination_only=nomination_only or binding.nomination_only,
                    )
                debt.append(
                    UncertaintyDebt(
                        kind=UncertaintyDebtKind.OPTIONAL_METHOD_UNAVAILABLE,
                        method=binding.method,
                        property_class=prop,
                        message=(
                            f"optional method {binding.method.value} unavailable; "
                            "adds uncertainty debt without inventing truth"
                        ),
                        reason_code="optional_method_unavailable",
                        receipt=receipts[0] if receipts else None,
                    )
                )
                continue
            if not meets and not binding.nomination_only:
                # Usable but cannot meet required assurance: skip without debt
                # unless it was the only path (handled after loop).
                continue
            selected.append(binding)

        if not selected:
            # All-optional portfolio with nothing usable → debt-only or abstain.
            if spec.required_methods():
                reason = "no required method could be selected"
                return StrategySelection(
                    property_class=prop,
                    strategy_id=spec.strategy_id,
                    outcome=SelectionOutcome.ABSTAIN,
                    receipts=tuple(all_receipts),
                    debt=tuple(debt),
                    required_assurance=target,
                    achieved_assurance=StrategyAssurance.UNVERIFIED,
                    abstention_reason=reason,
                    nomination_only=nomination_only,
                )
            if debt:
                return StrategySelection(
                    property_class=prop,
                    strategy_id=spec.strategy_id,
                    outcome=SelectionOutcome.DEBT_ONLY,
                    receipts=tuple(all_receipts),
                    debt=tuple(debt),
                    required_assurance=target,
                    achieved_assurance=StrategyAssurance.UNVERIFIED,
                    nomination_only=nomination_only,
                )
            return StrategySelection(
                property_class=prop,
                strategy_id=spec.strategy_id,
                outcome=SelectionOutcome.ABSTAIN,
                receipts=tuple(all_receipts),
                required_assurance=target,
                achieved_assurance=StrategyAssurance.UNVERIFIED,
                abstention_reason=(
                    f"no method can satisfy {target.value} for {prop.value}"
                ),
                nomination_only=nomination_only,
            )

        # Prefer least-cost sufficient single method; retain lower-cost helpers
        # that also meet the bar (or nomination companions).
        sufficient = [
            item
            for item in selected
            if item.nomination_only
            or item.max_assurance.satisfies(target)
            or prefer_nomination
        ]
        if not sufficient and not nomination_only:
            return StrategySelection(
                property_class=prop,
                strategy_id=spec.strategy_id,
                outcome=SelectionOutcome.ABSTAIN,
                selected_methods=(),
                receipts=tuple(all_receipts),
                debt=tuple(debt),
                required_assurance=target,
                achieved_assurance=StrategyAssurance.UNVERIFIED,
                abstention_reason=(
                    f"no available method reaches {target.value} assurance"
                ),
                nomination_only=False,
            )
        if not sufficient:
            sufficient = list(selected)

        # Least-cost sufficient: keep the cheapest method that meets assurance
        # plus any cheaper nomination-only helpers already selected.
        sufficient.sort(key=lambda item: (item.cost_rank, item.method.value))
        primary = sufficient[0]
        chosen = [primary]
        for item in sufficient[1:]:
            # Keep additional methods only when they are strictly cheaper
            # nomination helpers or same-cost alternatives already ordered.
            if item.nomination_only and item.cost_rank <= primary.cost_rank:
                chosen.append(item)
            elif (
                item.max_assurance.satisfies(target)
                and item.cost_rank == primary.cost_rank
            ):
                chosen.append(item)

        achieved = StrategyAssurance.UNVERIFIED
        for item in chosen:
            if item.nomination_only:
                if achieved.rank < StrategyAssurance.CANDIDATE.rank:
                    achieved = StrategyAssurance.CANDIDATE
                continue
            if item.max_assurance.rank > achieved.rank:
                achieved = item.max_assurance

        outcome = SelectionOutcome.SELECTED
        if debt:
            outcome = SelectionOutcome.PARTIAL

        return StrategySelection(
            property_class=prop,
            strategy_id=spec.strategy_id,
            outcome=outcome,
            selected_methods=tuple(chosen),
            receipts=tuple(all_receipts),
            debt=tuple(debt),
            required_assurance=target,
            achieved_assurance=achieved,
            nomination_only=nomination_only
            or all(item.nomination_only for item in chosen),
        )

    select_least_cost_sufficient = select

    def _normalize_capability_map(
        self,
        available: Mapping[str, AnalysisCapabilityReceipt]
        | Mapping[str, bool]
        | None,
        *,
        property_class: PropertyQuestionClass,
        strategy_id: str,
    ) -> dict[str, AnalysisCapabilityReceipt] | None:
        if available is None:
            return None
        if not isinstance(available, Mapping):
            raise AnalysisStrategyRegistryError(
                "available_capabilities must be a mapping"
            )
        result: dict[str, AnalysisCapabilityReceipt] = {}
        for key, value in available.items():
            capability_id = _text(key, "capability_id", maximum=_MAX_IDENTIFIER)
            if isinstance(value, AnalysisCapabilityReceipt):
                result[capability_id] = value
                continue
            if isinstance(value, bool):
                if value:
                    result[capability_id] = AnalysisCapabilityReceipt(
                        capability_id=capability_id,
                        provider_id="explicit",
                        method=StrategyMethod.PYTHON_AST,  # placeholder method
                        property_class=property_class,
                        admission=CapabilityAdmission.AVAILABLE,
                        health=AnalysisProviderHealth.HEALTHY,
                        provider_version="explicit",
                        config_digest="explicit:true",
                        reason_code="explicit_available",
                        strategy_id=strategy_id,
                        details={"explicit_bool": True},
                    )
                else:
                    result[capability_id] = AnalysisCapabilityReceipt(
                        capability_id=capability_id,
                        provider_id="explicit",
                        method=StrategyMethod.PYTHON_AST,
                        property_class=property_class,
                        admission=CapabilityAdmission.UNAVAILABLE,
                        health=AnalysisProviderHealth.UNAVAILABLE,
                        provider_version="explicit",
                        config_digest="explicit:false",
                        reason_code="explicit_unavailable",
                        strategy_id=strategy_id,
                        details={"explicit_bool": False},
                    )
                continue
            raise AnalysisStrategyRegistryError(
                "available_capabilities values must be receipts or booleans"
            )
        return result

    def strategies_for_operation(
        self, operation: Any
    ) -> tuple[AnalysisStrategySpec, ...]:
        """Property strategies that reference a given analysis operation."""

        op = normalize_analysis_operation(operation).value
        return tuple(
            item
            for item in self.strategies()
            if op in item.analysis_operations
        )

    def operations_for_property_class(
        self, property_class: Any
    ) -> tuple[str, ...]:
        return self.strategy(property_class).analysis_operations


def create_default_analysis_strategy_registry(
    *,
    include_provider_adapters: bool = True,
) -> AnalysisStrategyRegistry:
    """Build the production strategy registry without probing providers."""

    registry = AnalysisStrategyRegistry()
    for spec in default_strategy_specs():
        registry.register_strategy(spec)
    if include_provider_adapters:
        for adapter in default_lazy_provider_adapters():
            registry.register_provider_adapter(adapter)
    return registry


build_default_analysis_strategy_registry = create_default_analysis_strategy_registry


def property_class_for_operation(operation: Any) -> tuple[PropertyQuestionClass, ...]:
    """Map a transport operation onto zero or more property classes."""

    op = normalize_analysis_operation(operation)
    mapping: dict[AnalysisOperation, tuple[PropertyQuestionClass, ...]] = {
        AnalysisOperation.SYMBOL_IMPACT: (
            PropertyQuestionClass.SYNTAX_STRUCTURE,
            PropertyQuestionClass.CONTROL_DATA_FLOW,
        ),
        AnalysisOperation.GRAPH_RAG_RETRIEVAL: (
            PropertyQuestionClass.RETRIEVAL,
        ),
        AnalysisOperation.PREMISE_SELECTION: (
            PropertyQuestionClass.RETRIEVAL,
            PropertyQuestionClass.CONSTRAINT_SOLVING,
        ),
        AnalysisOperation.CONTRADICTION_SEARCH: (
            PropertyQuestionClass.CONSTRAINT_SOLVING,
            PropertyQuestionClass.CONTRACTS,
        ),
        AnalysisOperation.LOGIC_TRANSLATION: (
            PropertyQuestionClass.CONTRACTS,
            PropertyQuestionClass.CONSTRAINT_SOLVING,
        ),
        AnalysisOperation.PROOF_CANDIDATE_ANALYSIS: (
            PropertyQuestionClass.CONSTRAINT_SOLVING,
            PropertyQuestionClass.FORMAL_KERNELS,
            PropertyQuestionClass.CONTRACTS,
        ),
        AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS: (
            PropertyQuestionClass.CONSTRAINT_SOLVING,
            PropertyQuestionClass.CONTRACTS,
            PropertyQuestionClass.BEHAVIORAL_TESTS,
        ),
    }
    return mapping.get(op, ())


__all__ = [
    "ANALYSIS_CAPABILITY_RECEIPT_SCHEMA",
    "ANALYSIS_STRATEGY_INPUT_SCHEMA",
    "ANALYSIS_STRATEGY_OUTPUT_SCHEMA",
    "ANALYSIS_STRATEGY_REGISTRY_INTERFACE",
    "ANALYSIS_STRATEGY_REGISTRY_SCHEMA",
    "ANALYSIS_STRATEGY_REGISTRY_VERSION",
    "ANALYSIS_STRATEGY_SCHEMA",
    "ANALYSIS_STRATEGY_SELECTION_SCHEMA",
    "AnalysisCapabilityReceipt",
    "AnalysisStrategyRegistry",
    "AnalysisStrategyRegistryError",
    "AnalysisStrategySpec",
    "AuthorityUse",
    "CapabilityAdmission",
    "FallbackBehavior",
    "LazyProviderAdapter",
    "MethodRole",
    "PropertyClass",
    "PropertyQuestionClass",
    "SelectionOutcome",
    "StrategyAssurance",
    "StrategyBudget",
    "StrategyCacheRules",
    "StrategyMethod",
    "StrategyMethodBinding",
    "StrategySelection",
    "UncertaintyDebt",
    "UncertaintyDebtKind",
    "build_default_analysis_strategy_registry",
    "create_default_analysis_strategy_registry",
    "default_lazy_provider_adapters",
    "default_strategy_specs",
    "normalize_property_class",
    "normalize_strategy_assurance",
    "normalize_strategy_method",
    "property_class_for_operation",
]
