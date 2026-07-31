"""Compile deterministic-doctor findings into independent goals and gated Tactician plans.

LPR-034 — content-addressed, fail-closed bridge between:

* :class:`~ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts.DeterministicDoctorFinding`
* :class:`~ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus.ProgramLogicPremiseCorpusBuilder`
* :class:`~ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts.ProgramLogicGoal`
* domain-neutral datasets Code Tactician (optional; exact-first local planner fallback)
* :class:`~ipfs_accelerate_py.agent_supervisor.validation.tactician_plan_gate.TacticianPlanGate`

Invariants (acceptance / plan §4.13.4):

* Preserve required input/output/error/effect/auth/resource/state/schema/
  placement/information/memory facets and every unknown frontier.
* Candidate code, cache metadata, tests-by-mere-success, KG/vector/embedding
  scores, and Tactician output cannot author expectation premises.
* Authoritative exact local routes always precede approximate nominations.
* Reject cycles, axiom smuggling, self-validation, prompt directives, changed
  roots, missing consumers/facets, forged IDs, unbounded routes, and
  score-based authority.
* Emit deterministic plans with ``semantic_authority=false`` and no
  LLM/model-provider route, or a typed abstention.
* Never invokes an LLM or remote model provider.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Protocol

from ..analysis.deterministic_doctor_contracts import (
    DETERMINISTIC_DOCTOR_VERSION,
    DeterministicDoctorError,
    DeterministicDoctorFinding,
    DeterministicDoctorSafetyError,
    DoctorAuthorityRoots,
    DoctorEvidenceRole,
    DoctorEvidenceSnapshot,
    DoctorRejectionReason,
    DoctorRepairDisposition,
    DoctorResourceBounds,
)
from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    LogicFacetRef,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    PremiseAuthority,
    PremiseFeatureSet,
    PremiseLicensePolicy,
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
    ProgramLogicPremiseCorpusBuilder,
    is_expectation_source_class,
    is_hypothesis_source_class,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from ..validation.tactician_plan_gate import (
    TacticianPlanGate,
    TacticianPlanGateBounds,
    TacticianPlanGateDisposition,
    TacticianPlanGateError,
    TacticianPlanGateReceipt,
    TacticianPlanRejectionReason,
)


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_TACTICIAN_INTERFACE: Final[str] = "DeterministicDoctorTactician@1"
DOCTOR_REPAIR_GOAL_COMPILER_INTERFACE: Final[str] = "DoctorRepairGoalCompiler@1"
DOCTOR_GOAL_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-goal-compilation@1"
)
DOCTOR_TACTICIAN_PLAN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-tactician-plan-receipt@1"
)
DOCTOR_TACTICIAN_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-tactician-bounds@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-tactician@1"
PLANNER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.deterministic-doctor-tactician@1"
)
CONTRACT_VERSION: Final[int] = DETERMINISTIC_DOCTOR_VERSION

MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_REF_BYTES: Final[int] = 512
MAX_GOALS_PER_FINDING: Final[int] = 64
MAX_PREMISES_PER_FINDING: Final[int] = 256
MAX_SUBGOALS: Final[int] = 64
MAX_ROUTES: Final[int] = 32
MAX_CANDIDATES: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_FACETS: Final[int] = 32

# Closed facet inventory required by LPR-034 acceptance (input/output map to TYPE).
REQUIRED_FACET_KINDS: Final[tuple[LogicFacetKind, ...]] = (
    LogicFacetKind.TYPE,  # input / output shape
    LogicFacetKind.ERROR,
    LogicFacetKind.EFFECT,
    LogicFacetKind.AUTHORIZATION,
    LogicFacetKind.RESOURCE,
    LogicFacetKind.STATE,
    LogicFacetKind.SCHEMA,
    LogicFacetKind.PLACEMENT,
    LogicFacetKind.INFORMATION,
    LogicFacetKind.MEMORY,
)

# Exact-first source routes for deterministic doctor plans.
# Authoritative / local exact facts precede nominating approximate routes.
# LLM is intentionally absent from the admitted route set.
EXACT_FIRST_SOURCE_ROUTES: Final[tuple[SourceRouteKind, ...]] = (
    SourceRouteKind.REVIEWED_CONTRACT,
    SourceRouteKind.NORMATIVE_SPEC,
    SourceRouteKind.LOCAL_STATIC,
    SourceRouteKind.DATAFLOW,
    SourceRouteKind.GRAPH,
    SourceRouteKind.REVIEWED_TEST,
    SourceRouteKind.HISTORY,
    SourceRouteKind.RUNTIME_WITNESS,
    SourceRouteKind.KNOWLEDGE_GRAPH,
    SourceRouteKind.VECTOR,
    SourceRouteKind.TACTICIAN,
    # SourceRouteKind.SOLVER is allowed as nominating proof search only later.
    SourceRouteKind.SOLVER,
)

_AUTHORITATIVE_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.NORMATIVE_SPEC,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
    }
)
_NOMINATING_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.HISTORY,
        SourceRouteKind.VECTOR,
        SourceRouteKind.KNOWLEDGE_GRAPH,
        SourceRouteKind.TACTICIAN,
        SourceRouteKind.LLM,
        SourceRouteKind.SOLVER,
        SourceRouteKind.RUNTIME_WITNESS,
    }
)

_HYPOTHESIS_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
        PremiseSourceClass.COMMENT,
        PremiseSourceClass.RUNTIME_WITNESS,
        PremiseSourceClass.HISTORY,
        PremiseSourceClass.VECTOR_ANALOGUE,
        PremiseSourceClass.KNOWLEDGE_GRAPH,
        PremiseSourceClass.MODEL_HYPOTHESIS,
        PremiseSourceClass.GIT_LINEAGE,
    }
)

_STATIC_FACT_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
        PremiseSourceClass.VALUE_PROVENANCE,
        PremiseSourceClass.PROGRAM_GRAPH,
        PremiseSourceClass.SCHEMA_PROTOCOL,
        PremiseSourceClass.LOCAL_STATIC,
        PremiseSourceClass.THEOREM_CORPUS,
    }
)

# Map doctor candidate / signal labels onto premise source classes.
_CANDIDATE_SIGNAL_TO_SOURCE: Final[Mapping[str, PremiseSourceClass]] = {
    "exact_symbol": PremiseSourceClass.LOCAL_STATIC,
    "exact_contract": PremiseSourceClass.REVIEWED_CONTRACT,
    "exact_value": PremiseSourceClass.VALUE_PROVENANCE,
    "exact_lineage": PremiseSourceClass.GIT_LINEAGE,
    "exact_graph": PremiseSourceClass.PROGRAM_GRAPH,
    "lexical": PremiseSourceClass.HISTORY,
    "knowledge_graph": PremiseSourceClass.KNOWLEDGE_GRAPH,
    "vector": PremiseSourceClass.VECTOR_ANALOGUE,
    "embedding": PremiseSourceClass.VECTOR_ANALOGUE,
    "kg": PremiseSourceClass.KNOWLEDGE_GRAPH,
    "cache": PremiseSourceClass.COMMENT,
    "cache_metadata": PremiseSourceClass.COMMENT,
    "proof_cache": PremiseSourceClass.COMMENT,
    "test_success": PremiseSourceClass.COMMENT,
    "mere_success": PremiseSourceClass.COMMENT,
    "candidate": PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
    "candidate_code": PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
    "model": PremiseSourceClass.MODEL_HYPOTHESIS,
    "llm": PremiseSourceClass.MODEL_HYPOTHESIS,
    "tactician": PremiseSourceClass.MODEL_HYPOTHESIS,
}

_SOURCE_CLASS_TO_ROUTE: Final[Mapping[PremiseSourceClass, SourceRouteKind]] = {
    PremiseSourceClass.REVIEWED_CONTRACT: SourceRouteKind.REVIEWED_CONTRACT,
    PremiseSourceClass.NORMATIVE_SPEC: SourceRouteKind.NORMATIVE_SPEC,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST: SourceRouteKind.REVIEWED_TEST,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.VALUE_PROVENANCE: SourceRouteKind.DATAFLOW,
    PremiseSourceClass.PROGRAM_GRAPH: SourceRouteKind.GRAPH,
    PremiseSourceClass.SCHEMA_PROTOCOL: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.LOCAL_STATIC: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.COMMENT: SourceRouteKind.HISTORY,
    PremiseSourceClass.RUNTIME_WITNESS: SourceRouteKind.RUNTIME_WITNESS,
    PremiseSourceClass.HISTORY: SourceRouteKind.HISTORY,
    PremiseSourceClass.VECTOR_ANALOGUE: SourceRouteKind.VECTOR,
    PremiseSourceClass.KNOWLEDGE_GRAPH: SourceRouteKind.KNOWLEDGE_GRAPH,
    PremiseSourceClass.MODEL_HYPOTHESIS: SourceRouteKind.LLM,
    PremiseSourceClass.THEOREM_CORPUS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.GIT_LINEAGE: SourceRouteKind.HISTORY,
}

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
        "raw_text",
        "ast",
        "ast_body",
        "embedding",
        "query_vector",
        "model_output",
        "completion",
        "prompt",
        "prompt_body",
        "transcript",
        "theorem_text",
        "proof_script",
    }
)
_SECRET_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
        "authorization",
        "cookie",
        "session",
    }
)
_PROMPT_DIRECTIVE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(ignore previous|system prompt|you are an? ai|"
    r"disregard instructions|jailbreak|developer message)\b"
)
_SCORE_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "score_authority",
        "embedding_authority",
        "vector_authority",
        "kg_authority",
        "score_overrides_hard_failure",
        "rank_as_authority",
        "semantic_score_authority",
    }
)


# ---------------------------------------------------------------------------
# Errors / dispositions
# ---------------------------------------------------------------------------


class DoctorTacticianError(ContractValidationError):
    """Base failure for doctor goal compilation / Tactician planning."""


class DoctorTacticianAuthorityError(DoctorTacticianError):
    """Root, expectation, or semantic-authority boundary failure."""


class DoctorTacticianBoundsError(DoctorTacticianError):
    """A producer attempted to exceed fixed doctor-tactician budgets."""


class DoctorTacticianSafetyError(DoctorTacticianError):
    """Body/secret/prompt/model-route safety violation."""


class DoctorGoalCompilationDisposition(str, Enum):
    """Closed outcomes of compiling one finding into goals/premises."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class DoctorTacticianPlanDisposition(str, Enum):
    """Closed outcomes of a gated doctor Tactician plan."""

    PLANNED = "planned"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    GATE_REJECTED = "gate_rejected"
    GATE_ABSTAINED = "gate_abstained"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


class DoctorTacticianReasonCode(str, Enum):
    """Stable fail-closed reason codes for LPR-034."""

    OK = "ok"
    EXPECTATION_MISSING = "expectation_missing"
    EXPECTATION_CONFLICT = "expectation_conflict"
    EXPECTATION_SELF_AUTHORED = "expectation_self_authored"
    CANDIDATE_AUTHORED_EXPECTATION = "candidate_authored_expectation"
    CACHE_METADATA_EXPECTATION = "cache_metadata_expectation"
    TEST_SUCCESS_EXPECTATION = "test_success_expectation"
    SCORE_BASED_AUTHORITY = "score_based_authority"
    SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
    SELF_VALIDATION = "self_validation"
    AXIOM_SMUGGLING = "axiom_smuggling"
    PROMPT_DIRECTIVE = "prompt_directive"
    CHANGED_ROOTS = "changed_roots"
    MIXED_ROOTS = "mixed_roots"
    MISSING_CONSUMER = "missing_consumer"
    MISSING_FACET = "missing_facet"
    OPEN_REQUIRED_FRONTIER = "open_required_frontier"
    FORGED_IDENTITY = "forged_identity"
    UNBOUNDED_ROUTE = "unbounded_route"
    CYCLE = "cycle"
    LLM_ROUTE = "llm_route"
    MODEL_PROVIDER_ROUTE = "model_provider_route"
    BODY_OR_SECRET = "body_or_secret"
    FINDING_UNSUPPORTED = "finding_unsupported"
    FINDING_ABSTAIN = "finding_abstain"
    NO_GOALS = "no_goals"
    GATE_REJECTED = "gate_rejected"
    GATE_ABSTAINED = "gate_abstained"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_REFUSED = "provider_refused"
    MALFORMED_INPUT = "malformed_input"
    STALE_SNAPSHOT = "stale_snapshot"
    BUDGET_EXCEEDED = "budget_exceeded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if required and not text:
        raise DoctorTacticianError(f"{field_name} is required")
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise DoctorTacticianBoundsError(f"{field_name} is invalid or exceeds its bound")
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=MAX_REF_BYTES)
    if any(ch.isspace() for ch in text):
        raise DoctorTacticianError(f"{field_name} must be a compact identifier")
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorTacticianError(f"{field_name} must be a boolean")
    return value


def _ids(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_PREMISES_PER_FINDING,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise DoctorTacticianError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise DoctorTacticianBoundsError(f"{field_name} exceeds its item bound")
    items: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _identifier(item, field_name)
        if text not in seen:
            seen.add(text)
            items.append(text)
    if not preserve_order:
        items.sort()
    if required and not items:
        raise DoctorTacticianError(f"{field_name} must not be empty")
    return tuple(items)


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(value))
    except (TypeError, ValueError) as exc:
        raise DoctorTacticianError(f"{field_name} has an unsupported value") from exc


def _stable_id(prefix: str, payload: Any) -> str:
    material = content_identity(
        {
            "schema": f"ipfs_accelerate_py/agent-supervisor/doctor-tactician/{prefix}@1",
            "payload": payload,
        }
    )
    return f"{prefix}:{material}"


def _digest(statement_ref: str, extra: str = "") -> str:
    material = f"{statement_ref}\0{extra}".encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_MARKERS or normalized in _SECRET_MARKERS:
                raise DoctorTacticianSafetyError(
                    f"{field_name} may not contain body or secret field {key!r}"
                )
            if _PROMPT_DIRECTIVE_RE.search(str(key)):
                raise DoctorTacticianSafetyError(
                    f"{field_name} may not embed prompt directives in keys"
                )
            _assert_body_free(item, field_name)
        return
    if isinstance(value, (bytes, bytearray)):
        raise DoctorTacticianSafetyError(f"{field_name} may not contain binary bodies")
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            _assert_body_free(item, field_name)
        return
    if isinstance(value, str) and _PROMPT_DIRECTIVE_RE.search(value):
        raise DoctorTacticianSafetyError(
            f"{field_name} may not contain prompt directives"
        )


def _bounded(record: CanonicalContract, name: str) -> None:
    encoded = record.to_json_bytes() if hasattr(record, "to_json_bytes") else None
    if encoded is None:
        payload = record.to_dict()
        encoded = repr(payload).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise DoctorTacticianBoundsError(f"{name} exceeds its serialized byte bound")


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise DoctorTacticianAuthorityError(
            "stored content identity does not match the canonical record"
        )


def _decode_fields(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") not in (None, schema):
        raise DoctorTacticianError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, CONTRACT_VERSION):
        raise DoctorTacticianError(f"{name} has an unsupported contract version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise DoctorTacticianError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    return {field: payload[field] for field in fields if field in payload}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        if isinstance(result, Mapping):
            return dict(result)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _detect_cycle(nodes: Mapping[str, Sequence[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visited:
            return False
        if node in visiting:
            return True
        visiting.add(node)
        for dep in nodes.get(node, ()):
            if dfs(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node) for node in nodes)


def doctor_roots_to_program_logic_roots(
    roots: DoctorAuthorityRoots | Mapping[str, Any],
    *,
    objective_id: str,
    trace_id: str,
    change_id: str,
    consumer_id: str,
) -> ProgramLogicAuthorityRoots:
    """Project doctor authority roots into program-logic prediction roots."""
    if isinstance(roots, Mapping):
        doctor = DoctorAuthorityRoots(
            **{
                key: roots[key]
                for key in DoctorAuthorityRoots.__dataclass_fields__
                if key != "SCHEMA" and key in roots
            }
        )
    elif isinstance(roots, DoctorAuthorityRoots):
        doctor = roots
    else:
        raise DoctorTacticianError("roots must be DoctorAuthorityRoots")
    return ProgramLogicAuthorityRoots(
        repository_id=doctor.repository_id,
        objective_id=_identifier(objective_id, "objective_id"),
        trace_id=_identifier(trace_id, "trace_id"),
        change_id=_identifier(change_id, "change_id"),
        consumer_id=_identifier(consumer_id, "consumer_id"),
        forest_id=doctor.forest_id,
        tree_id=doctor.tree_id,
        overlay_id=doctor.overlay_id,
        graph_id=doctor.graph_id,
        index_id=doctor.index_id,
        corpus_id=doctor.corpus_id,
        model_id=doctor.model_id,
        translator_id=doctor.translator_id,
        toolchain_id=doctor.toolchain_id,
        policy_id=doctor.policy_id,
        environment_id=doctor.environment_id,
    )


def _route_authority(route: SourceRouteKind) -> SourceAuthorityClass:
    if route in _AUTHORITATIVE_ROUTES:
        return SourceAuthorityClass.AUTHORITATIVE
    if route is SourceRouteKind.REVIEWED_TEST:
        return SourceAuthorityClass.CONFORMANCE
    if route in _NOMINATING_ROUTES:
        return SourceAuthorityClass.NOMINATING
    return SourceAuthorityClass.NONE


def _source_class_from_label(label: str) -> PremiseSourceClass:
    normalized = label.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in _CANDIDATE_SIGNAL_TO_SOURCE:
        return _CANDIDATE_SIGNAL_TO_SOURCE[normalized]
    try:
        return PremiseSourceClass(normalized)
    except ValueError:
        return PremiseSourceClass.CANDIDATE_IMPLEMENTATION


def _is_cache_metadata_ref(ref: str) -> bool:
    lowered = ref.casefold()
    return any(
        token in lowered
        for token in (
            "cache:meta",
            "cache_metadata",
            "proof_cache_meta",
            "cache:hit",
            "cache:timeout",
            "cache:negative",
        )
    )


def _is_test_success_only_ref(ref: str) -> bool:
    lowered = ref.casefold()
    return any(
        token in lowered
        for token in (
            "test:success",
            "test_success",
            "mere_success",
            "tests_passed_only",
            "green_test_only",
        )
    )


def _is_score_authority_payload(payload: Mapping[str, Any]) -> bool:
    for key, value in payload.items():
        normalized = str(key).casefold().replace("-", "_")
        if normalized in _SCORE_AUTHORITY_KEYS and value:
            return True
        if normalized in {
            "semantic_authority",
            "expectation_authority",
            "write_authority",
        } and value is True:
            return True
        if normalized in {"score_selects_target", "score_selects_value", "score_as_proof"}:
            if value:
                return True
    return False


def _license() -> PremiseLicensePolicy:
    return PremiseLicensePolicy(
        license_id="license:doctor-internal",
        redaction_policy="span_only",
        export_policy="internal",
    )


def _features_for_finding(finding: DeterministicDoctorFinding) -> PremiseFeatureSet:
    return PremiseFeatureSet(
        symbol_feature_refs=tuple(finding.affected_symbol_refs[:32]),
        type_feature_refs=tuple(
            ref for ref in finding.expected_behavior_refs if "type" in ref.casefold()
        )[:32],
        effect_feature_refs=tuple(
            ref for ref in finding.observed_fact_refs if "effect" in ref.casefold()
        )[:32],
        import_feature_refs=(),
    )


# ---------------------------------------------------------------------------
# Bounds / records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorTacticianBounds(CanonicalContract):
    """Integer budgets for doctor goal compilation and Tactician planning."""

    SCHEMA: ClassVar[str] = DOCTOR_TACTICIAN_BOUNDS_SCHEMA

    max_goals: int = MAX_GOALS_PER_FINDING
    max_premises: int = MAX_PREMISES_PER_FINDING
    max_subgoals: int = MAX_SUBGOALS
    max_routes: int = MAX_ROUTES
    max_candidates: int = MAX_CANDIDATES
    allow_approximate_routes: bool = True
    allow_model_hypothesis: bool = False
    require_exact_before_approximate: bool = True
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_goals",
            "max_premises",
            "max_subgoals",
            "max_routes",
            "max_candidates",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise DoctorTacticianBoundsError(f"{name} must be a positive integer")
            hard = {
                "max_goals": MAX_GOALS_PER_FINDING,
                "max_premises": MAX_PREMISES_PER_FINDING,
                "max_subgoals": MAX_SUBGOALS,
                "max_routes": MAX_ROUTES,
                "max_candidates": MAX_CANDIDATES,
            }[name]
            if value > hard:
                raise DoctorTacticianBoundsError(f"{name} exceeds hard maximum {hard}")
        for flag in (
            "allow_approximate_routes",
            "allow_model_hypothesis",
            "require_exact_before_approximate",
            "semantic_authority",
        ):
            object.__setattr__(self, flag, _bool(getattr(self, flag), flag))
        if self.semantic_authority is not False:
            raise DoctorTacticianSafetyError(
                "doctor tactician bounds cannot claim semantic_authority"
            )
        if self.allow_model_hypothesis is not False:
            raise DoctorTacticianSafetyError(
                "deterministic doctor mode forbids model-hypothesis routes"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "allow_model_hypothesis", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "max_goals": self.max_goals,
            "max_premises": self.max_premises,
            "max_subgoals": self.max_subgoals,
            "max_routes": self.max_routes,
            "max_candidates": self.max_candidates,
            "allow_approximate_routes": self.allow_approximate_routes,
            "allow_model_hypothesis": False,
            "require_exact_before_approximate": self.require_exact_before_approximate,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorTacticianBounds":
        fields = (
            "max_goals",
            "max_premises",
            "max_subgoals",
            "max_routes",
            "max_candidates",
            "allow_approximate_routes",
            "allow_model_hypothesis",
            "require_exact_before_approximate",
            "semantic_authority",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "doctor tactician bounds")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorGoalCompilation(CanonicalContract):
    """Independent goal / premise inventory compiled from one doctor finding."""

    SCHEMA: ClassVar[str] = DOCTOR_GOAL_COMPILATION_SCHEMA

    roots: ProgramLogicAuthorityRoots
    compilation_id: str
    finding_id: str
    snapshot_id: str
    disposition: DoctorGoalCompilationDisposition
    goals: tuple[ProgramLogicGoal, ...]
    corpus: ProgramLogicPremiseCorpus
    required_facet_ids: tuple[str, ...] = ()
    unknown_frontier_refs: tuple[str, ...] = ()
    excluded_expectation_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    selected_expectation_ids: tuple[str, ...] = ()
    selected_observation_ids: tuple[str, ...] = ()
    selected_hypothesis_ids: tuple[str, ...] = ()
    consumer_ids: tuple[str, ...] = ()
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramLogicAuthorityRoots):
            raise DoctorTacticianError("roots must be ProgramLogicAuthorityRoots")
        object.__setattr__(
            self, "compilation_id", _identifier(self.compilation_id, "compilation_id")
        )
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorGoalCompilationDisposition, "disposition"),
        )
        if not isinstance(self.goals, tuple):
            object.__setattr__(self, "goals", tuple(self.goals or ()))
        if len(self.goals) > MAX_GOALS_PER_FINDING:
            raise DoctorTacticianBoundsError("goals exceeds its item bound")
        for goal in self.goals:
            if not isinstance(goal, ProgramLogicGoal):
                raise DoctorTacticianError("goals must contain ProgramLogicGoal values")
            if goal.roots.content_id != self.roots.content_id:
                raise DoctorTacticianAuthorityError(
                    "goal roots must match compilation roots"
                )
        if not isinstance(self.corpus, ProgramLogicPremiseCorpus):
            raise DoctorTacticianError("corpus must be ProgramLogicPremiseCorpus")
        if self.corpus.roots.content_id != self.roots.content_id:
            raise DoctorTacticianAuthorityError(
                "corpus roots must match compilation roots"
            )
        object.__setattr__(
            self,
            "required_facet_ids",
            _ids(self.required_facet_ids, "required_facet_ids", limit=MAX_FACETS),
        )
        object.__setattr__(
            self,
            "unknown_frontier_refs",
            _ids(self.unknown_frontier_refs, "unknown_frontier_refs"),
        )
        object.__setattr__(
            self,
            "excluded_expectation_refs",
            _ids(self.excluded_expectation_refs, "excluded_expectation_refs"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "selected_expectation_ids",
            _ids(self.selected_expectation_ids, "selected_expectation_ids"),
        )
        object.__setattr__(
            self,
            "selected_observation_ids",
            _ids(self.selected_observation_ids, "selected_observation_ids"),
        )
        object.__setattr__(
            self,
            "selected_hypothesis_ids",
            _ids(self.selected_hypothesis_ids, "selected_hypothesis_ids"),
        )
        object.__setattr__(
            self, "consumer_ids", _ids(self.consumer_ids, "consumer_ids")
        )
        if self.semantic_authority is not False:
            raise DoctorTacticianSafetyError(
                "goal compilations cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )
        _bounded(self, "doctor goal compilation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "compilation_id": self.compilation_id,
            "finding_id": self.finding_id,
            "snapshot_id": self.snapshot_id,
            "disposition": self.disposition.value,
            "goals": [goal.to_dict() for goal in self.goals],
            "corpus": self.corpus.to_dict(),
            "required_facet_ids": list(self.required_facet_ids),
            "unknown_frontier_refs": list(self.unknown_frontier_refs),
            "excluded_expectation_refs": list(self.excluded_expectation_refs),
            "reason_codes": list(self.reason_codes),
            "selected_expectation_ids": list(self.selected_expectation_ids),
            "selected_observation_ids": list(self.selected_observation_ids),
            "selected_hypothesis_ids": list(self.selected_hypothesis_ids),
            "consumer_ids": list(self.consumer_ids),
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorGoalCompilation":
        fields = (
            "roots",
            "compilation_id",
            "finding_id",
            "snapshot_id",
            "disposition",
            "goals",
            "corpus",
            "required_facet_ids",
            "unknown_frontier_refs",
            "excluded_expectation_refs",
            "reason_codes",
            "selected_expectation_ids",
            "selected_observation_ids",
            "selected_hypothesis_ids",
            "consumer_ids",
            "semantic_authority",
            "invalidation_refs",
            "producer_id",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "doctor goal compilation")
        roots = values["roots"]
        values["roots"] = (
            roots
            if isinstance(roots, ProgramLogicAuthorityRoots)
            else ProgramLogicAuthorityRoots.from_dict(roots)
            if isinstance(roots, Mapping) and "schema" in roots
            else ProgramLogicAuthorityRoots(**roots)
        )
        goals = values.get("goals") or ()
        values["goals"] = tuple(
            item
            if isinstance(item, ProgramLogicGoal)
            else ProgramLogicGoal.from_dict(item)
            if isinstance(item, Mapping) and "schema" in item
            else ProgramLogicGoal(**item)
            for item in goals
        )
        corpus = values["corpus"]
        values["corpus"] = (
            corpus
            if isinstance(corpus, ProgramLogicPremiseCorpus)
            else ProgramLogicPremiseCorpus.from_dict(corpus)
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorTacticianPlanReceipt(CanonicalContract):
    """Gated, non-authoritative Tactician plan receipt for one doctor finding."""

    SCHEMA: ClassVar[str] = DOCTOR_TACTICIAN_PLAN_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    finding_id: str
    snapshot_id: str
    compilation_id: str
    disposition: DoctorTacticianPlanDisposition
    reason_codes: tuple[str, ...] = ()
    plan: TacticianSearchPlan | None = None
    gate_receipt: TacticianPlanGateReceipt | None = None
    ordered_source_routes: tuple[str, ...] = ()
    selected_premise_ids: tuple[str, ...] = ()
    excluded_premise_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    required_facet_ids: tuple[str, ...] = ()
    unknown_frontier_refs: tuple[str, ...] = ()
    exclusion_rationale_refs: tuple[str, ...] = ()
    completeness: str = "complete"
    budget_refs: tuple[str, ...] = ()
    provider_status: str = ""
    planner_id: str = PLANNER_ID
    semantic_authority: bool = False
    model_invocation_count: int = 0
    llm_route_present: bool = False
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramLogicAuthorityRoots):
            raise DoctorTacticianError("roots must be ProgramLogicAuthorityRoots")
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "compilation_id", _identifier(self.compilation_id, "compilation_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorTacticianPlanDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES),
        )
        if self.plan is not None:
            if not isinstance(self.plan, TacticianSearchPlan):
                raise DoctorTacticianError("plan must be TacticianSearchPlan")
            if self.plan.semantic_authority is not False:
                raise DoctorTacticianSafetyError(
                    "plan cannot claim semantic_authority"
                )
            if self.plan.roots.content_id != self.roots.content_id:
                raise DoctorTacticianAuthorityError(
                    "plan roots must match receipt roots"
                )
            if SourceRouteKind.LLM in self.plan.ordered_source_routes:
                raise DoctorTacticianSafetyError(
                    "deterministic doctor plans cannot include an LLM route"
                )
        if self.gate_receipt is not None:
            if not isinstance(self.gate_receipt, TacticianPlanGateReceipt):
                raise DoctorTacticianError(
                    "gate_receipt must be TacticianPlanGateReceipt"
                )
            if self.gate_receipt.semantic_authority is not False:
                raise DoctorTacticianSafetyError(
                    "gate receipt cannot claim semantic_authority"
                )
            if self.gate_receipt.write_authority is not False:
                raise DoctorTacticianSafetyError(
                    "gate receipt cannot claim write authority"
                )
        object.__setattr__(
            self,
            "ordered_source_routes",
            _ids(
                self.ordered_source_routes,
                "ordered_source_routes",
                limit=MAX_ROUTES,
                preserve_order=True,
            ),
        )
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
        object.__setattr__(self, "goal_ids", _ids(self.goal_ids, "goal_ids"))
        object.__setattr__(
            self,
            "required_facet_ids",
            _ids(self.required_facet_ids, "required_facet_ids", limit=MAX_FACETS),
        )
        object.__setattr__(
            self,
            "unknown_frontier_refs",
            _ids(self.unknown_frontier_refs, "unknown_frontier_refs"),
        )
        object.__setattr__(
            self,
            "exclusion_rationale_refs",
            _ids(self.exclusion_rationale_refs, "exclusion_rationale_refs"),
        )
        object.__setattr__(
            self,
            "completeness",
            _text(self.completeness or "complete", "completeness", limit=64),
        )
        object.__setattr__(
            self, "budget_refs", _ids(self.budget_refs, "budget_refs")
        )
        object.__setattr__(
            self,
            "provider_status",
            _text(self.provider_status, "provider_status", required=False, limit=128),
        )
        object.__setattr__(
            self, "planner_id", _text(self.planner_id or PLANNER_ID, "planner_id")
        )
        if self.semantic_authority is not False:
            raise DoctorTacticianSafetyError(
                "receipts cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        if not isinstance(self.model_invocation_count, int) or isinstance(
            self.model_invocation_count, bool
        ):
            raise DoctorTacticianError("model_invocation_count must be an integer")
        if self.model_invocation_count != 0:
            raise DoctorTacticianSafetyError(
                "deterministic doctor mode requires zero model invocations"
            )
        object.__setattr__(self, "model_invocation_count", 0)
        object.__setattr__(
            self, "llm_route_present", _bool(self.llm_route_present, "llm_route_present")
        )
        if self.llm_route_present:
            raise DoctorTacticianSafetyError(
                "deterministic doctor receipts cannot record an LLM route"
            )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )
        # Disposition consistency.
        if (
            self.disposition is DoctorTacticianPlanDisposition.PLANNED
            and self.plan is None
        ):
            raise DoctorTacticianError("planned receipts require a search plan")
        if self.disposition is DoctorTacticianPlanDisposition.PLANNED and not self.goal_ids:
            raise DoctorTacticianError("planned receipts require goal ids")
        _bounded(self, "doctor tactician plan receipt")

    @property
    def is_planned(self) -> bool:
        return self.disposition is DoctorTacticianPlanDisposition.PLANNED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "finding_id": self.finding_id,
            "snapshot_id": self.snapshot_id,
            "compilation_id": self.compilation_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "gate_receipt": (
                self.gate_receipt.to_dict() if self.gate_receipt is not None else None
            ),
            "ordered_source_routes": list(self.ordered_source_routes),
            "selected_premise_ids": list(self.selected_premise_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "goal_ids": list(self.goal_ids),
            "required_facet_ids": list(self.required_facet_ids),
            "unknown_frontier_refs": list(self.unknown_frontier_refs),
            "exclusion_rationale_refs": list(self.exclusion_rationale_refs),
            "completeness": self.completeness,
            "budget_refs": list(self.budget_refs),
            "provider_status": self.provider_status,
            "planner_id": self.planner_id,
            "semantic_authority": False,
            "model_invocation_count": 0,
            "llm_route_present": False,
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorTacticianPlanReceipt":
        fields = (
            "roots",
            "receipt_id",
            "finding_id",
            "snapshot_id",
            "compilation_id",
            "disposition",
            "reason_codes",
            "plan",
            "gate_receipt",
            "ordered_source_routes",
            "selected_premise_ids",
            "excluded_premise_ids",
            "goal_ids",
            "required_facet_ids",
            "unknown_frontier_refs",
            "exclusion_rationale_refs",
            "completeness",
            "budget_refs",
            "provider_status",
            "planner_id",
            "semantic_authority",
            "model_invocation_count",
            "llm_route_present",
            "invalidation_refs",
            "producer_id",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "doctor tactician plan receipt"
        )
        roots = values["roots"]
        values["roots"] = (
            roots
            if isinstance(roots, ProgramLogicAuthorityRoots)
            else ProgramLogicAuthorityRoots.from_dict(roots)
            if isinstance(roots, Mapping) and "schema" in roots
            else ProgramLogicAuthorityRoots(**roots)
        )
        plan = values.get("plan")
        if plan is not None and not isinstance(plan, TacticianSearchPlan):
            values["plan"] = (
                TacticianSearchPlan.from_dict(plan)
                if isinstance(plan, Mapping)
                else plan
            )
        gate = values.get("gate_receipt")
        if gate is not None and not isinstance(gate, TacticianPlanGateReceipt):
            values["gate_receipt"] = (
                TacticianPlanGateReceipt.from_dict(gate)
                if isinstance(gate, Mapping)
                else gate
            )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# DoctorRepairGoalCompiler
# ---------------------------------------------------------------------------


class DoctorRepairGoalCompiler:
    """Compile one :class:`DeterministicDoctorFinding` into independent goals.

    Expectations and observed facts are admitted before any candidate or
    Tactician material. Candidates / scores / cache metadata / mere test
    success can only become non-authoritative hypotheses.
    """

    def __init__(
        self,
        bounds: DoctorTacticianBounds | Mapping[str, Any] | None = None,
    ) -> None:
        if bounds is None:
            self._bounds = DoctorTacticianBounds()
        elif isinstance(bounds, DoctorTacticianBounds):
            self._bounds = bounds
        elif isinstance(bounds, Mapping):
            self._bounds = DoctorTacticianBounds.from_dict(bounds)
        else:
            raise DoctorTacticianError("bounds must be DoctorTacticianBounds")

    @property
    def bounds(self) -> DoctorTacticianBounds:
        return self._bounds

    def compile(
        self,
        finding: DeterministicDoctorFinding | Mapping[str, Any],
        *,
        snapshot: DoctorEvidenceSnapshot | Mapping[str, Any] | None = None,
        candidates: Sequence[Any] = (),
        current_roots: DoctorAuthorityRoots | Mapping[str, Any] | None = None,
        objective_id: str = "",
        consumer_id: str = "",
        extra_expectation_refs: Sequence[str] = (),
        extra_observation_refs: Sequence[str] = (),
    ) -> DoctorGoalCompilation:
        typed_finding = self._decode_finding(finding)
        typed_snapshot = self._decode_snapshot(snapshot)
        doctor_roots = typed_finding.roots
        if current_roots is not None:
            expected = (
                current_roots
                if isinstance(current_roots, DoctorAuthorityRoots)
                else DoctorAuthorityRoots(
                    **{
                        key: current_roots[key]
                        for key in DoctorAuthorityRoots.__dataclass_fields__
                        if key != "SCHEMA" and key in current_roots
                    }
                )
            )
            if expected.content_id != doctor_roots.content_id:
                return self._reject_compilation(
                    typed_finding,
                    typed_snapshot,
                    reasons=(DoctorTacticianReasonCode.CHANGED_ROOTS.value,),
                    objective_id=objective_id,
                    consumer_id=consumer_id,
                )

        if typed_snapshot is not None:
            if typed_snapshot.roots.content_id != doctor_roots.content_id:
                return self._reject_compilation(
                    typed_finding,
                    typed_snapshot,
                    reasons=(DoctorTacticianReasonCode.MIXED_ROOTS.value,),
                    objective_id=objective_id,
                    consumer_id=consumer_id,
                )
            if typed_snapshot.snapshot_id != typed_finding.snapshot_id:
                return self._reject_compilation(
                    typed_finding,
                    typed_snapshot,
                    reasons=(DoctorTacticianReasonCode.STALE_SNAPSHOT.value,),
                    objective_id=objective_id,
                    consumer_id=consumer_id,
                )

        # Safety scans on finding payload.
        try:
            _assert_body_free(typed_finding.to_dict(), "finding")
        except DoctorTacticianSafetyError:
            return self._reject_compilation(
                typed_finding,
                typed_snapshot,
                reasons=(DoctorTacticianReasonCode.BODY_OR_SECRET.value,),
                objective_id=objective_id,
                consumer_id=consumer_id,
            )

        if typed_finding.semantic_authority is not False:
            return self._reject_compilation(
                typed_finding,
                typed_snapshot,
                reasons=(DoctorTacticianReasonCode.SEMANTIC_AUTHORITY_CLAIM.value,),
                objective_id=objective_id,
                consumer_id=consumer_id,
            )

        # Unsupported / abstain findings produce typed abstentions with
        # preserved frontiers, not plans.
        if typed_finding.disposition is DoctorRepairDisposition.ABSTAIN:
            return self._reject_compilation(
                typed_finding,
                typed_snapshot,
                reasons=(DoctorTacticianReasonCode.FINDING_ABSTAIN.value,),
                disposition=DoctorGoalCompilationDisposition.ABSTAINED,
                objective_id=objective_id,
                consumer_id=consumer_id,
                preserve_frontiers=True,
            )

        primary_consumer = (
            _identifier(consumer_id, "consumer_id")
            if consumer_id
            else (
                typed_finding.consumer_refs[0]
                if typed_finding.consumer_refs
                else f"consumer:{typed_finding.finding_id}"
            )
        )
        objective = (
            _identifier(objective_id, "objective_id")
            if objective_id
            else f"objective:{typed_finding.finding_id}"
        )
        roots = doctor_roots_to_program_logic_roots(
            doctor_roots,
            objective_id=objective,
            trace_id=typed_finding.trace_ref or f"trace:{typed_finding.finding_id}",
            change_id=typed_finding.change_ref or f"change:{typed_finding.finding_id}",
            consumer_id=primary_consumer,
        )

        reasons: list[str] = []
        excluded_expectation_refs: list[str] = []
        builder = ProgramLogicPremiseCorpusBuilder(roots)
        expectation_ids: list[str] = []
        observation_ids: list[str] = []
        hypothesis_ids: list[str] = []

        # --- Expectations (independent authority only) ---------------------
        expectation_refs = list(typed_finding.expected_behavior_refs)
        expectation_refs.extend(extra_expectation_refs)
        for ref in expectation_refs:
            ref_id = _identifier(ref, "expected_behavior_refs")
            if ref_id in typed_finding.observed_fact_refs:
                reasons.append(DoctorTacticianReasonCode.EXPECTATION_CONFLICT.value)
                excluded_expectation_refs.append(ref_id)
                continue
            if _is_cache_metadata_ref(ref_id):
                reasons.append(DoctorTacticianReasonCode.CACHE_METADATA_EXPECTATION.value)
                excluded_expectation_refs.append(ref_id)
                continue
            if _is_test_success_only_ref(ref_id):
                reasons.append(DoctorTacticianReasonCode.TEST_SUCCESS_EXPECTATION.value)
                excluded_expectation_refs.append(ref_id)
                continue
            if ref_id.startswith("candidate:") or ref_id.startswith("hypothesis:"):
                reasons.append(
                    DoctorTacticianReasonCode.CANDIDATE_AUTHORED_EXPECTATION.value
                )
                excluded_expectation_refs.append(ref_id)
                continue
            if ref_id.startswith("score:") or ref_id.startswith("embedding:"):
                reasons.append(DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value)
                excluded_expectation_refs.append(ref_id)
                continue
            if _PROMPT_DIRECTIVE_RE.search(ref_id):
                reasons.append(DoctorTacticianReasonCode.PROMPT_DIRECTIVE.value)
                excluded_expectation_refs.append(ref_id)
                continue
            premise_id = f"premise:expectation:{_stable_id('exp', ref_id)[-24:]}"
            builder.add_expectation(
                premise_id=premise_id,
                source_class=PremiseSourceClass.REVIEWED_CONTRACT,
                statement_ref=ref_id,
                lowering_ref=f"lower:expectation:{ref_id}",
                statement_digest=_digest(ref_id, typed_finding.finding_id),
                source_precedence=100,
                features=_features_for_finding(typed_finding),
                license_policy=_license(),
                contract_identity=ref_id,
                graph_identity=roots.graph_id,
                invalidator_refs=typed_finding.invalidation_refs
                or (roots.tree_id,),
            )
            expectation_ids.append(premise_id)

        if not expectation_ids:
            # Supported findings require independent expected-behavior authority.
            if typed_finding.disposition is DoctorRepairDisposition.SUPPORTED:
                reasons.append(DoctorTacticianReasonCode.EXPECTATION_MISSING.value)
                return self._reject_compilation(
                    typed_finding,
                    typed_snapshot,
                    reasons=tuple(dict.fromkeys(reasons)),
                    objective_id=objective,
                    consumer_id=primary_consumer,
                    roots=roots,
                )

        # --- Observed facts (static, never expectation) --------------------
        observation_refs = list(typed_finding.observed_fact_refs)
        observation_refs.extend(extra_observation_refs)
        for ref in observation_refs:
            ref_id = _identifier(ref, "observed_fact_refs")
            if ref_id in typed_finding.expected_behavior_refs:
                # Already rejected at finding construction for overlap, but
                # keep a defensive diagnostic.
                reasons.append(DoctorTacticianReasonCode.EXPECTATION_CONFLICT.value)
                continue
            premise_id = f"premise:observation:{_stable_id('obs', ref_id)[-24:]}"
            builder.add_static_fact(
                premise_id=premise_id,
                source_class=PremiseSourceClass.LOCAL_STATIC,
                statement_ref=ref_id,
                lowering_ref=f"lower:observation:{ref_id}",
                statement_digest=_digest(ref_id, typed_finding.finding_id),
                source_precedence=50,
                features=_features_for_finding(typed_finding),
                license_policy=_license(),
                graph_identity=roots.graph_id,
                invalidator_refs=typed_finding.invalidation_refs
                or (roots.tree_id,),
            )
            observation_ids.append(premise_id)

        # Also admit structural graph/symbol facts from the finding.
        for symbol in typed_finding.affected_symbol_refs:
            premise_id = f"premise:symbol:{_stable_id('sym', symbol)[-24:]}"
            builder.add_static_fact(
                premise_id=premise_id,
                source_class=PremiseSourceClass.PROGRAM_GRAPH,
                statement_ref=f"symbol-fact:{symbol}",
                lowering_ref=f"lower:symbol:{symbol}",
                statement_digest=_digest(symbol, "symbol"),
                source_precedence=55,
                features=PremiseFeatureSet(symbol_feature_refs=(symbol,)),
                license_policy=_license(),
                graph_identity=roots.graph_id,
                invalidator_refs=(roots.graph_id, roots.tree_id),
            )
            observation_ids.append(premise_id)

        # --- Candidates as hypotheses only ---------------------------------
        if len(candidates) > self._bounds.max_candidates:
            raise DoctorTacticianBoundsError("candidates exceed max_candidates")
        for index, candidate in enumerate(candidates):
            hyp_result = self._admit_candidate_hypothesis(
                builder=builder,
                roots=roots,
                finding=typed_finding,
                candidate=candidate,
                index=index,
            )
            if hyp_result is None:
                continue
            premise_id, reject_reason = hyp_result
            if reject_reason:
                reasons.append(reject_reason)
                continue
            if premise_id:
                hypothesis_ids.append(premise_id)

        # Pre-existing premise refs on the finding: re-classify carefully.
        for ref in typed_finding.premise_refs:
            if ref in expectation_ids or ref in observation_ids or ref in hypothesis_ids:
                continue
            # Unknown external premise refs become non-authoritative hypotheses.
            premise_id = f"premise:external:{_stable_id('ext', ref)[-24:]}"
            builder.add_hypothesis(
                premise_id=premise_id,
                source_class=PremiseSourceClass.HISTORY,
                statement_ref=ref,
                lowering_ref=f"lower:external:{ref}",
                statement_digest=_digest(ref, "external"),
                source_precedence=10,
                license_policy=_license(),
                graph_identity=roots.graph_id,
                invalidator_refs=(roots.tree_id,),
            )
            hypothesis_ids.append(premise_id)

        # Bound checks.
        if (
            len(expectation_ids) + len(observation_ids) + len(hypothesis_ids)
            > self._bounds.max_premises
        ):
            raise DoctorTacticianBoundsError("premises exceed max_premises")

        corpus = builder.build()

        # Reject self-validation / axiom-smuggling premises if any snuck in.
        for premise in corpus.premises:
            if premise.self_validation:
                reasons.append(DoctorTacticianReasonCode.SELF_VALIDATION.value)
            if premise.expectation_authority and (
                premise.source_class in _HYPOTHESIS_SOURCE_CLASSES
                or premise.authority is PremiseAuthority.HYPOTHESIS
            ):
                reasons.append(DoctorTacticianReasonCode.AXIOM_SMUGGLING.value)
            if premise.semantic_authority is not False:
                reasons.append(DoctorTacticianReasonCode.SEMANTIC_AUTHORITY_CLAIM.value)

        hard_rejects = {
            DoctorTacticianReasonCode.AXIOM_SMUGGLING.value,
            DoctorTacticianReasonCode.SELF_VALIDATION.value,
            DoctorTacticianReasonCode.SEMANTIC_AUTHORITY_CLAIM.value,
            DoctorTacticianReasonCode.PROMPT_DIRECTIVE.value,
            DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value,
            DoctorTacticianReasonCode.CANDIDATE_AUTHORED_EXPECTATION.value,
        }
        if hard_rejects.intersection(reasons):
            return self._reject_compilation(
                typed_finding,
                typed_snapshot,
                reasons=tuple(dict.fromkeys(reasons)),
                objective_id=objective,
                consumer_id=primary_consumer,
                roots=roots,
                corpus=corpus,
            )

        # --- Goals + facets + frontiers ------------------------------------
        subject = (
            typed_finding.affected_symbol_refs[0]
            if typed_finding.affected_symbol_refs
            else f"subject:{typed_finding.finding_id}"
        )
        consumers = list(typed_finding.consumer_refs) or [primary_consumer]
        # Every resolved consumer gets a goal; missing consumers on supported
        # findings with empty consumer list still use the primary consumer.
        goals: list[ProgramLogicGoal] = []
        required_facet_ids: list[str] = []
        unknown_frontiers = list(typed_finding.open_frontier_refs)

        required_open = [
            ref
            for ref in unknown_frontiers
            if ref.startswith("frontier:required:")
        ]
        if (
            required_open
            and typed_finding.disposition is DoctorRepairDisposition.SUPPORTED
        ):
            reasons.append(DoctorTacticianReasonCode.OPEN_REQUIRED_FRONTIER.value)

        for consumer in consumers[: self._bounds.max_goals]:
            facets = self._build_required_facets(
                subject_symbol_id=subject,
                consumer_id=consumer,
                finding=typed_finding,
            )
            required_facet_ids.extend(facet.facet_id for facet in facets)
            # Unsupported facets for open required frontiers.
            unsupported: list[LogicFacetRef] = []
            for frontier in required_open:
                unsupported.append(
                    LogicFacetRef(
                        facet_id=f"facet:frontier:{_stable_id('fr', frontier)[-16:]}",
                        kind=LogicFacetKind.INFORMATION,
                        subject_symbol_id=subject,
                        contract_ref=frontier,
                        unsupported=True,
                    )
                )

            goal_disposition = GoalDisposition.OPEN
            if typed_finding.disposition is DoctorRepairDisposition.APPROVAL_REQUIRED:
                goal_disposition = GoalDisposition.OPEN
            if required_open:
                goal_disposition = GoalDisposition.RESIDUAL

            goal = ProgramLogicGoal(
                roots=roots,
                goal_id=_stable_id(
                    "goal",
                    {
                        "finding": typed_finding.finding_id,
                        "consumer": consumer,
                        "subject": subject,
                    },
                ),
                family=GoalFamily.BEHAVIOR,
                disposition=goal_disposition,
                positive_statement_ref=(
                    typed_finding.expected_behavior_refs[0]
                    if typed_finding.expected_behavior_refs
                    else f"stmt:repair:{typed_finding.finding_id}"
                ),
                affected_symbol_ids=tuple(typed_finding.affected_symbol_refs)
                or (subject,),
                source_refs=tuple(
                    list(expectation_ids[:8]) + list(observation_ids[:8])
                ),
                required_facets=tuple(facets),
                unsupported_facets=tuple(unsupported),
                assumption_refs=tuple(observation_ids[:16]),
                assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
                proof_status=ProofStatus.UNPROVED,
                bound_refs=tuple(
                    sorted(
                        {
                            typed_finding.finding_id,
                            typed_finding.snapshot_id,
                            roots.tree_id,
                            roots.corpus_id,
                            *unknown_frontiers[:16],
                        }
                    )
                ),
                invalidation_refs=tuple(
                    sorted(
                        set(typed_finding.invalidation_refs)
                        | {
                            roots.tree_id,
                            roots.corpus_id,
                            roots.policy_id,
                            typed_finding.snapshot_id,
                        }
                    )
                ),
            )
            goals.append(goal)

        if not goals:
            reasons.append(DoctorTacticianReasonCode.NO_GOALS.value)
            return self._reject_compilation(
                typed_finding,
                typed_snapshot,
                reasons=tuple(dict.fromkeys(reasons)),
                objective_id=objective,
                consumer_id=primary_consumer,
                roots=roots,
                corpus=corpus,
            )

        # Consumers must be dispositioned later; for compilation we only
        # require that every finding consumer produced a goal.
        missing_consumers = set(typed_finding.consumer_refs) - {
            # Goals bind consumer via roots.consumer_id for primary; multi-
            # consumer findings encode consumer in goal_id material.
            primary_consumer
        }
        # Multi-consumer findings: ensure we created one goal per consumer.
        if typed_finding.consumer_refs and len(goals) < len(
            set(typed_finding.consumer_refs)
        ):
            reasons.append(DoctorTacticianReasonCode.MISSING_CONSUMER.value)

        # Facet inventory completeness.
        for goal in goals:
            kinds = {facet.kind for facet in goal.required_facets}
            missing_kinds = set(REQUIRED_FACET_KINDS) - kinds
            if missing_kinds:
                reasons.append(DoctorTacticianReasonCode.MISSING_FACET.value)

        disposition = DoctorGoalCompilationDisposition.COMPLETE
        if reasons or required_open or any(
            g.disposition is GoalDisposition.RESIDUAL for g in goals
        ):
            disposition = DoctorGoalCompilationDisposition.PARTIAL
        if not expectation_ids and typed_finding.disposition is not DoctorRepairDisposition.SUPPORTED:
            disposition = DoctorGoalCompilationDisposition.PARTIAL

        compilation_id = _stable_id(
            "compilation",
            {
                "finding": typed_finding.finding_id,
                "snapshot": typed_finding.snapshot_id,
                "goals": [g.goal_id for g in goals],
                "corpus": corpus.content_id,
                "roots": roots.content_id,
            },
        )
        invalidation = tuple(
            sorted(
                set(typed_finding.invalidation_refs)
                | {
                    roots.tree_id,
                    roots.corpus_id,
                    roots.policy_id,
                    typed_finding.snapshot_id,
                    corpus.content_id,
                }
            )
        )
        return DoctorGoalCompilation(
            roots=roots,
            compilation_id=compilation_id,
            finding_id=typed_finding.finding_id,
            snapshot_id=typed_finding.snapshot_id,
            disposition=disposition,
            goals=tuple(goals),
            corpus=corpus,
            required_facet_ids=tuple(sorted(set(required_facet_ids))),
            unknown_frontier_refs=tuple(sorted(set(unknown_frontiers))),
            excluded_expectation_refs=tuple(sorted(set(excluded_expectation_refs))),
            reason_codes=tuple(dict.fromkeys(reasons)),
            selected_expectation_ids=tuple(sorted(set(expectation_ids))),
            selected_observation_ids=tuple(sorted(set(observation_ids))),
            selected_hypothesis_ids=tuple(sorted(set(hypothesis_ids))),
            consumer_ids=tuple(sorted(set(consumers))),
            semantic_authority=False,
            invalidation_refs=invalidation,
        )

    def _admit_candidate_hypothesis(
        self,
        *,
        builder: ProgramLogicPremiseCorpusBuilder,
        roots: ProgramLogicAuthorityRoots,
        finding: DeterministicDoctorFinding,
        candidate: Any,
        index: int,
    ) -> tuple[str, str] | None:
        """Return (premise_id, reason) or (premise_id, '') or None to skip."""
        payload = _mapping(candidate)
        _assert_body_free(payload, "candidate")
        if _is_score_authority_payload(payload):
            return "", DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value
        if payload.get("semantic_authority") is True:
            return "", DoctorTacticianReasonCode.SEMANTIC_AUTHORITY_CLAIM.value
        if payload.get("expectation_authority") is True:
            return (
                "",
                DoctorTacticianReasonCode.CANDIDATE_AUTHORED_EXPECTATION.value,
            )
        if payload.get("self_validation") is True:
            return "", DoctorTacticianReasonCode.SELF_VALIDATION.value

        # Nested candidate.candidate (DoctorCandidateNomination).
        nested = payload.get("candidate")
        if nested is not None:
            nested_map = _mapping(nested)
            payload = {**nested_map, **{k: v for k, v in payload.items() if k != "candidate"}}

        ref = (
            payload.get("candidate_ref")
            or payload.get("symbol_id")
            or payload.get("premise_id")
            or payload.get("ref")
            or f"candidate:{finding.finding_id}:{index}"
        )
        ref_id = _identifier(ref, "candidate_ref")
        signal = str(
            payload.get("primary_signal")
            or payload.get("signal")
            or payload.get("source_class")
            or payload.get("kind")
            or "candidate"
        )
        source_class = _source_class_from_label(signal)
        # Force candidate / approximate material into hypothesis classes.
        if is_expectation_source_class(source_class):
            # Exact contract signal may exist, but still cannot author expectation
            # *from candidate code* — demote to static fact if exact, else hypothesis.
            if signal in {"exact_contract", "exact_symbol", "exact_graph", "exact_value"}:
                source_class = {
                    "exact_contract": PremiseSourceClass.LOCAL_STATIC,
                    "exact_symbol": PremiseSourceClass.LOCAL_STATIC,
                    "exact_graph": PremiseSourceClass.PROGRAM_GRAPH,
                    "exact_value": PremiseSourceClass.VALUE_PROVENANCE,
                }[signal]
            else:
                source_class = PremiseSourceClass.CANDIDATE_IMPLEMENTATION

        if _is_cache_metadata_ref(ref_id) or signal in {
            "cache",
            "cache_metadata",
            "proof_cache",
        }:
            source_class = PremiseSourceClass.COMMENT
        if _is_test_success_only_ref(ref_id) or signal in {
            "test_success",
            "mere_success",
        }:
            source_class = PremiseSourceClass.COMMENT

        premise_id = f"premise:hypothesis:{_stable_id('hyp', ref_id)[-24:]}"
        # Exact structural candidates may be static facts; approximate stay hypotheses.
        if source_class in _STATIC_FACT_SOURCE_CLASSES and signal.startswith("exact_"):
            builder.add_static_fact(
                premise_id=premise_id,
                source_class=source_class,
                statement_ref=f"candidate-fact:{ref_id}",
                lowering_ref=f"lower:candidate:{ref_id}",
                statement_digest=_digest(ref_id, "candidate-static"),
                source_precedence=40,
                features=PremiseFeatureSet(
                    symbol_feature_refs=tuple(
                        filter(
                            None,
                            (
                                str(payload.get("symbol_id") or ""),
                            ),
                        )
                    ),
                ),
                license_policy=_license(),
                graph_identity=roots.graph_id,
                invalidator_refs=(roots.tree_id, finding.snapshot_id),
            )
        else:
            if source_class not in _HYPOTHESIS_SOURCE_CLASSES:
                source_class = PremiseSourceClass.CANDIDATE_IMPLEMENTATION
            builder.add_hypothesis(
                premise_id=premise_id,
                source_class=source_class,
                statement_ref=f"candidate-hypothesis:{ref_id}",
                lowering_ref=f"lower:candidate:{ref_id}",
                statement_digest=_digest(ref_id, "candidate-hypothesis"),
                source_precedence=10,
                features=PremiseFeatureSet(
                    symbol_feature_refs=tuple(
                        filter(
                            None,
                            (
                                str(payload.get("symbol_id") or ""),
                            ),
                        )
                    ),
                ),
                license_policy=_license(),
                graph_identity=roots.graph_id,
                invalidator_refs=(roots.tree_id, finding.snapshot_id),
            )
        return premise_id, ""

    def _build_required_facets(
        self,
        *,
        subject_symbol_id: str,
        consumer_id: str,
        finding: DeterministicDoctorFinding,
    ) -> list[LogicFacetRef]:
        facets: list[LogicFacetRef] = []
        # Token map avoids secret-marker false positives (e.g. "authorization").
        kind_token = {
            LogicFacetKind.TYPE: "type",
            LogicFacetKind.ERROR: "error",
            LogicFacetKind.EFFECT: "effect",
            LogicFacetKind.AUTHORIZATION: "authz",
            LogicFacetKind.RESOURCE: "resource",
            LogicFacetKind.STATE: "state",
            LogicFacetKind.SCHEMA: "schema",
            LogicFacetKind.PLACEMENT: "placement",
            LogicFacetKind.INFORMATION: "information",
            LogicFacetKind.MEMORY: "memory",
        }
        for kind in REQUIRED_FACET_KINDS:
            token = kind_token[kind]
            facet_id = f"facet:{token}:{subject_symbol_id}:{consumer_id}"
            # Keep facet ids compact and free of secret-marker substrings.
            if len(facet_id) > MAX_REF_BYTES:
                facet_id = f"facet:{token}:{_stable_id('f', facet_id)[-32:]}"
            facets.append(
                LogicFacetRef(
                    facet_id=facet_id,
                    kind=kind,
                    subject_symbol_id=subject_symbol_id,
                    contract_ref=(
                        finding.expected_behavior_refs[0]
                        if finding.expected_behavior_refs
                        else ""
                    ),
                    unsupported=False,
                )
            )
        return facets

    def _decode_finding(
        self, finding: DeterministicDoctorFinding | Mapping[str, Any]
    ) -> DeterministicDoctorFinding:
        if isinstance(finding, DeterministicDoctorFinding):
            return finding
        if isinstance(finding, Mapping):
            if "schema" in finding:
                return DeterministicDoctorFinding.from_dict(finding)
            return DeterministicDoctorFinding(**finding)
        raise DoctorTacticianError("finding must be DeterministicDoctorFinding")

    def _decode_snapshot(
        self, snapshot: DoctorEvidenceSnapshot | Mapping[str, Any] | None
    ) -> DoctorEvidenceSnapshot | None:
        if snapshot is None:
            return None
        if isinstance(snapshot, DoctorEvidenceSnapshot):
            return snapshot
        if isinstance(snapshot, Mapping):
            if "schema" in snapshot:
                return DoctorEvidenceSnapshot.from_dict(snapshot)
            return DoctorEvidenceSnapshot(**snapshot)
        raise DoctorTacticianError("snapshot must be DoctorEvidenceSnapshot")

    def _reject_compilation(
        self,
        finding: DeterministicDoctorFinding,
        snapshot: DoctorEvidenceSnapshot | None,
        *,
        reasons: Sequence[str],
        disposition: DoctorGoalCompilationDisposition = DoctorGoalCompilationDisposition.REJECTED,
        objective_id: str = "",
        consumer_id: str = "",
        roots: ProgramLogicAuthorityRoots | None = None,
        corpus: ProgramLogicPremiseCorpus | None = None,
        preserve_frontiers: bool = False,
    ) -> DoctorGoalCompilation:
        primary_consumer = (
            consumer_id
            or (
                finding.consumer_refs[0]
                if finding.consumer_refs
                else f"consumer:{finding.finding_id}"
            )
        )
        objective = objective_id or f"objective:{finding.finding_id}"
        if roots is None:
            roots = doctor_roots_to_program_logic_roots(
                finding.roots,
                objective_id=objective,
                trace_id=finding.trace_ref or f"trace:{finding.finding_id}",
                change_id=finding.change_ref or f"change:{finding.finding_id}",
                consumer_id=primary_consumer,
            )
        if corpus is None:
            corpus = ProgramLogicPremiseCorpusBuilder(roots).build()
        frontiers = (
            tuple(finding.open_frontier_refs) if preserve_frontiers else ()
        )
        compilation_id = _stable_id(
            "compilation",
            {
                "finding": finding.finding_id,
                "disposition": disposition.value,
                "reasons": list(reasons),
                "roots": roots.content_id,
            },
        )
        return DoctorGoalCompilation(
            roots=roots,
            compilation_id=compilation_id,
            finding_id=finding.finding_id,
            snapshot_id=finding.snapshot_id,
            disposition=disposition,
            goals=(),
            corpus=corpus,
            required_facet_ids=(),
            unknown_frontier_refs=frontiers,
            excluded_expectation_refs=(),
            reason_codes=tuple(reasons),
            selected_expectation_ids=(),
            selected_observation_ids=(),
            selected_hypothesis_ids=(),
            consumer_ids=tuple(finding.consumer_refs) or (primary_consumer,),
            semantic_authority=False,
            invalidation_refs=tuple(
                sorted(
                    set(finding.invalidation_refs)
                    | {roots.tree_id, roots.corpus_id, finding.snapshot_id}
                )
            ),
        )


# ---------------------------------------------------------------------------
# Deterministic local planner (no LLM; exact-first)
# ---------------------------------------------------------------------------


class DeterministicLocalDoctorPlanner:
    """Build a finite exact-first :class:`TacticianSearchPlan` without a model.

    This is the hermetic deterministic planner used when the optional datasets
    Tactician is unavailable or when tests inject a pure local path. It never
    adds axioms, never selects write paths, and never claims semantic authority.
    """

    def __init__(self, bounds: DoctorTacticianBounds | None = None) -> None:
        self._bounds = bounds or DoctorTacticianBounds()

    def plan(
        self,
        *,
        roots: ProgramLogicAuthorityRoots,
        goals: Sequence[ProgramLogicGoal],
        corpus: ProgramLogicPremiseCorpus,
        compilation: DoctorGoalCompilation | None = None,
    ) -> TacticianSearchPlan:
        if not goals:
            raise DoctorTacticianError("local planner requires at least one goal")
        if roots.content_id != corpus.roots.content_id:
            raise DoctorTacticianAuthorityError("corpus roots must match plan roots")
        for goal in goals:
            if goal.roots.content_id != roots.content_id:
                raise DoctorTacticianAuthorityError("goal roots must match plan roots")

        # Partition premises by authority class.
        selected: list[str] = []
        excluded: list[str] = []
        exclusion_rationales: list[str] = []
        route_set: list[SourceRouteKind] = []

        def add_route(route: SourceRouteKind) -> None:
            if route not in route_set:
                route_set.append(route)

        for premise in sorted(corpus.premises, key=lambda item: item.premise_id):
            if premise.self_validation:
                raise DoctorTacticianAuthorityError(
                    "self-validating premises cannot enter a search plan"
                )
            if premise.semantic_authority is not False:
                raise DoctorTacticianSafetyError(
                    "premises cannot claim semantic authority"
                )
            route = _SOURCE_CLASS_TO_ROUTE.get(
                premise.source_class, SourceRouteKind.LOCAL_STATIC
            )
            if route is SourceRouteKind.LLM:
                # Deterministic mode: never admit LLM routes or model hypotheses.
                excluded.append(premise.premise_id)
                exclusion_rationales.append(
                    f"rationale:deny-llm:{premise.premise_id}"
                )
                continue
            if (
                premise.authority is PremiseAuthority.HYPOTHESIS
                or premise.source_class in _HYPOTHESIS_SOURCE_CLASSES
            ):
                # Approximate / nominating material stays selectable only as
                # nomination after exact routes, never as expectation.
                if not self._bounds.allow_approximate_routes and route in _NOMINATING_ROUTES:
                    excluded.append(premise.premise_id)
                    exclusion_rationales.append(
                        f"rationale:deny-approximate:{premise.premise_id}"
                    )
                    continue
                # Keep nominating hypotheses *excluded* from axiom selection;
                # they remain visible as nominating routes only.
                excluded.append(premise.premise_id)
                exclusion_rationales.append(
                    f"rationale:hypothesis-nomination-only:{premise.premise_id}"
                )
                add_route(route)
                continue
            # Expectation + static facts are selected axioms for search.
            selected.append(premise.premise_id)
            add_route(route)

        # Exact-first order: authoritative routes first, then remaining admitted.
        ordered_routes = [
            route
            for route in EXACT_FIRST_SOURCE_ROUTES
            if route in route_set and route is not SourceRouteKind.LLM
        ]
        # Ensure at least one exact local route when we have selected premises.
        if not ordered_routes:
            ordered_routes = [SourceRouteKind.LOCAL_STATIC]
        if self._bounds.require_exact_before_approximate:
            last_auth = -1
            first_nom = len(ordered_routes)
            for index, route in enumerate(ordered_routes):
                if route in _AUTHORITATIVE_ROUTES:
                    last_auth = max(last_auth, index)
                if route in _NOMINATING_ROUTES:
                    first_nom = min(first_nom, index)
            if last_auth >= 0 and first_nom < len(ordered_routes) and last_auth > first_nom:
                raise DoctorTacticianAuthorityError(
                    "authoritative exact routes must precede approximate nominations"
                )
        if len(ordered_routes) > self._bounds.max_routes:
            raise DoctorTacticianBoundsError("ordered source routes exceed max_routes")

        # One subgoal per required facet (covers gate facet inventory).
        subgoals: list[LogicSubgoal] = []
        for goal in goals:
            parent = ""
            for facet in goal.required_facets:
                if facet.unsupported:
                    continue
                # Derive a compact token from the facet id (already secret-safe).
                kind_token = facet.facet_id.split(":")[1] if ":" in facet.facet_id else "facet"
                subgoal_id = f"subgoal:{_stable_id('sg', goal.goal_id + kind_token)[-40:]}"
                # Prefer authoritative local route for facet coverage.
                route = SourceRouteKind.LOCAL_STATIC
                if SourceRouteKind.REVIEWED_CONTRACT in ordered_routes:
                    route = SourceRouteKind.REVIEWED_CONTRACT
                subgoals.append(
                    LogicSubgoal(
                        subgoal_id=subgoal_id,
                        goal_id=goal.goal_id,
                        disposition=SubgoalDisposition.PLANNED,
                        claim_ref=facet.facet_id,
                        parent_subgoal_id=parent,
                        depends_on=(parent,) if parent else (),
                        source_route=route,
                        source_authority=_route_authority(route),
                        proof_status=ProofStatus.UNPROVED,
                        score_millipercent=0,
                    )
                )
                parent = subgoal_id
            # Residual / unknown frontier subgoals.
            for frontier in (compilation.unknown_frontier_refs if compilation else ()):
                subgoal_id = f"subgoal:frontier:{_stable_id('fr', frontier + goal.goal_id)[-32:]}"
                subgoals.append(
                    LogicSubgoal(
                        subgoal_id=subgoal_id,
                        goal_id=goal.goal_id,
                        disposition=SubgoalDisposition.RESIDUAL,
                        claim_ref=frontier,
                        source_route=SourceRouteKind.LOCAL_STATIC,
                        source_authority=SourceAuthorityClass.AUTHORITATIVE,
                        proof_status=ProofStatus.UNPROVED,
                        score_millipercent=0,
                    )
                )
            if len(subgoals) > self._bounds.max_subgoals:
                raise DoctorTacticianBoundsError("subgoals exceed max_subgoals")

        # Cycle check (defensive).
        adjacency = {
            item.subgoal_id: list(item.depends_on)
            + ([item.parent_subgoal_id] if item.parent_subgoal_id else [])
            for item in subgoals
        }
        if _detect_cycle(adjacency):
            raise DoctorTacticianAuthorityError("subgoal dependency graph contains a cycle")

        plan_id = _stable_id(
            "plan",
            {
                "roots": roots.content_id,
                "goals": [g.goal_id for g in goals],
                "selected": selected,
                "excluded": excluded,
                "routes": [r.value for r in ordered_routes],
            },
        )
        return TacticianSearchPlan(
            roots=roots,
            plan_id=plan_id,
            goal_ids=tuple(g.goal_id for g in goals),
            ordered_source_routes=tuple(ordered_routes),
            query_refs=tuple(
                sorted(
                    {
                        f"query:{g.goal_id}"
                        for g in goals
                    }
                )
            ),
            selected_premise_ids=tuple(sorted(set(selected))),
            excluded_premise_ids=tuple(sorted(set(excluded))),
            exclusion_rationale_refs=tuple(sorted(set(exclusion_rationales))),
            subgoals=tuple(subgoals),
            stop_policy_ref="stop:deterministic-doctor-tactician@1",
            escalation_policy_ref="escalation:deterministic-doctor-tactician@1",
            abstention_policy_ref="abstention:deterministic-doctor-tactician@1",
            resource_policy_ref="resource:deterministic-doctor-tactician@1",
            planner_id=PLANNER_ID,
            model_id="",
            config_id="config:deterministic-doctor-tactician@1",
            semantic_authority=False,
            invalidation_refs=tuple(
                sorted(
                    {
                        roots.tree_id,
                        roots.corpus_id,
                        roots.policy_id,
                        corpus.content_id,
                        *(compilation.invalidation_refs if compilation else ()),
                    }
                )
            ),
        )


# ---------------------------------------------------------------------------
# Optional provider protocol
# ---------------------------------------------------------------------------


class DoctorTacticianProvider(Protocol):
    """Minimal protocol for an injectable Code Tactician provider."""

    def plan(self, request: Any) -> Any:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# DeterministicDoctorTactician
# ---------------------------------------------------------------------------


class DeterministicDoctorTactician:
    """Compile findings into independent goals, then plan and gate search.

    The Tactician stage never authors expectations. It only orders exact-first
    proof/synthesis search over already-compiled premises and goals.
    """

    def __init__(
        self,
        *,
        bounds: DoctorTacticianBounds | Mapping[str, Any] | None = None,
        goal_compiler: DoctorRepairGoalCompiler | None = None,
        plan_gate: TacticianPlanGate | None = None,
        provider: DoctorTacticianProvider | None = None,
        use_provider: bool = False,
        local_planner: DeterministicLocalDoctorPlanner | None = None,
    ) -> None:
        if bounds is None:
            self._bounds = DoctorTacticianBounds()
        elif isinstance(bounds, DoctorTacticianBounds):
            self._bounds = bounds
        elif isinstance(bounds, Mapping):
            self._bounds = DoctorTacticianBounds.from_dict(bounds)
        else:
            raise DoctorTacticianError("bounds must be DoctorTacticianBounds")
        self._compiler = goal_compiler or DoctorRepairGoalCompiler(self._bounds)
        gate_bounds = TacticianPlanGateBounds(
            max_subgoals=self._bounds.max_subgoals,
            max_premises=self._bounds.max_premises,
            max_routes=self._bounds.max_routes,
            allow_model_hypothesis=False,
            allow_approximate_routes=self._bounds.allow_approximate_routes,
            semantic_authority=False,
        )
        self._gate = plan_gate or TacticianPlanGate(gate_bounds)
        self._provider = provider
        self._use_provider = bool(use_provider)
        self._local_planner = local_planner or DeterministicLocalDoctorPlanner(
            self._bounds
        )

    @property
    def bounds(self) -> DoctorTacticianBounds:
        return self._bounds

    @property
    def goal_compiler(self) -> DoctorRepairGoalCompiler:
        return self._compiler

    def compile_finding(
        self,
        finding: DeterministicDoctorFinding | Mapping[str, Any],
        **kwargs: Any,
    ) -> DoctorGoalCompilation:
        """Compile independent goals/premises without planning."""
        return self._compiler.compile(finding, **kwargs)

    def plan_finding(
        self,
        finding: DeterministicDoctorFinding | Mapping[str, Any],
        *,
        snapshot: DoctorEvidenceSnapshot | Mapping[str, Any] | None = None,
        candidates: Sequence[Any] = (),
        current_roots: DoctorAuthorityRoots | Mapping[str, Any] | None = None,
        compilation: DoctorGoalCompilation | None = None,
        objective_id: str = "",
        consumer_id: str = "",
        score_override_attempt: bool = False,
    ) -> DoctorTacticianPlanReceipt:
        """Compile (unless provided), plan exact-first, and gate the result."""
        if score_override_attempt:
            # Explicit rejection of score-based authority promotion.
            typed = self._compiler._decode_finding(finding)
            roots = doctor_roots_to_program_logic_roots(
                typed.roots,
                objective_id=objective_id or f"objective:{typed.finding_id}",
                trace_id=typed.trace_ref or f"trace:{typed.finding_id}",
                change_id=typed.change_ref or f"change:{typed.finding_id}",
                consumer_id=consumer_id
                or (
                    typed.consumer_refs[0]
                    if typed.consumer_refs
                    else f"consumer:{typed.finding_id}"
                ),
            )
            return DoctorTacticianPlanReceipt(
                roots=roots,
                receipt_id=_stable_id(
                    "receipt",
                    {
                        "finding": typed.finding_id,
                        "reason": DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value,
                    },
                ),
                finding_id=typed.finding_id,
                snapshot_id=typed.snapshot_id,
                compilation_id="compilation:none",
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reason_codes=(DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value,),
                semantic_authority=False,
                model_invocation_count=0,
                llm_route_present=False,
                invalidation_refs=tuple(
                    sorted(set(typed.invalidation_refs) | {roots.tree_id})
                ),
            )

        if compilation is None:
            compilation = self._compiler.compile(
                finding,
                snapshot=snapshot,
                candidates=candidates,
                current_roots=current_roots,
                objective_id=objective_id,
                consumer_id=consumer_id,
            )

        if compilation.disposition is DoctorGoalCompilationDisposition.REJECTED:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reasons=compilation.reason_codes
                or (DoctorTacticianReasonCode.MALFORMED_INPUT.value,),
            )
        if compilation.disposition is DoctorGoalCompilationDisposition.ABSTAINED:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.ABSTAINED,
                reasons=compilation.reason_codes
                or (DoctorTacticianReasonCode.FINDING_ABSTAIN.value,),
            )
        if not compilation.goals:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.ABSTAINED,
                reasons=(DoctorTacticianReasonCode.NO_GOALS.value,),
            )

        # Prefer local deterministic planner (always available, no model).
        # Optional provider path is opt-in and still gated + LLM-free.
        provider_status = "local_deterministic"
        plan: TacticianSearchPlan | None = None
        try:
            if self._use_provider:
                plan, provider_status = self._plan_via_provider(compilation)
            if plan is None:
                plan = self._local_planner.plan(
                    roots=compilation.roots,
                    goals=compilation.goals,
                    corpus=compilation.corpus,
                    compilation=compilation,
                )
                provider_status = "local_deterministic"
        except DoctorTacticianError as exc:
            code = DoctorTacticianReasonCode.PROVIDER_REFUSED.value
            message = str(exc).casefold()
            if "cycle" in message:
                code = DoctorTacticianReasonCode.CYCLE.value
            elif "llm" in message:
                code = DoctorTacticianReasonCode.LLM_ROUTE.value
            elif "unbounded" in message or "exceed" in message:
                code = DoctorTacticianReasonCode.UNBOUNDED_ROUTE.value
            elif "root" in message:
                code = DoctorTacticianReasonCode.CHANGED_ROOTS.value
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reasons=(code,),
                provider_status=provider_status,
            )

        # Hard post-conditions before gate.
        if plan.semantic_authority is not False:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reasons=(DoctorTacticianReasonCode.SEMANTIC_AUTHORITY_CLAIM.value,),
                plan=plan,
                provider_status=provider_status,
            )
        if SourceRouteKind.LLM in plan.ordered_source_routes:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reasons=(DoctorTacticianReasonCode.LLM_ROUTE.value,),
                plan=plan,
                provider_status=provider_status,
            )
        if not self._routes_exact_first(plan.ordered_source_routes):
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.REJECTED,
                reasons=(DoctorTacticianReasonCode.UNBOUNDED_ROUTE.value,),
                plan=plan,
                provider_status=provider_status,
            )

        try:
            gate_receipt = self._gate.evaluate(
                plan=plan,
                goals=compilation.goals,
                corpus=compilation.corpus,
                current_roots=compilation.roots,
                score_override_attempt=False,
            )
        except TacticianPlanGateError as exc:
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.GATE_REJECTED,
                reasons=(DoctorTacticianReasonCode.GATE_REJECTED.value, str(exc)[:200]),
                plan=plan,
                provider_status=provider_status,
            )

        if gate_receipt.disposition is TacticianPlanGateDisposition.REJECTED:
            codes = [DoctorTacticianReasonCode.GATE_REJECTED.value]
            codes.extend(item.value for item in gate_receipt.reasons)
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.GATE_REJECTED,
                reasons=tuple(codes),
                plan=plan,
                gate_receipt=gate_receipt,
                provider_status=provider_status,
            )
        if gate_receipt.disposition is TacticianPlanGateDisposition.ABSTAINED:
            codes = [DoctorTacticianReasonCode.GATE_ABSTAINED.value]
            codes.extend(item.value for item in gate_receipt.reasons)
            return self._receipt_from_compilation(
                compilation,
                disposition=DoctorTacticianPlanDisposition.GATE_ABSTAINED,
                reasons=tuple(codes),
                plan=plan,
                gate_receipt=gate_receipt,
                provider_status=provider_status,
            )

        # Preserve partial compilation diagnostics while still emitting a plan.
        reason_codes = list(compilation.reason_codes)
        if compilation.disposition is DoctorGoalCompilationDisposition.COMPLETE:
            reason_codes = [DoctorTacticianReasonCode.OK.value, *reason_codes]

        return self._receipt_from_compilation(
            compilation,
            disposition=DoctorTacticianPlanDisposition.PLANNED,
            reasons=tuple(dict.fromkeys(reason_codes))
            or (DoctorTacticianReasonCode.OK.value,),
            plan=plan,
            gate_receipt=gate_receipt,
            provider_status=provider_status,
        )

    def plan_findings(
        self,
        findings: Sequence[DeterministicDoctorFinding | Mapping[str, Any]],
        **kwargs: Any,
    ) -> tuple[DoctorTacticianPlanReceipt, ...]:
        """Plan each finding independently (no cross-finding axiom smuggling)."""
        if len(findings) > MAX_GOALS_PER_FINDING:
            raise DoctorTacticianBoundsError("findings exceed hard bound")
        return tuple(self.plan_finding(item, **kwargs) for item in findings)

    def _plan_via_provider(
        self, compilation: DoctorGoalCompilation
    ) -> tuple[TacticianSearchPlan | None, str]:
        provider = self._provider
        if provider is None:
            provider = self._load_default_provider()
        if provider is None:
            return None, "provider_unavailable"
        try:
            # Lazy import of request types to keep planning package free of a
            # hard integrations dependency at import time.
            from ..integrations.ipfs_datasets_tactician_provider import (
                CODE_SOURCE_PRECEDENCE,
                CodeSourceType,
                CodeTacticianPolicy,
                CodeTacticianRequest,
                CodeTacticianStatus,
            )
        except Exception:
            return None, "provider_import_failed"

        denied = (CodeSourceType.MODEL_HYPOTHESIS,)
        policy = CodeTacticianPolicy(
            policy_id="code-tactician.policy.deterministic-doctor@1",
            source_class_order=CODE_SOURCE_PRECEDENCE,
            max_sources=min(32, self._bounds.max_premises),
            max_routes=self._bounds.max_routes,
            max_subgoals=self._bounds.max_subgoals,
            max_premises=self._bounds.max_premises,
            denied_source_types=denied,
            allow_approximate_routes=self._bounds.allow_approximate_routes,
            allow_model_hypothesis=False,
            require_local_before_approximate=True,
            network_allowed=False,
            write_allowed=False,
            proof_execution_allowed=False,
            semantic_authority=False,
        )
        request = CodeTacticianRequest(
            roots=compilation.roots,
            goals=compilation.goals,
            corpus=compilation.corpus,
            policy=policy,
            expected_roots=compilation.roots,
            admitted_tree_id=compilation.roots.tree_id,
            admitted_corpus_id=compilation.roots.corpus_id,
            information_demands=tuple(
                item
                for item in CODE_SOURCE_PRECEDENCE
                if item is not CodeSourceType.MODEL_HYPOTHESIS
            )[:8],
            metadata={
                "finding_id": compilation.finding_id,
                "compilation_id": compilation.compilation_id,
                "semantic_authority": False,
            },
        )
        response = provider.plan(request)
        status = getattr(response, "status", None)
        status_value = getattr(status, "value", str(status or ""))
        if getattr(response, "semantic_authority", False) is not False:
            raise DoctorTacticianSafetyError(
                "provider response claimed semantic_authority"
            )
        if status is CodeTacticianStatus.PLANNED or status_value == "planned":
            plan = getattr(response, "plan", None)
            if plan is None:
                return None, "provider_planned_without_plan"
            if not isinstance(plan, TacticianSearchPlan):
                if isinstance(plan, Mapping):
                    plan = TacticianSearchPlan.from_dict(plan)
                else:
                    raise DoctorTacticianError("provider plan is not TacticianSearchPlan")
            return plan, status_value
        if status_value in {"unavailable", "abstained"}:
            return None, status_value
        raise DoctorTacticianError(
            f"provider refused planning: {status_value}"
        )

    def _load_default_provider(self) -> DoctorTacticianProvider | None:
        try:
            from ..integrations.ipfs_datasets_tactician_provider import (
                IpfsDatasetsTacticianProvider,
            )
        except Exception:
            return None
        # Inject a deterministic local fake planner so optional package absence
        # does not force abstention when use_provider=True without datasets.
        local = self._local_planner

        class _LocalAsGeneric:
            def plan(self, goal: Any, sources: Sequence[Any], policy: Any = None, **_: Any) -> Any:
                # The provider will project this fake plan; we return a minimal
                # structure mirroring the fake used in provider tests.
                goal_id = (
                    goal.get("goal_id")
                    if isinstance(goal, Mapping)
                    else getattr(goal, "goal_id", "goal:unknown")
                )

                @dataclass(frozen=True)
                class _Route:
                    route_id: str
                    source_id: str
                    source_class: str
                    stage_index: int
                    disposition: str
                    rationale: str

                @dataclass(frozen=True)
                class _Plan:
                    plan_id: str
                    goal_id: str
                    selected_routes: list
                    excluded_routes: list
                    subgoals: list
                    stop_disposition: str = "continue"
                    planner_id: str = PLANNER_ID
                    semantic_authority: bool = False

                selected = []
                for index, source in enumerate(list(sources)[:16]):
                    source_id = (
                        source["source_id"]
                        if isinstance(source, Mapping)
                        else getattr(source, "source_id", f"source:{index}")
                    )
                    source_class = (
                        source["source_class"]
                        if isinstance(source, Mapping)
                        else getattr(source, "source_class", "local_static")
                    )
                    if str(source_class) in {"model_hypothesis", "llm"}:
                        continue
                    selected.append(
                        _Route(
                            route_id=f"route:{source_id}",
                            source_id=str(source_id),
                            source_class=str(source_class),
                            stage_index=len(selected),
                            disposition="selected",
                            rationale="deterministic local projection",
                        )
                    )
                return _Plan(
                    plan_id=f"generic-plan:{goal_id}",
                    goal_id=str(goal_id),
                    selected_routes=selected,
                    excluded_routes=[],
                    subgoals=[],
                )

        return IpfsDatasetsTacticianProvider(planner_factory=lambda: _LocalAsGeneric())

    def _routes_exact_first(self, routes: Sequence[SourceRouteKind]) -> bool:
        last_auth = -1
        first_nom = len(routes)
        for index, route in enumerate(routes):
            if route is SourceRouteKind.LLM:
                return False
            if route in _AUTHORITATIVE_ROUTES:
                last_auth = max(last_auth, index)
            if route in _NOMINATING_ROUTES:
                first_nom = min(first_nom, index)
        if not self._bounds.require_exact_before_approximate:
            return True
        if last_auth >= 0 and first_nom < len(routes) and last_auth > first_nom:
            return False
        return True

    def _receipt_from_compilation(
        self,
        compilation: DoctorGoalCompilation,
        *,
        disposition: DoctorTacticianPlanDisposition,
        reasons: Sequence[str],
        plan: TacticianSearchPlan | None = None,
        gate_receipt: TacticianPlanGateReceipt | None = None,
        provider_status: str = "",
    ) -> DoctorTacticianPlanReceipt:
        ordered_routes = (
            tuple(route.value for route in plan.ordered_source_routes)
            if plan is not None
            else ()
        )
        return DoctorTacticianPlanReceipt(
            roots=compilation.roots,
            receipt_id=_stable_id(
                "receipt",
                {
                    "compilation": compilation.compilation_id,
                    "disposition": disposition.value,
                    "plan": plan.plan_id if plan is not None else "",
                    "reasons": list(reasons),
                },
            ),
            finding_id=compilation.finding_id,
            snapshot_id=compilation.snapshot_id,
            compilation_id=compilation.compilation_id,
            disposition=disposition,
            reason_codes=tuple(reasons),
            plan=plan,
            gate_receipt=gate_receipt,
            ordered_source_routes=ordered_routes,
            selected_premise_ids=(
                plan.selected_premise_ids if plan is not None else ()
            ),
            excluded_premise_ids=(
                plan.excluded_premise_ids if plan is not None else ()
            ),
            goal_ids=tuple(g.goal_id for g in compilation.goals),
            required_facet_ids=compilation.required_facet_ids,
            unknown_frontier_refs=compilation.unknown_frontier_refs,
            exclusion_rationale_refs=(
                plan.exclusion_rationale_refs if plan is not None else ()
            ),
            completeness=(
                "complete"
                if compilation.disposition is DoctorGoalCompilationDisposition.COMPLETE
                else compilation.disposition.value
            ),
            budget_refs=(
                f"budget:max_goals:{self._bounds.max_goals}",
                f"budget:max_premises:{self._bounds.max_premises}",
                f"budget:max_subgoals:{self._bounds.max_subgoals}",
                f"budget:max_routes:{self._bounds.max_routes}",
            ),
            provider_status=provider_status,
            planner_id=PLANNER_ID,
            semantic_authority=False,
            model_invocation_count=0,
            llm_route_present=False,
            invalidation_refs=compilation.invalidation_refs,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def compile_doctor_repair_goals(
    finding: DeterministicDoctorFinding | Mapping[str, Any],
    **kwargs: Any,
) -> DoctorGoalCompilation:
    """Module-level helper for :class:`DoctorRepairGoalCompiler.compile`."""
    return DoctorRepairGoalCompiler().compile(finding, **kwargs)


def plan_doctor_finding(
    finding: DeterministicDoctorFinding | Mapping[str, Any],
    **kwargs: Any,
) -> DoctorTacticianPlanReceipt:
    """Module-level helper for :class:`DeterministicDoctorTactician.plan_finding`."""
    return DeterministicDoctorTactician().plan_finding(finding, **kwargs)


def exact_first_route_order() -> tuple[str, ...]:
    """Public exact-first route order for diagnostics and tests."""
    return tuple(route.value for route in EXACT_FIRST_SOURCE_ROUTES)


def required_facet_kinds() -> tuple[str, ...]:
    """Public closed facet inventory preserved for every repair goal."""
    return tuple(kind.value for kind in REQUIRED_FACET_KINDS)


__all__ = [
    "DETERMINISTIC_DOCTOR_TACTICIAN_INTERFACE",
    "DOCTOR_REPAIR_GOAL_COMPILER_INTERFACE",
    "DOCTOR_GOAL_COMPILATION_SCHEMA",
    "DOCTOR_TACTICIAN_PLAN_RECEIPT_SCHEMA",
    "DOCTOR_TACTICIAN_BOUNDS_SCHEMA",
    "PRODUCER_ID",
    "PLANNER_ID",
    "REQUIRED_FACET_KINDS",
    "EXACT_FIRST_SOURCE_ROUTES",
    "DoctorTacticianError",
    "DoctorTacticianAuthorityError",
    "DoctorTacticianBoundsError",
    "DoctorTacticianSafetyError",
    "DoctorGoalCompilationDisposition",
    "DoctorTacticianPlanDisposition",
    "DoctorTacticianReasonCode",
    "DoctorTacticianBounds",
    "DoctorGoalCompilation",
    "DoctorTacticianPlanReceipt",
    "DoctorRepairGoalCompiler",
    "DeterministicLocalDoctorPlanner",
    "DeterministicDoctorTactician",
    "doctor_roots_to_program_logic_roots",
    "compile_doctor_repair_goals",
    "plan_doctor_finding",
    "exact_first_route_order",
    "required_facet_kinds",
]
