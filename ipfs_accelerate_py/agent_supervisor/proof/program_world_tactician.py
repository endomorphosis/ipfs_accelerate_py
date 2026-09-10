"""Program-world Tactician integration (SAWM-018).

Interface: ``ProgramWorldProofSearch@1``
Evidence: ``sawm/tactician-hammer@1``

Compile finite transition/repair/successor uncertainties into content-addressed
goals and a body-free premise corpus, then run the *landed* datasets Logic
Tactician.  This module is an accelerator operational adapter:

* Datasets remains semantic/formal authority and owns ``LogicTactician``.
* Accelerate compiles obligations, binds current-tree identities, and records
  capability probes.  It does not mint a second tactic engine, prover, proof
  store, logic family, or acceptance gate.
* Tactician plans are advisory (``semantic_authority=false``).
* Models and vector/KG sources may nominate premises only; they cannot create
  proof, countermodel truth, or authority.
* Unavailability is typed.  Contradictions abstain.

Importing this module performs no I/O and starts no threads.  Optional
Tactician/Hammer packages are imported only when planning is requested.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    LogicSubgoal,
    ProgramLogicAuthorityError,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    PremiseAuthority,
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from .formal_verification_contracts import content_identity


PROGRAM_WORLD_PROOF_SEARCH_INTERFACE: Final[str] = "ProgramWorldProofSearch@1"
PROGRAM_WORLD_TACTICIAN_INTERFACE: Final[str] = "ProgramWorldTactician@1"
PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE: Final[str] = "ProgramWorldPremiseCorpus@1"
SAWM_TACTICIAN_HAMMER_EVIDENCE: Final[str] = "sawm/tactician-hammer@1"

PROGRAM_WORLD_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-obligation@1"
)
PROGRAM_WORLD_PREMISE_NOMINATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-premise-nomination@1"
)
PROGRAM_WORLD_PREMISE_CORPUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-premise-corpus@1"
)
PROGRAM_WORLD_GOAL_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-goal-compilation@1"
)
PROGRAM_WORLD_CAPABILITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-proof-capability@1"
)
PROGRAM_WORLD_OBLIGATION_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-obligation-inventory@1"
)

PRODUCER_ID: Final[str] = "program-world-tactician@1"
CONTRACT_VERSION: Final[int] = 1
GENERIC_TACTICIAN_MODULE: Final[str] = "ipfs_datasets_py.logic.tactician"
GENERIC_TACTICIAN_INTERFACE: Final[str] = "ipfs_datasets_py.logic.tactician@1"
HAMMER_MODULE: Final[str] = "ipfs_datasets_py.logic.hammers"
FORMALIZATION_MODULE: Final[str] = "ipfs_datasets_py.logic.formalization"
LOGIC_FAMILY_MODULE: Final[str] = "ipfs_datasets_py.logic.families"

MAX_OBLIGATIONS: Final[int] = 256
MAX_PREMISES: Final[int] = 256
MAX_GOALS: Final[int] = 64
MAX_TEXT_CHARS: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 10_000
DEFAULT_TACTICIAN_POLICY_ID: Final[str] = "logic.tactician.policy.default@1"

ADMITTED_LANGUAGES: Final[frozenset[str]] = frozenset({"python"})
ADMITTED_LOGIC_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "fol",
        "first_order",
        "propositional",
        "smt",
        "lean",
        "coq",
        "isabelle",
        "hol",
    }
)

_SOURCE_CLASS_ORDER: Final[tuple[PremiseSourceClass, ...]] = (
    PremiseSourceClass.REVIEWED_CONTRACT,
    PremiseSourceClass.NORMATIVE_SPEC,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
    PremiseSourceClass.VALUE_PROVENANCE,
    PremiseSourceClass.PROGRAM_GRAPH,
    PremiseSourceClass.SCHEMA_PROTOCOL,
    PremiseSourceClass.LOCAL_STATIC,
    PremiseSourceClass.THEOREM_CORPUS,
    PremiseSourceClass.GIT_LINEAGE,
    PremiseSourceClass.HISTORY,
    PremiseSourceClass.RUNTIME_WITNESS,
    PremiseSourceClass.VECTOR_ANALOGUE,
    PremiseSourceClass.KNOWLEDGE_GRAPH,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
    PremiseSourceClass.COMMENT,
    PremiseSourceClass.MODEL_HYPOTHESIS,
)

_AXIOM_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.REVIEWED_CONTRACT,
        PremiseSourceClass.NORMATIVE_SPEC,
        PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
        PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
        PremiseSourceClass.VALUE_PROVENANCE,
        PremiseSourceClass.PROGRAM_GRAPH,
        PremiseSourceClass.SCHEMA_PROTOCOL,
        PremiseSourceClass.LOCAL_STATIC,
        PremiseSourceClass.THEOREM_CORPUS,
    }
)

_NOMINATING_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.VECTOR_ANALOGUE,
        PremiseSourceClass.KNOWLEDGE_GRAPH,
        PremiseSourceClass.MODEL_HYPOTHESIS,
        PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
        PremiseSourceClass.COMMENT,
        PremiseSourceClass.HISTORY,
        PremiseSourceClass.GIT_LINEAGE,
        PremiseSourceClass.RUNTIME_WITNESS,
    }
)

_SOURCE_TO_ROUTE: Final[dict[PremiseSourceClass, SourceRouteKind]] = {
    PremiseSourceClass.REVIEWED_CONTRACT: SourceRouteKind.REVIEWED_CONTRACT,
    PremiseSourceClass.NORMATIVE_SPEC: SourceRouteKind.NORMATIVE_SPEC,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST: SourceRouteKind.REVIEWED_TEST,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.VALUE_PROVENANCE: SourceRouteKind.DATAFLOW,
    PremiseSourceClass.PROGRAM_GRAPH: SourceRouteKind.GRAPH,
    PremiseSourceClass.SCHEMA_PROTOCOL: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.LOCAL_STATIC: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.THEOREM_CORPUS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.GIT_LINEAGE: SourceRouteKind.HISTORY,
    PremiseSourceClass.HISTORY: SourceRouteKind.HISTORY,
    PremiseSourceClass.RUNTIME_WITNESS: SourceRouteKind.RUNTIME_WITNESS,
    PremiseSourceClass.VECTOR_ANALOGUE: SourceRouteKind.VECTOR,
    PremiseSourceClass.KNOWLEDGE_GRAPH: SourceRouteKind.KNOWLEDGE_GRAPH,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.COMMENT: SourceRouteKind.HISTORY,
    PremiseSourceClass.MODEL_HYPOTHESIS: SourceRouteKind.LLM,
}

_SOURCE_TO_AUTHORITY: Final[dict[PremiseSourceClass, SourceAuthorityClass]] = {
    PremiseSourceClass.REVIEWED_CONTRACT: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.NORMATIVE_SPEC: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST: SourceAuthorityClass.CONFORMANCE,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.VALUE_PROVENANCE: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.PROGRAM_GRAPH: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.SCHEMA_PROTOCOL: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.LOCAL_STATIC: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.THEOREM_CORPUS: SourceAuthorityClass.AUTHORITATIVE,
    PremiseSourceClass.GIT_LINEAGE: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.HISTORY: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.RUNTIME_WITNESS: SourceAuthorityClass.DIAGNOSTIC,
    PremiseSourceClass.VECTOR_ANALOGUE: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.KNOWLEDGE_GRAPH: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.COMMENT: SourceAuthorityClass.NOMINATING,
    PremiseSourceClass.MODEL_HYPOTHESIS: SourceAuthorityClass.NOMINATING,
}

_KIND_TO_FAMILY: Final[dict[str, GoalFamily]] = {
    "transition": GoalFamily.BEHAVIOR,
    "repair": GoalFamily.REFINEMENT,
    "successor": GoalFamily.BEHAVIOR,
    "consistency": GoalFamily.CONSISTENCY,
    "contract": GoalFamily.POSITIVE,
    "counterexample": GoalFamily.COUNTEREXAMPLE,
    "positive": GoalFamily.POSITIVE,
    "negative": GoalFamily.NEGATIVE,
}

_AUTHORITY_PROMOTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "authoritative",
        "authoritative_assurance",
        "expectation_authority",
        "kernel_checked",
        "kernel_verified",
        "proof_authority",
        "proof_success",
        "semantic_authority",
        "verified",
        "write_authority",
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "code",
        "completion",
        "contents",
        "file_text",
        "proof_script",
        "prompt",
        "snippet",
        "source_body",
        "source_text",
        "theorem_text",
        "transcript",
    }
)

_KERNEL_BINARIES: Final[tuple[str, ...]] = (
    "lean",
    "lake",
    "coqc",
    "coqtop",
    "isabelle",
)
_SOLVER_BINARIES: Final[tuple[str, ...]] = ("z3", "cvc5", "vampire", "eprover")

PlannerFactory = Callable[[], Any]


# ---------------------------------------------------------------------------
# Errors / closed vocabularies
# ---------------------------------------------------------------------------


class ProgramWorldTacticianError(ValueError):
    """Closed program-world Tactician contract violation."""


class ProgramWorldTacticianBoundsError(ProgramWorldTacticianError):
    """A compilation exceeded a deterministic compactness bound."""


class ProgramWorldStaleBindingError(ProgramWorldTacticianError):
    """Tree, environment, policy, or corpus bindings are not current."""


class ProgramWorldObligationKind(str, Enum):
    """Closed families of finite program-world proof obligations."""

    TRANSITION = "transition"
    REPAIR = "repair"
    SUCCESSOR = "successor"
    CONSISTENCY = "consistency"
    CONTRACT = "contract"
    COUNTEREXAMPLE = "counterexample"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ProgramWorldPolarity(str, Enum):
    """Closed obligation polarity."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    COUNTEREXAMPLE = "counterexample"


class ProgramWorldCompilationDisposition(str, Enum):
    """Closed compilation outcomes.  Plans never self-admit."""

    PLANNED = "planned"
    COMPILED = "compiled"
    ABSTAINED = "abstained"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    REJECTED = "rejected"


class ProgramWorldReasonCode(str, Enum):
    """Closed reason codes for compilation and proof-search stages."""

    OK = "ok"
    EMPTY_OBLIGATIONS = "empty_obligations"
    FINITE_INVENTORY = "finite_inventory"
    TACTICIAN_PLANNED = "tactician_planned"
    TACTICIAN_UNAVAILABLE = "tactician_unavailable"
    HAMMER_UNAVAILABLE = "hammer_unavailable"
    KERNEL_UNAVAILABLE = "kernel_unavailable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    FORMALIZATION_UNAVAILABLE = "formalization_unavailable"
    LOGIC_FAMILY_UNAVAILABLE = "logic_family_unavailable"
    LOGIC_FAMILY_UNSUPPORTED = "logic_family_unsupported"
    LANGUAGE_UNAVAILABLE = "language_unavailable"
    CONTRADICTION = "contradiction"
    CONFLICTING_OBLIGATIONS = "conflicting_obligations"
    STALE_TREE = "stale_tree"
    STALE_ENVIRONMENT = "stale_environment"
    STALE_POLICY = "stale_policy"
    STALE_CORPUS = "stale_corpus"
    MODEL_NOMINATION_ONLY = "model_nomination_only"
    MODEL_CANNOT_CREATE_PROOF = "model_cannot_create_proof"
    MODEL_CANNOT_CREATE_COUNTERMODEL_TRUTH = (
        "model_cannot_create_countermodel_truth"
    )
    TIMEOUT = "timeout"
    UNSUPPORTED_TRANSLATION = "unsupported_translation"
    RECONSTRUCTED = "reconstructed"
    REPLAYED = "replayed"
    CANDIDATE_NOT_ADMITTED = "candidate_not_admitted"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    ABSTAINED = "abstained"
    MALFORMED = "malformed"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    AUTHORITY_PROMOTION_REJECTED = "authority_promotion_rejected"


class ProgramWorldCapabilityKind(str, Enum):
    """Named proof-pipeline surfaces that must be probed, never assumed."""

    TACTICIAN = "tactician"
    HAMMER = "hammer"
    FORMALIZATION = "formalization"
    LOGIC_FAMILY = "logic_family"
    BACKEND = "backend"
    NATIVE_RECONSTRUCTION = "native_reconstruction"


class ProgramWorldCapabilityStatus(str, Enum):
    """Closed capability probe outcomes."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ProgramWorldTacticianError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise ProgramWorldTacticianError(f"{field_name} must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ProgramWorldTacticianBoundsError(f"{field_name} exceeds its character bound")
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name)
    if any(char.isspace() for char in text):
        raise ProgramWorldTacticianError(
            f"{field_name} must be an opaque compact identifier"
        )
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramWorldTacticianError(f"{field_name} must be a boolean")
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ProgramWorldTacticianError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _sha256_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_payload(payload: Mapping[str, Any], *, prefix: str) -> str:
    return f"{prefix}:{content_identity(payload)}"


def _assert_body_free(value: Any, *, field_name: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramWorldTacticianError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ProgramWorldTacticianError(
                    f"{field_name} may not contain free-form body field {key!r}"
                )
            if normalized in _AUTHORITY_PROMOTION_KEYS and item is True:
                raise ProgramWorldTacticianError(
                    f"{field_name} rejects authority promotion via {key}"
                )
            _assert_body_free(item, field_name=field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name=field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ProgramWorldTacticianError(f"{field_name} may not contain binary bodies")


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProgramWorldTacticianError(f"{field_name} must be an object")
    result = {str(key): item for key, item in value.items()}
    _assert_body_free(result, field_name=field_name)
    return result


def _sequence(value: Any, field_name: str, *, limit: int) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProgramWorldTacticianError(f"{field_name} must be a sequence")
    if len(value) > limit:
        raise ProgramWorldTacticianBoundsError(f"{field_name} exceeds its item bound")
    return tuple(value)


def _ids(values: Any, field_name: str, *, limit: int = MAX_COLLECTION_ITEMS) -> tuple[str, ...]:
    items = _sequence(values, field_name, limit=limit)
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        ident = _identifier(item, field_name)
        if ident not in seen:
            seen.add(ident)
            result.append(ident)
    return tuple(result)


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.pop("schema", None)
        payload.pop("content_id", None)
        payload.pop("contract_version", None)
        return ProgramLogicAuthorityRoots(**payload)
    raise ProgramWorldTacticianError("roots must be ProgramLogicAuthorityRoots")


def _source_class(value: Any) -> PremiseSourceClass:
    return _enum(value, PremiseSourceClass, "source_class")  # type: ignore[return-value]


def _default_authority(source_class: PremiseSourceClass) -> PremiseAuthority:
    if source_class in {
        PremiseSourceClass.REVIEWED_CONTRACT,
        PremiseSourceClass.NORMATIVE_SPEC,
        PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
    }:
        return PremiseAuthority.EXPECTATION
    if source_class in _AXIOM_SOURCE_CLASSES:
        return PremiseAuthority.STATIC_FACT
    return PremiseAuthority.HYPOTHESIS


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _which(binary: str) -> str:
    path = shutil.which(binary)
    return path or ""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramWorldCapabilityReceipt:
    """Typed probe of one landed Tactician/Hammer/kernel surface."""

    kind: ProgramWorldCapabilityKind
    status: ProgramWorldCapabilityStatus
    module_or_binary: str = ""
    interface_version: str = ""
    reason_code: str = ""
    diagnostic: str = ""
    reconstruction_compatible: bool = False

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CAPABILITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldCapabilityKind, "kind")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ProgramWorldCapabilityStatus, "status")
        )
        object.__setattr__(
            self,
            "module_or_binary",
            _text(self.module_or_binary, "module_or_binary", required=False),
        )
        object.__setattr__(
            self,
            "interface_version",
            _text(self.interface_version, "interface_version", required=False),
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", required=False)
        )
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False)
        )
        object.__setattr__(
            self,
            "reconstruction_compatible",
            _bool(self.reconstruction_compatible, "reconstruction_compatible"),
        )
        if (
            self.status is ProgramWorldCapabilityStatus.AVAILABLE
            and not self.module_or_binary
        ):
            raise ProgramWorldTacticianError(
                "available capability requires an exact module or binary path"
            )

    @property
    def available(self) -> bool:
        return self.status is ProgramWorldCapabilityStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind.value,
            "status": self.status.value,
            "available": self.available,
            "module_or_binary": self.module_or_binary,
            "interface_version": self.interface_version,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "reconstruction_compatible": self.reconstruction_compatible,
        }


@dataclass(frozen=True)
class ProgramWorldObligation:
    """One finite, body-free transition/repair/successor proof obligation."""

    obligation_id: str
    kind: ProgramWorldObligationKind
    statement_ref: str
    subject_cid: str
    polarity: ProgramWorldPolarity = ProgramWorldPolarity.POSITIVE
    language: str = "python"
    logic_family: str = "fol"
    environment_binding_cid: str = ""
    tree_id: str = ""
    policy_cid: str = ""
    evidence_cids: tuple[str, ...] = ()
    assumption_cids: tuple[str, ...] = ()
    unavailable_dimensions: tuple[str, ...] = ()
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_OBLIGATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _identifier(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldObligationKind, "kind")
        )
        object.__setattr__(
            self, "statement_ref", _identifier(self.statement_ref, "statement_ref")
        )
        object.__setattr__(
            self, "subject_cid", _identifier(self.subject_cid, "subject_cid")
        )
        object.__setattr__(
            self, "polarity", _enum(self.polarity, ProgramWorldPolarity, "polarity")
        )
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(
            self, "logic_family", _text(self.logic_family, "logic_family")
        )
        object.__setattr__(
            self,
            "environment_binding_cid",
            _text(self.environment_binding_cid, "environment_binding_cid", required=False),
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id", required=False))
        object.__setattr__(
            self, "policy_cid", _text(self.policy_cid, "policy_cid", required=False)
        )
        object.__setattr__(
            self, "evidence_cids", _ids(self.evidence_cids, "evidence_cids")
        )
        object.__setattr__(
            self, "assumption_cids", _ids(self.assumption_cids, "assumption_cids")
        )
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _ids(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        metadata = MappingProxyType(_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "metadata", metadata)
        if self.kind is ProgramWorldObligationKind.COUNTEREXAMPLE:
            object.__setattr__(self, "polarity", ProgramWorldPolarity.COUNTEREXAMPLE)
        if self.kind is ProgramWorldObligationKind.NEGATIVE:
            object.__setattr__(self, "polarity", ProgramWorldPolarity.NEGATIVE)

    @property
    def obligation_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "obligation_id": self.obligation_id,
            "kind": self.kind.value,
            "statement_ref": self.statement_ref,
            "subject_cid": self.subject_cid,
            "polarity": self.polarity.value,
            "language": self.language,
            "logic_family": self.logic_family,
            "environment_binding_cid": self.environment_binding_cid,
            "tree_id": self.tree_id,
            "policy_cid": self.policy_cid,
            "evidence_cids": list(self.evidence_cids),
            "assumption_cids": list(self.assumption_cids),
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgramWorldPremiseNomination:
    """One current-tree or nominating premise.  Nominations are never axioms."""

    premise_id: str
    statement_ref: str
    source_class: PremiseSourceClass
    origin: str = "current_tree"
    evidence_cid: str = ""
    tree_id: str = ""
    nominated_by_model: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PREMISE_NOMINATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "premise_id", _identifier(self.premise_id, "premise_id")
        )
        object.__setattr__(
            self, "statement_ref", _identifier(self.statement_ref, "statement_ref")
        )
        object.__setattr__(self, "source_class", _source_class(self.source_class))
        origin = _text(self.origin, "origin")
        if origin not in {"current_tree", "model", "vector", "knowledge_graph", "history"}:
            raise ProgramWorldTacticianError(
                "origin must be current_tree, model, vector, knowledge_graph, or history"
            )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(
            self, "evidence_cid", _text(self.evidence_cid, "evidence_cid", required=False)
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id", required=False))
        object.__setattr__(
            self,
            "nominated_by_model",
            _bool(self.nominated_by_model, "nominated_by_model"),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )
        if self.nominated_by_model or self.origin == "model":
            object.__setattr__(self, "source_class", PremiseSourceClass.MODEL_HYPOTHESIS)
            object.__setattr__(self, "nominated_by_model", True)
            object.__setattr__(self, "origin", "model")

    @property
    def axiom_eligible(self) -> bool:
        return (
            not self.nominated_by_model
            and self.origin == "current_tree"
            and self.source_class in _AXIOM_SOURCE_CLASSES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "premise_id": self.premise_id,
            "statement_ref": self.statement_ref,
            "source_class": self.source_class.value,
            "origin": self.origin,
            "evidence_cid": self.evidence_cid,
            "tree_id": self.tree_id,
            "nominated_by_model": self.nominated_by_model,
            "axiom_eligible": self.axiom_eligible,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgramWorldPremiseCorpus:
    """Content-addressed program-world premise corpus over landed LPR records."""

    roots: ProgramLogicAuthorityRoots
    inner: ProgramLogicPremiseCorpus
    current_tree_premise_ids: tuple[str, ...] = ()
    nominated_premise_ids: tuple[str, ...] = ()
    axiom_premise_ids: tuple[str, ...] = ()
    excluded_premise_ids: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PREMISE_CORPUS_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        if not isinstance(self.inner, ProgramLogicPremiseCorpus):
            raise ProgramWorldTacticianError(
                "inner must be ProgramLogicPremiseCorpus"
            )
        if self.inner.roots.content_id != self.roots.content_id:
            raise ProgramWorldStaleBindingError(
                "premise corpus roots must match compilation roots"
            )
        object.__setattr__(
            self,
            "current_tree_premise_ids",
            _ids(self.current_tree_premise_ids, "current_tree_premise_ids"),
        )
        object.__setattr__(
            self,
            "nominated_premise_ids",
            _ids(self.nominated_premise_ids, "nominated_premise_ids"),
        )
        object.__setattr__(
            self, "axiom_premise_ids", _ids(self.axiom_premise_ids, "axiom_premise_ids")
        )
        object.__setattr__(
            self,
            "excluded_premise_ids",
            _ids(self.excluded_premise_ids, "excluded_premise_ids"),
        )
        overlap = set(self.axiom_premise_ids) & set(self.nominated_premise_ids)
        if overlap:
            raise ProgramWorldTacticianError(
                "model-nominated premises cannot be selected as axioms"
            )

    @property
    def corpus_cid(self) -> str:
        return self.inner.corpus_id

    @property
    def consistency_disposition(self) -> ConsistencyDisposition:
        return self.inner.consistency_disposition

    @property
    def semantic_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "roots": self.roots.to_dict(),
            "corpus_cid": self.corpus_cid,
            "inner": self.inner.to_dict(),
            "current_tree_premise_ids": list(self.current_tree_premise_ids),
            "nominated_premise_ids": list(self.nominated_premise_ids),
            "axiom_premise_ids": list(self.axiom_premise_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "consistency_disposition": self.consistency_disposition.value,
            "semantic_authority": False,
        }


@dataclass(frozen=True)
class ProgramWorldObligationInventory:
    """Finite inventory of compiled obligations.  Absence is not closure."""

    obligation_cids: tuple[str, ...]
    goal_ids: tuple[str, ...]
    languages: tuple[str, ...]
    logic_families: tuple[str, ...]
    complete: bool
    unavailable_dimensions: tuple[str, ...] = ()
    conflict_statement_refs: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_OBLIGATION_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_cids", _ids(self.obligation_cids, "obligation_cids")
        )
        object.__setattr__(self, "goal_ids", _ids(self.goal_ids, "goal_ids"))
        object.__setattr__(self, "languages", _ids(self.languages, "languages"))
        object.__setattr__(
            self, "logic_families", _ids(self.logic_families, "logic_families")
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _ids(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(
            self,
            "conflict_statement_refs",
            _ids(self.conflict_statement_refs, "conflict_statement_refs"),
        )

    @property
    def inventory_cid(self) -> str:
        return content_identity(self.to_dict())

    @property
    def finite(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "obligation_cids": list(self.obligation_cids),
            "goal_ids": list(self.goal_ids),
            "languages": list(self.languages),
            "logic_families": list(self.logic_families),
            "complete": self.complete,
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "conflict_statement_refs": list(self.conflict_statement_refs),
            "finite": True,
        }


@dataclass(frozen=True)
class ProgramWorldGoalCompilation:
    """Compiled finite goals, premise corpus, and advisory tactic plan."""

    roots: ProgramLogicAuthorityRoots
    disposition: ProgramWorldCompilationDisposition
    reason_code: ProgramWorldReasonCode
    obligations: tuple[ProgramWorldObligation, ...]
    goals: tuple[ProgramLogicGoal, ...]
    corpus: ProgramWorldPremiseCorpus
    inventory: ProgramWorldObligationInventory
    tactic_plan: TacticianSearchPlan | None = None
    capabilities: tuple[ProgramWorldCapabilityReceipt, ...] = ()
    diagnostic: str = ""
    semantic_authority: bool = False

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_GOAL_COMPILATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProgramWorldCompilationDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, ProgramWorldReasonCode, "reason_code"),
        )
        obligations = tuple(self.obligations)
        if len(obligations) > MAX_OBLIGATIONS:
            raise ProgramWorldTacticianBoundsError("obligation bound exceeded")
        for item in obligations:
            if not isinstance(item, ProgramWorldObligation):
                raise ProgramWorldTacticianError(
                    "obligations must be ProgramWorldObligation values"
                )
        object.__setattr__(self, "obligations", obligations)
        goals = tuple(self.goals)
        if len(goals) > MAX_GOALS:
            raise ProgramWorldTacticianBoundsError("goal bound exceeded")
        for item in goals:
            if not isinstance(item, ProgramLogicGoal):
                raise ProgramWorldTacticianError("goals must be ProgramLogicGoal values")
        object.__setattr__(self, "goals", goals)
        if not isinstance(self.corpus, ProgramWorldPremiseCorpus):
            raise ProgramWorldTacticianError("corpus must be ProgramWorldPremiseCorpus")
        if not isinstance(self.inventory, ProgramWorldObligationInventory):
            raise ProgramWorldTacticianError(
                "inventory must be ProgramWorldObligationInventory"
            )
        if self.tactic_plan is not None and not isinstance(
            self.tactic_plan, TacticianSearchPlan
        ):
            raise ProgramWorldTacticianError(
                "tactic_plan must be TacticianSearchPlan or None"
            )
        if self.tactic_plan is not None and self.tactic_plan.semantic_authority:
            raise ProgramLogicAuthorityError(
                "tactician search plans cannot claim semantic authority"
            )
        capabilities = tuple(self.capabilities)
        for item in capabilities:
            if not isinstance(item, ProgramWorldCapabilityReceipt):
                raise ProgramWorldTacticianError(
                    "capabilities must be ProgramWorldCapabilityReceipt values"
                )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False)
        )
        if self.semantic_authority is not False:
            raise ProgramWorldTacticianError(
                "goal compilation cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @property
    def compilation_cid(self) -> str:
        return content_identity(self.to_dict())

    def capability(self, kind: ProgramWorldCapabilityKind | str) -> ProgramWorldCapabilityReceipt | None:
        key = kind.value if isinstance(kind, ProgramWorldCapabilityKind) else kind
        for item in self.capabilities:
            if item.kind.value == key:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": PROGRAM_WORLD_PROOF_SEARCH_INTERFACE,
            "evidence": SAWM_TACTICIAN_HAMMER_EVIDENCE,
            "roots": self.roots.to_dict(),
            "disposition": self.disposition.value,
            "reason_code": self.reason_code.value,
            "obligations": [item.to_dict() for item in self.obligations],
            "goals": [item.to_dict() for item in self.goals],
            "corpus": self.corpus.to_dict(),
            "inventory": self.inventory.to_dict(),
            "tactic_plan": None if self.tactic_plan is None else self.tactic_plan.to_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "diagnostic": self.diagnostic,
            "semantic_authority": False,
            "producer_id": PRODUCER_ID,
            "contract_version": CONTRACT_VERSION,
        }


# ---------------------------------------------------------------------------
# Capability probe (validation-environment PATH, never assumed)
# ---------------------------------------------------------------------------


def probe_program_world_proof_capabilities() -> tuple[ProgramWorldCapabilityReceipt, ...]:
    """Probe landed Tactician/Hammer/kernel surfaces against the current PATH.

    Discovery is not authority.  Package presence and ``shutil.which`` hits
    never promote a surface to proof or completion authority.  Binaries are
    not executed.
    """

    receipts: list[ProgramWorldCapabilityReceipt] = []

    tactician_ok = _module_available(GENERIC_TACTICIAN_MODULE)
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.TACTICIAN,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if tactician_ok
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=GENERIC_TACTICIAN_MODULE if tactician_ok else "",
            interface_version=GENERIC_TACTICIAN_INTERFACE if tactician_ok else "",
            reason_code="" if tactician_ok else ProgramWorldReasonCode.TACTICIAN_UNAVAILABLE.value,
            diagnostic="" if tactician_ok else "logic.tactician module is not importable",
        )
    )

    hammer_ok = _module_available(HAMMER_MODULE)
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.HAMMER,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if hammer_ok
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=HAMMER_MODULE if hammer_ok else "",
            interface_version="HammerBackend@1" if hammer_ok else "",
            reason_code="" if hammer_ok else ProgramWorldReasonCode.HAMMER_UNAVAILABLE.value,
            diagnostic="" if hammer_ok else "logic.hammers module is not importable",
        )
    )

    formalization_ok = _module_available(FORMALIZATION_MODULE)
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.FORMALIZATION,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if formalization_ok
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=FORMALIZATION_MODULE if formalization_ok else "",
            reason_code=(
                ""
                if formalization_ok
                else ProgramWorldReasonCode.FORMALIZATION_UNAVAILABLE.value
            ),
            diagnostic=(
                ""
                if formalization_ok
                else "logic.formalization module is not importable"
            ),
        )
    )

    family_ok = _module_available(LOGIC_FAMILY_MODULE)
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.LOGIC_FAMILY,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if family_ok
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=LOGIC_FAMILY_MODULE if family_ok else "",
            reason_code=(
                "" if family_ok else ProgramWorldReasonCode.LOGIC_FAMILY_UNAVAILABLE.value
            ),
            diagnostic="" if family_ok else "logic.families module is not importable",
        )
    )

    solver_hits = tuple(
        f"{name}:{path}" for name in _SOLVER_BINARIES if (path := _which(name))
    )
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.BACKEND,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if solver_hits
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=solver_hits[0].split(":", 1)[-1] if solver_hits else "",
            reason_code="" if solver_hits else ProgramWorldReasonCode.BACKEND_UNAVAILABLE.value,
            diagnostic=(
                ",".join(solver_hits)
                if solver_hits
                else "no allowlisted SMT/ATP solver on PATH"
            ),
        )
    )

    kernel_hits = tuple(
        f"{name}:{path}" for name in _KERNEL_BINARIES if (path := _which(name))
    )
    receipts.append(
        ProgramWorldCapabilityReceipt(
            kind=ProgramWorldCapabilityKind.NATIVE_RECONSTRUCTION,
            status=(
                ProgramWorldCapabilityStatus.AVAILABLE
                if kernel_hits
                else ProgramWorldCapabilityStatus.UNAVAILABLE
            ),
            module_or_binary=kernel_hits[0].split(":", 1)[-1] if kernel_hits else "",
            reason_code="" if kernel_hits else ProgramWorldReasonCode.KERNEL_UNAVAILABLE.value,
            diagnostic=(
                ",".join(kernel_hits)
                if kernel_hits
                else "no Lean/Coq/Isabelle kernel on PATH"
            ),
            reconstruction_compatible=bool(kernel_hits),
        )
    )
    return tuple(receipts)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _decode_obligation(value: Any) -> ProgramWorldObligation:
    if isinstance(value, ProgramWorldObligation):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.pop("schema", None)
        payload.pop("obligation_cid", None)
        return ProgramWorldObligation(**payload)
    raise ProgramWorldTacticianError("obligation must be ProgramWorldObligation")


def _decode_nomination(value: Any) -> ProgramWorldPremiseNomination:
    if isinstance(value, ProgramWorldPremiseNomination):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.pop("schema", None)
        payload.pop("axiom_eligible", None)
        return ProgramWorldPremiseNomination(**payload)
    raise ProgramWorldTacticianError(
        "premise must be ProgramWorldPremiseNomination"
    )


def _conflict_statements(
    obligations: Sequence[ProgramWorldObligation],
) -> tuple[str, ...]:
    polarities: dict[str, set[str]] = {}
    for item in obligations:
        polarities.setdefault(item.statement_ref, set()).add(item.polarity.value)
    conflicts = [
        statement
        for statement, seen in polarities.items()
        if {"positive", "negative"} <= seen or {"positive", "counterexample"} <= seen
    ]
    return tuple(sorted(conflicts))


def _compile_goals(
    *,
    roots: ProgramLogicAuthorityRoots,
    obligations: Sequence[ProgramWorldObligation],
) -> tuple[ProgramLogicGoal, ...]:
    goals: list[ProgramLogicGoal] = []
    for obligation in obligations:
        family = _KIND_TO_FAMILY.get(obligation.kind.value, GoalFamily.POSITIVE)
        if obligation.polarity is ProgramWorldPolarity.NEGATIVE:
            family = GoalFamily.NEGATIVE
        elif obligation.polarity is ProgramWorldPolarity.COUNTEREXAMPLE:
            family = GoalFamily.COUNTEREXAMPLE
        disposition = GoalDisposition.OPEN
        if obligation.unavailable_dimensions:
            disposition = GoalDisposition.UNSUPPORTED
        if obligation.language not in ADMITTED_LANGUAGES:
            disposition = GoalDisposition.UNSUPPORTED
        if obligation.logic_family not in ADMITTED_LOGIC_FAMILIES:
            disposition = GoalDisposition.UNSUPPORTED
        goal = ProgramLogicGoal(
            roots=roots,
            goal_id=obligation.obligation_id,
            family=family,
            disposition=disposition,
            positive_statement_ref=obligation.statement_ref,
            negative_target_ref=(
                obligation.statement_ref if family is GoalFamily.NEGATIVE else ""
            ),
            counterexample_target_ref=(
                obligation.statement_ref
                if family is GoalFamily.COUNTEREXAMPLE
                else ""
            ),
            affected_symbol_ids=(obligation.subject_cid,),
            source_refs=obligation.evidence_cids,
            assumption_refs=obligation.assumption_cids,
            assumption_authority=SourceAuthorityClass.NONE,
            proof_status=ProofStatus.UNPROVED,
            logic_family_refs=(f"logic-family:{obligation.logic_family}",),
            bound_refs=(obligation.subject_cid,),
            invalidation_refs=(roots.tree_id, roots.environment_id, roots.policy_id),
        )
        goals.append(goal)
    return tuple(goals)


def _compile_corpus(
    *,
    roots: ProgramLogicAuthorityRoots,
    nominations: Sequence[ProgramWorldPremiseNomination],
) -> ProgramWorldPremiseCorpus:
    premises: list[ProgramLogicPremise] = []
    current_ids: list[str] = []
    nominated_ids: list[str] = []
    axiom_ids: list[str] = []
    excluded_ids: list[str] = []
    for nomination in nominations:
        if nomination.tree_id and nomination.tree_id != roots.tree_id:
            raise ProgramWorldStaleBindingError("premise tree_id does not match roots")
        source_class = nomination.source_class
        authority = _default_authority(source_class)
        if nomination.nominated_by_model:
            authority = PremiseAuthority.HYPOTHESIS
            source_class = PremiseSourceClass.MODEL_HYPOTHESIS
        premise = ProgramLogicPremise(
            roots=roots,
            premise_id=nomination.premise_id,
            source_class=source_class,
            statement_ref=nomination.statement_ref,
            statement_digest=_sha256_digest(nomination.statement_ref),
            lowering_ref=f"lowering:{source_class.value}",
            authority=authority,
            source_precedence=_SOURCE_CLASS_ORDER.index(source_class)
            if source_class in _SOURCE_CLASS_ORDER
            else len(_SOURCE_CLASS_ORDER),
            expectation_authority=authority is PremiseAuthority.EXPECTATION,
            semantic_authority=False,
            source_route=_SOURCE_TO_ROUTE[source_class],
            source_authority_class=_SOURCE_TO_AUTHORITY[source_class],
            tree_identity=roots.tree_id,
            graph_identity=roots.graph_id,
        )
        premises.append(premise)
        if nomination.nominated_by_model or source_class in _NOMINATING_SOURCE_CLASSES:
            nominated_ids.append(nomination.premise_id)
            excluded_ids.append(nomination.premise_id)
        else:
            current_ids.append(nomination.premise_id)
            if nomination.axiom_eligible:
                axiom_ids.append(nomination.premise_id)
            else:
                excluded_ids.append(nomination.premise_id)
    inner = ProgramLogicPremiseCorpus(roots=roots, premises=tuple(premises))
    return ProgramWorldPremiseCorpus(
        roots=roots,
        inner=inner,
        current_tree_premise_ids=tuple(current_ids),
        nominated_premise_ids=tuple(nominated_ids),
        axiom_premise_ids=tuple(axiom_ids),
        excluded_premise_ids=tuple(excluded_ids),
    )


def _route_kind_for_source_class(source_class: str) -> SourceRouteKind:
    try:
        parsed = PremiseSourceClass(source_class)
    except ValueError:
        return SourceRouteKind.LOCAL_STATIC
    return _SOURCE_TO_ROUTE.get(parsed, SourceRouteKind.LOCAL_STATIC)


def _plan_from_tactician(
    *,
    roots: ProgramLogicAuthorityRoots,
    goals: Sequence[ProgramLogicGoal],
    corpus: ProgramWorldPremiseCorpus,
    native_plan: Any,
) -> TacticianSearchPlan:
    selected_ids: list[str] = []
    excluded_ids: list[str] = list(corpus.excluded_premise_ids)
    routes: list[SourceRouteKind] = []
    for route in getattr(native_plan, "selected_routes", ()) or ():
        source_id = str(getattr(route, "source_id", "") or "")
        source_class = str(getattr(route, "source_class", "") or "")
        if source_id and source_id in set(corpus.axiom_premise_ids):
            selected_ids.append(source_id)
        elif source_id:
            excluded_ids.append(source_id)
        routes.append(_route_kind_for_source_class(source_class))
    for route in getattr(native_plan, "excluded_routes", ()) or ():
        source_id = str(getattr(route, "source_id", "") or "")
        if source_id:
            excluded_ids.append(source_id)
    if not routes:
        routes = [SourceRouteKind.LOCAL_STATIC, SourceRouteKind.GRAPH]
    # Model nominations stay excluded even if a planner listed them.
    nominated = set(corpus.nominated_premise_ids)
    selected_ids = [item for item in selected_ids if item not in nominated]
    selected_set = set(selected_ids)
    excluded_ids = [
        item
        for item in dict.fromkeys([*excluded_ids, *sorted(nominated)])
        if item not in selected_set
    ]
    subgoals: list[LogicSubgoal] = []
    goal_ids = tuple(item.goal_id for item in goals)
    native_subgoals = list(getattr(native_plan, "subgoals", ()) or ())
    if native_subgoals:
        for item in native_subgoals:
            parent = str(getattr(item, "parent_goal_id", "") or goal_ids[0])
            if parent not in goal_ids:
                parent = goal_ids[0]
            subgoals.append(
                LogicSubgoal(
                    subgoal_id=str(getattr(item, "subgoal_id")),
                    goal_id=parent,
                    disposition=SubgoalDisposition.PLANNED,
                    claim_ref=str(getattr(item, "statement_ref")),
                    depends_on=tuple(getattr(item, "depends_on", ()) or ()),
                    source_route=SourceRouteKind.LOCAL_STATIC,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    proof_status=ProofStatus.UNPROVED,
                )
            )
    else:
        for goal in goals:
            subgoals.append(
                LogicSubgoal(
                    subgoal_id=f"subgoal:{goal.goal_id}",
                    goal_id=goal.goal_id,
                    disposition=SubgoalDisposition.PLANNED,
                    claim_ref=goal.positive_statement_ref,
                    source_route=SourceRouteKind.LOCAL_STATIC,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    proof_status=ProofStatus.UNPROVED,
                )
            )
    plan_id = _digest_payload(
        {
            "native_plan_id": str(getattr(native_plan, "plan_id", "") or ""),
            "goal_ids": list(goal_ids),
            "corpus": corpus.corpus_cid,
        },
        prefix="plan",
    )
    return TacticianSearchPlan(
        roots=roots,
        plan_id=plan_id,
        goal_ids=goal_ids,
        ordered_source_routes=tuple(routes),
        selected_premise_ids=tuple(dict.fromkeys(selected_ids)),
        excluded_premise_ids=tuple(dict.fromkeys(excluded_ids)),
        subgoals=tuple(subgoals),
        planned_logic_family_refs=tuple(
            ref for goal in goals for ref in goal.logic_family_refs
        ),
        stop_policy_ref="stop:budget-or-gap-closed",
        abstention_policy_ref="abstain:contradiction-or-unavailable",
        resource_policy_ref="resource:finite-no-network",
        planner_id=str(getattr(native_plan, "planner_id", "") or GENERIC_TACTICIAN_INTERFACE),
        config_id=str(getattr(native_plan, "config_root", "") or DEFAULT_TACTICIAN_POLICY_ID),
        semantic_authority=False,
        invalidation_refs=(roots.tree_id, roots.environment_id, corpus.corpus_cid),
    )


def _run_landed_tactician(
    *,
    roots: ProgramLogicAuthorityRoots,
    goals: Sequence[ProgramLogicGoal],
    corpus: ProgramWorldPremiseCorpus,
    nominations: Sequence[ProgramWorldPremiseNomination],
    planner_factory: PlannerFactory | None,
) -> tuple[TacticianSearchPlan | None, str]:
    try:
        tactician_pkg = importlib.import_module(GENERIC_TACTICIAN_MODULE)
        models = importlib.import_module(f"{GENERIC_TACTICIAN_MODULE}.models")
        policy_mod = importlib.import_module(f"{GENERIC_TACTICIAN_MODULE}.policy")
    except Exception as exc:  # noqa: BLE001 - typed unavailability
        return None, f"{type(exc).__name__}: {exc}"

    planner = None
    if planner_factory is not None:
        planner = planner_factory()
    else:
        planner_cls = getattr(tactician_pkg, "LogicTactician", None)
        if planner_cls is None:
            planner_mod = importlib.import_module(f"{GENERIC_TACTICIAN_MODULE}.planner")
            planner_cls = getattr(planner_mod, "LogicTactician")
        planner = planner_cls()

    default_policy = policy_mod.default_policy(
        policy_id=DEFAULT_TACTICIAN_POLICY_ID,
        source_class_order=[item.value for item in _SOURCE_CLASS_ORDER],
    )
    sources = []
    for nomination in nominations:
        sources.append(
            models.TacticianSource(
                source_id=nomination.premise_id,
                source_class=nomination.source_class.value,
                precedence=_SOURCE_CLASS_ORDER.index(nomination.source_class)
                if nomination.source_class in _SOURCE_CLASS_ORDER
                else len(_SOURCE_CLASS_ORDER),
                rationale=(
                    "model-nomination"
                    if nomination.nominated_by_model
                    else "current-tree-premise"
                ),
                source_root=roots.tree_id,
            )
        )
    if not sources:
        sources.append(
            models.TacticianSource(
                source_id="premise:empty-current-tree",
                source_class=PremiseSourceClass.LOCAL_STATIC.value,
                precedence=0,
                rationale="empty-corpus-placeholder",
                source_root=roots.tree_id,
            )
        )

    primary = goals[0]
    native_goal = models.TacticianGoal(
        goal_id=primary.goal_id,
        statement_ref=primary.positive_statement_ref,
        goal_family=primary.family.value,
        goal_root=roots.tree_id,
        corpus_root=corpus.corpus_cid,
        config_root=DEFAULT_TACTICIAN_POLICY_ID,
        authority_roots={
            "tree_id": roots.tree_id,
            "environment_id": roots.environment_id,
            "policy_id": roots.policy_id,
            "graph_id": roots.graph_id,
        },
        proof_gaps=[goal.goal_id for goal in goals[1:]],
        assumptions=list(primary.assumption_refs),
    )
    try:
        native_plan = planner.plan(native_goal, sources, default_policy)
        if getattr(native_plan, "semantic_authority", False):
            raise ProgramLogicAuthorityError(
                "landed tactician plan claimed semantic authority"
            )
        return _plan_from_tactician(
            roots=roots,
            goals=goals,
            corpus=corpus,
            native_plan=native_plan,
        ), ""
    except Exception as exc:  # noqa: BLE001 - typed unavailability
        return None, f"{type(exc).__name__}: {exc}"


class ProgramWorldTactician:
    """Operational compiler from program-world uncertainties onto LogicTactician."""

    INTERFACE: ClassVar[str] = PROGRAM_WORLD_TACTICIAN_INTERFACE

    def __init__(self, *, planner_factory: PlannerFactory | None = None) -> None:
        self._planner_factory = planner_factory

    def compile(
        self,
        *,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
        obligations: Sequence[ProgramWorldObligation | Mapping[str, Any]],
        premises: Sequence[ProgramWorldPremiseNomination | Mapping[str, Any]] = (),
        expected_tree_id: str = "",
        expected_environment_id: str = "",
        expected_policy_id: str = "",
        capabilities: Sequence[ProgramWorldCapabilityReceipt] | None = None,
    ) -> ProgramWorldGoalCompilation:
        bound_roots = _roots(roots)
        if expected_tree_id and expected_tree_id != bound_roots.tree_id:
            raise ProgramWorldStaleBindingError("tree_id is not current")
        if (
            expected_environment_id
            and expected_environment_id != bound_roots.environment_id
        ):
            raise ProgramWorldStaleBindingError("environment_id is not current")
        if expected_policy_id and expected_policy_id != bound_roots.policy_id:
            raise ProgramWorldStaleBindingError("policy_id is not current")

        decoded_obligations = tuple(
            _decode_obligation(item)
            for item in _sequence(obligations, "obligations", limit=MAX_OBLIGATIONS)
        )
        decoded_premises = tuple(
            _decode_nomination(item)
            for item in _sequence(premises, "premises", limit=MAX_PREMISES)
        )
        probed = tuple(capabilities) if capabilities is not None else probe_program_world_proof_capabilities()

        if not decoded_obligations:
            empty_corpus = _compile_corpus(roots=bound_roots, nominations=())
            empty_inventory = ProgramWorldObligationInventory(
                obligation_cids=(),
                goal_ids=(),
                languages=(),
                logic_families=(),
                complete=False,
            )
            return ProgramWorldGoalCompilation(
                roots=bound_roots,
                disposition=ProgramWorldCompilationDisposition.REJECTED,
                reason_code=ProgramWorldReasonCode.EMPTY_OBLIGATIONS,
                obligations=(),
                goals=(),
                corpus=empty_corpus,
                inventory=empty_inventory,
                capabilities=probed,
                diagnostic="compile_program_world_goals requires a finite obligation set",
            )

        languages = {item.language for item in decoded_obligations}
        families = {item.logic_family for item in decoded_obligations}
        unavailable = tuple(
            sorted(
                {
                    dim
                    for item in decoded_obligations
                    for dim in item.unavailable_dimensions
                }
            )
        )
        for obligation in decoded_obligations:
            if obligation.tree_id and obligation.tree_id != bound_roots.tree_id:
                raise ProgramWorldStaleBindingError("obligation tree_id is stale")
            if (
                obligation.environment_binding_cid
                and obligation.environment_binding_cid != bound_roots.environment_id
            ):
                raise ProgramWorldStaleBindingError(
                    "obligation environment binding is stale"
                )
            if obligation.policy_cid and obligation.policy_cid != bound_roots.policy_id:
                raise ProgramWorldStaleBindingError("obligation policy_cid is stale")

        unsupported_language = languages - ADMITTED_LANGUAGES
        unsupported_family = families - ADMITTED_LOGIC_FAMILIES
        conflicts = _conflict_statements(decoded_obligations)
        goals = _compile_goals(roots=bound_roots, obligations=decoded_obligations)
        corpus = _compile_corpus(roots=bound_roots, nominations=decoded_premises)
        inventory = ProgramWorldObligationInventory(
            obligation_cids=tuple(item.obligation_cid for item in decoded_obligations),
            goal_ids=tuple(item.goal_id for item in goals),
            languages=tuple(sorted(languages)),
            logic_families=tuple(sorted(families)),
            complete=not unavailable and not conflicts,
            unavailable_dimensions=unavailable,
            conflict_statement_refs=conflicts,
        )

        if conflicts:
            return ProgramWorldGoalCompilation(
                roots=bound_roots,
                disposition=ProgramWorldCompilationDisposition.CONFLICT,
                reason_code=ProgramWorldReasonCode.CONFLICTING_OBLIGATIONS,
                obligations=decoded_obligations,
                goals=goals,
                corpus=corpus,
                inventory=inventory,
                capabilities=probed,
                diagnostic="contradictory polarities on the same statement abstain",
            )
        if unsupported_language:
            return ProgramWorldGoalCompilation(
                roots=bound_roots,
                disposition=ProgramWorldCompilationDisposition.UNSUPPORTED,
                reason_code=ProgramWorldReasonCode.LANGUAGE_UNAVAILABLE,
                obligations=decoded_obligations,
                goals=goals,
                corpus=corpus,
                inventory=inventory,
                capabilities=probed,
                diagnostic="language is typed unavailable: "
                + ",".join(sorted(unsupported_language)),
            )
        if unsupported_family:
            return ProgramWorldGoalCompilation(
                roots=bound_roots,
                disposition=ProgramWorldCompilationDisposition.UNSUPPORTED,
                reason_code=ProgramWorldReasonCode.LOGIC_FAMILY_UNSUPPORTED,
                obligations=decoded_obligations,
                goals=goals,
                corpus=corpus,
                inventory=inventory,
                capabilities=probed,
                diagnostic="logic family is unsupported: "
                + ",".join(sorted(unsupported_family)),
            )

        tactician_cap = next(
            (
                item
                for item in probed
                if item.kind is ProgramWorldCapabilityKind.TACTICIAN
            ),
            None,
        )
        plan = None
        diagnostic = ""
        if (tactician_cap is not None and tactician_cap.available) or self._planner_factory:
            plan, diagnostic = _run_landed_tactician(
                roots=bound_roots,
                goals=goals,
                corpus=corpus,
                nominations=decoded_premises,
                planner_factory=self._planner_factory,
            )
            if plan is None:
                return ProgramWorldGoalCompilation(
                    roots=bound_roots,
                    disposition=ProgramWorldCompilationDisposition.UNAVAILABLE,
                    reason_code=ProgramWorldReasonCode.TACTICIAN_UNAVAILABLE,
                    obligations=decoded_obligations,
                    goals=goals,
                    corpus=corpus,
                    inventory=inventory,
                    capabilities=probed,
                    diagnostic=diagnostic or "landed LogicTactician is unavailable",
                )
            return ProgramWorldGoalCompilation(
                roots=bound_roots,
                disposition=ProgramWorldCompilationDisposition.PLANNED,
                reason_code=ProgramWorldReasonCode.TACTICIAN_PLANNED,
                obligations=decoded_obligations,
                goals=goals,
                corpus=corpus,
                inventory=inventory,
                tactic_plan=plan,
                capabilities=probed,
            )

        return ProgramWorldGoalCompilation(
            roots=bound_roots,
            disposition=ProgramWorldCompilationDisposition.UNAVAILABLE,
            reason_code=ProgramWorldReasonCode.TACTICIAN_UNAVAILABLE,
            obligations=decoded_obligations,
            goals=goals,
            corpus=corpus,
            inventory=inventory,
            capabilities=probed,
            diagnostic="landed LogicTactician capability is not available",
        )


def compile_program_world_goals(
    *,
    roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
    obligations: Sequence[ProgramWorldObligation | Mapping[str, Any]],
    premises: Sequence[ProgramWorldPremiseNomination | Mapping[str, Any]] = (),
    expected_tree_id: str = "",
    expected_environment_id: str = "",
    expected_policy_id: str = "",
    planner_factory: PlannerFactory | None = None,
    capabilities: Sequence[ProgramWorldCapabilityReceipt] | None = None,
) -> ProgramWorldGoalCompilation:
    """Compile finite program-world uncertainties and run landed Tactician."""

    return ProgramWorldTactician(planner_factory=planner_factory).compile(
        roots=roots,
        obligations=obligations,
        premises=premises,
        expected_tree_id=expected_tree_id,
        expected_environment_id=expected_environment_id,
        expected_policy_id=expected_policy_id,
        capabilities=capabilities,
    )


__all__ = [
    "ADMITTED_LANGUAGES",
    "ADMITTED_LOGIC_FAMILIES",
    "PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE",
    "PROGRAM_WORLD_PROOF_SEARCH_INTERFACE",
    "PROGRAM_WORLD_TACTICIAN_INTERFACE",
    "SAWM_TACTICIAN_HAMMER_EVIDENCE",
    "ProgramWorldCapabilityKind",
    "ProgramWorldCapabilityReceipt",
    "ProgramWorldCapabilityStatus",
    "ProgramWorldCompilationDisposition",
    "ProgramWorldGoalCompilation",
    "ProgramWorldObligation",
    "ProgramWorldObligationInventory",
    "ProgramWorldObligationKind",
    "ProgramWorldPolarity",
    "ProgramWorldPremiseCorpus",
    "ProgramWorldPremiseNomination",
    "ProgramWorldReasonCode",
    "ProgramWorldStaleBindingError",
    "ProgramWorldTactician",
    "ProgramWorldTacticianBoundsError",
    "ProgramWorldTacticianError",
    "compile_program_world_goals",
    "probe_program_world_proof_capabilities",
]
