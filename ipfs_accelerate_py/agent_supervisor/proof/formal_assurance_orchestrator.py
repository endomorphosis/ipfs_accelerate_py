"""FACP-050 — proof cache and solver orchestration adapter.

Composes the existing trust-aware proof cache and multi-prover router through
one fail-closed adapter. Capsule identity binds incremental cache keys;
obligations escalate through the cheapest sound ladder; disagreement yields an
explicit conflict receipt; unknown and unavailable never become verified.

This module does **not**:

* fork :mod:`formal_verification_cache` or :mod:`multi_prover_router` authority
* treat solver / LLM candidates as proofs
* admit cache reuse when the semantic closure changed
* invoke an LLM as an assurance stage
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Optional

from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    canonical_json,
)
from .formal_verification_cache import (
    CacheLookupStatus,
    ProofCacheKey,
    build_proof_cache_key,
)
from .multi_prover_router import (
    AttemptOutcome,
    MultiProverRouter,
    PortfolioResult,
    PortfolioVerdict,
    PropertyKind,
    PropertyObligation,
    ProverOutput,
    ProverRole,
)


# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

TASK_ID: Final[str] = "FACP-050"
GOAL_ID: Final[str] = "FACP-G640"
BUNDLE: Final[str] = "facp/proof/orchestration"
SCHEMA: Final[str] = "facp/proof-orchestration@1"
PROOF_ROUTER_SCHEMA: Final[str] = "facp/proof-router@1"
PROOF_CACHE_KEY_SCHEMA: Final[str] = "facp/proof-cache-key@1"
SOLVER_CONFLICT_SCHEMA: Final[str] = "facp/solver-conflict@1"
CACHE_REUSE_SCHEMA: Final[str] = "facp/proof-cache-reuse@1"
ESCALATION_SCHEMA: Final[str] = "facp/proof-escalation@1"
RESULT_SCHEMA: Final[str] = "facp/proof-orchestration-result@1"
INTERFACE: Final[str] = "FormalAssuranceOrchestrator@1"
ANALYZER_VERSION: Final[str] = "formal-assurance-orchestrator/v1"
TOOLCHAIN_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.proof.formal_assurance_orchestrator/"
    + ANALYZER_VERSION
)

# Explicit non-assurance stages — never admitted into the ladder.
PROHIBITED_ASSURANCE_STAGES: Final[frozenset[str]] = frozenset(
    {
        "llm",
        "llm_output",
        "language_model",
        "heuristic",
        "proposal_only",
    }
)


class OrchestratorError(ContractValidationError):
    """Malformed orchestration input or fail-closed policy violation."""


class EscalationStage(str, Enum):
    """Cheapest-sound ladder from RPS/PRM (plan §6).

    Order is authoritative for cost/escalation comparisons. ``ABSTRACT_INTERPRETATION``
    is the plan's "AI" rung and is **not** a language-model stage.
    """

    SCHEMA = "schema"
    ABSTRACT_INTERPRETATION = "abstract_interpretation"
    DATALOG = "datalog"
    EGRAPH = "egraph"
    SMT = "smt"
    ALLOY = "alloy"
    TLA = "tla"
    SPECIALIZED = "specialized"
    LEAN = "lean"
    HUMAN = "human"

    @property
    def rank(self) -> int:
        return _STAGE_RANK[self]

    @property
    def default_cost(self) -> int:
        """Integer cost units (canonical proof contracts forbid floats)."""

        return _STAGE_DEFAULT_COST[self]

    @property
    def authoritative(self) -> bool:
        """Whether a conclusive result at this stage may become verified."""

        return self in _AUTHORITATIVE_STAGES


_STAGE_ORDER: Final[tuple[EscalationStage, ...]] = tuple(EscalationStage)
_STAGE_RANK: Final[Mapping[EscalationStage, int]] = MappingProxyType(
    {stage: index for index, stage in enumerate(_STAGE_ORDER)}
)
_STAGE_DEFAULT_COST: Final[Mapping[EscalationStage, int]] = MappingProxyType(
    {
        EscalationStage.SCHEMA: 1,
        EscalationStage.ABSTRACT_INTERPRETATION: 2,
        EscalationStage.DATALOG: 3,
        EscalationStage.EGRAPH: 4,
        EscalationStage.SMT: 8,
        EscalationStage.ALLOY: 12,
        EscalationStage.TLA: 16,
        EscalationStage.SPECIALIZED: 20,
        EscalationStage.LEAN: 40,
        EscalationStage.HUMAN: 100,
    }
)
_AUTHORITATIVE_STAGES: Final[frozenset[EscalationStage]] = frozenset(
    {
        EscalationStage.SMT,
        EscalationStage.ALLOY,
        EscalationStage.TLA,
        EscalationStage.SPECIALIZED,
        EscalationStage.LEAN,
        EscalationStage.HUMAN,
    }
)

# Property-kind → cheapest sound starting stage (still may escalate).
_PROPERTY_ENTRY_STAGE: Final[Mapping[PropertyKind, EscalationStage]] = MappingProxyType(
    {
        PropertyKind.FINITE_CONSTRAINT: EscalationStage.SMT,
        PropertyKind.STATE_MACHINE: EscalationStage.TLA,
        PropertyKind.AUTHORIZATION: EscalationStage.DATALOG,
        PropertyKind.PROTOCOL: EscalationStage.SPECIALIZED,
        PropertyKind.HYPERPROPERTY: EscalationStage.SPECIALIZED,
        PropertyKind.RUNTIME_TRACE: EscalationStage.ABSTRACT_INTERPRETATION,
        PropertyKind.KERNEL_CHECK: EscalationStage.LEAN,
        PropertyKind.TYPED_PLANNING: EscalationStage.EGRAPH,
        PropertyKind.TEMPORAL_DEONTIC: EscalationStage.DATALOG,
        PropertyKind.FIRST_ORDER_THEOREM: EscalationStage.LEAN,
    }
)


class OrchestratorVerdict(str, Enum):
    """Closed orchestration outcome vocabulary."""

    VERIFIED = "verified"
    DISPROVED = "disproved"
    NONVERIFIED = "nonverified"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    CACHE_HIT = "cache_hit"


class StageOutcome(str, Enum):
    """Raw outcome reported by one ladder stage (pre-authority promotion)."""

    VERIFIED = "verified"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    CANDIDATE = "candidate"
    ERROR = "error"


class CacheReuseKind(str, Enum):
    """Formal derivation class for an admitted cache reuse."""

    UNCHANGED = "unchanged"
    EQUIVALENT = "equivalent"


class ConflictKind(str, Enum):
    STAGE_DISAGREEMENT = "stage_disagreement"
    AUTHORITY_DISAGREEMENT = "authority_disagreement"
    CANDIDATE_AUTHORITY_CLASH = "candidate_authority_clash"
    CACHE_CLOSURE_MISMATCH = "cache_closure_mismatch"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise OrchestratorError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise OrchestratorError(f"{name} must not be empty")
    return value


def _strings(values: Iterable[Any] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise OrchestratorError(f"{name} must be a sequence")
    seen: list[str] = []
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.append(text)
    return tuple(sorted(seen))


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise OrchestratorError(f"{name} must be an object with string keys")
    try:
        result = canonical_json(dict(value))
        return __import__("json").loads(result)
    except (TypeError, ValueError) as exc:
        raise OrchestratorError(f"{name} must be canonical JSON") from exc


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise OrchestratorError(f"{name} is unsupported: {value!r}") from exc


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Reject bools and floats: proof contracts require integer cost units.
        raise OrchestratorError(f"{name} must be a non-negative integer")
    if value < 0:
        raise OrchestratorError(f"{name} must be a non-negative integer")
    return value


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _reject_prohibited_stage(stage_name: str) -> None:
    normalized = stage_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in PROHIBITED_ASSURANCE_STAGES:
        raise OrchestratorError(
            f"prohibited assurance stage {stage_name!r}; LLM/heuristic stages "
            "are never admitted into the proof ladder"
        )


def escalation_ladder() -> tuple[EscalationStage, ...]:
    """Return the fixed cheapest-sound escalation ladder."""

    return _STAGE_ORDER


def stage_for_property(property_kind: PropertyKind | str) -> EscalationStage:
    """Cheapest sound entry stage for a reviewed property family."""

    kind = _enum(property_kind, PropertyKind, "property_kind")
    return _PROPERTY_ENTRY_STAGE[kind]


def next_stronger_stage(stage: EscalationStage | str) -> Optional[EscalationStage]:
    current = _enum(stage, EscalationStage, "stage")
    rank = current.rank
    if rank + 1 >= len(_STAGE_ORDER):
        return None
    return _STAGE_ORDER[rank + 1]


# ---------------------------------------------------------------------------
# Cache key (capsule-bound)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapsuleBoundProofCacheKey:
    """FACP proof-cache key bound to capsule / translation identity.

    Evidence subset: claim, spec, code, assumptions, environment, solver,
    revision, tactic — plus capsule and optional translation receipt CIDs that
    pin the semantic closure.
    """

    claim_id: str
    spec_id: str
    code_id: str
    assumptions: tuple[str, ...]
    environment_id: str
    solver_id: str
    revision_id: str
    tactic_id: str
    capsule_cid: str
    translation_receipt_cid: str = ""
    property_kind: str = ""
    semantic_closure_id: str = ""
    schema: str = PROOF_CACHE_KEY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "claim_id",
            "spec_id",
            "code_id",
            "environment_id",
            "solver_id",
            "revision_id",
            "tactic_id",
            "capsule_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(
            self,
            "translation_receipt_cid",
            _text(self.translation_receipt_cid, "translation_receipt_cid", required=False),
        )
        object.__setattr__(
            self,
            "property_kind",
            _text(self.property_kind, "property_kind", required=False),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROOF_CACHE_KEY_SCHEMA:
            raise OrchestratorError(
                f"unsupported proof-cache-key schema: {self.schema}"
            )
        closure = self.semantic_closure_id or self._derive_closure()
        object.__setattr__(self, "semantic_closure_id", _text(closure, "semantic_closure_id"))

    def _derive_closure(self) -> str:
        return _content_id(
            "semantic-closure",
            {
                "capsule_cid": self.capsule_cid,
                "translation_receipt_cid": self.translation_receipt_cid,
                "claim_id": self.claim_id,
                "spec_id": self.spec_id,
                "code_id": self.code_id,
                "assumptions": list(self.assumptions),
                "environment_id": self.environment_id,
                "revision_id": self.revision_id,
            },
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_id": self.claim_id,
            "spec_id": self.spec_id,
            "code_id": self.code_id,
            "assumptions": list(self.assumptions),
            "environment_id": self.environment_id,
            "solver_id": self.solver_id,
            "revision_id": self.revision_id,
            "tactic_id": self.tactic_id,
            "capsule_cid": self.capsule_cid,
            "translation_receipt_cid": self.translation_receipt_cid,
            "property_kind": self.property_kind,
            "semantic_closure_id": self.semantic_closure_id,
        }

    @property
    def key_id(self) -> str:
        return _content_id("facp-proof-cache-key", self.identity_payload())

    @property
    def cache_key(self) -> str:
        return self.key_id

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["key_id"] = self.key_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapsuleBoundProofCacheKey":
        if not isinstance(payload, Mapping):
            raise OrchestratorError("proof cache key must be an object")
        return cls(
            claim_id=str(payload.get("claim_id") or ""),
            spec_id=str(payload.get("spec_id") or ""),
            code_id=str(payload.get("code_id") or ""),
            assumptions=tuple(payload.get("assumptions") or ()),
            environment_id=str(payload.get("environment_id") or ""),
            solver_id=str(payload.get("solver_id") or ""),
            revision_id=str(payload.get("revision_id") or ""),
            tactic_id=str(payload.get("tactic_id") or ""),
            capsule_cid=str(payload.get("capsule_cid") or ""),
            translation_receipt_cid=str(payload.get("translation_receipt_cid") or ""),
            property_kind=str(payload.get("property_kind") or ""),
            semantic_closure_id=str(payload.get("semantic_closure_id") or ""),
            schema=str(payload.get("schema") or PROOF_CACHE_KEY_SCHEMA),
        )

    def to_legacy_proof_cache_key(self) -> ProofCacheKey:
        """Project into the existing trust-aware proof-cache key surface."""

        return build_proof_cache_key(
            obligation={
                "claim_id": self.claim_id,
                "spec_id": self.spec_id,
                "code_id": self.code_id,
                "property_kind": self.property_kind,
            },
            premises=list(self.assumptions),
            translator=self.translation_receipt_cid or "translator:none",
            solver=self.solver_id,
            kernel="kernel:facp-orchestrator",
            toolchain=TOOLCHAIN_ID,
            theorem_registry=self.revision_id,
            policy={"capsule_cid": self.capsule_cid, "tactic_id": self.tactic_id},
            resource_budget={"environment_id": self.environment_id},
            candidate_tree=self.semantic_closure_id,
        )


def build_capsule_bound_cache_key(
    *,
    claim_id: str,
    spec_id: str,
    code_id: str,
    assumptions: Sequence[str] | None = None,
    environment_id: str,
    solver_id: str,
    revision_id: str,
    tactic_id: str,
    capsule_cid: str,
    translation_receipt_cid: str = "",
    property_kind: PropertyKind | str = "",
) -> CapsuleBoundProofCacheKey:
    """Construct a capsule-bound FACP proof-cache key."""

    kind_text = ""
    if property_kind not in (None, ""):
        kind_text = _enum(property_kind, PropertyKind, "property_kind").value
    return CapsuleBoundProofCacheKey(
        claim_id=claim_id,
        spec_id=spec_id,
        code_id=code_id,
        assumptions=tuple(assumptions or ()),
        environment_id=environment_id,
        solver_id=solver_id,
        revision_id=revision_id,
        tactic_id=tactic_id,
        capsule_cid=capsule_cid,
        translation_receipt_cid=translation_receipt_cid,
        property_kind=kind_text,
    )


# ---------------------------------------------------------------------------
# Cache reuse derivation / conflict / escalation receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheReuseDerivation:
    """Formal explanation required before a cache hit may be reused."""

    kind: CacheReuseKind
    explanation: str
    prior_key_id: str
    current_key_id: str
    prior_closure_id: str
    current_closure_id: str
    path: tuple[str, ...] = ()
    schema: str = CACHE_REUSE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CacheReuseKind, "kind"))
        object.__setattr__(
            self, "explanation", _text(self.explanation, "explanation")
        )
        object.__setattr__(
            self, "prior_key_id", _text(self.prior_key_id, "prior_key_id")
        )
        object.__setattr__(
            self, "current_key_id", _text(self.current_key_id, "current_key_id")
        )
        object.__setattr__(
            self, "prior_closure_id", _text(self.prior_closure_id, "prior_closure_id")
        )
        object.__setattr__(
            self,
            "current_closure_id",
            _text(self.current_closure_id, "current_closure_id"),
        )
        object.__setattr__(self, "path", _strings(self.path, "path"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != CACHE_REUSE_SCHEMA:
            raise OrchestratorError(f"unsupported cache-reuse schema: {self.schema}")
        if self.kind is CacheReuseKind.UNCHANGED:
            if self.prior_key_id != self.current_key_id:
                raise OrchestratorError(
                    "unchanged reuse requires identical cache key identities"
                )
            if self.prior_closure_id != self.current_closure_id:
                raise OrchestratorError(
                    "unchanged reuse requires identical semantic closures"
                )
        elif self.kind is CacheReuseKind.EQUIVALENT:
            if self.prior_closure_id == self.current_closure_id and (
                self.prior_key_id == self.current_key_id
            ):
                raise OrchestratorError(
                    "equivalent reuse requires a distinct but justified derivation"
                )
            if not self.path:
                raise OrchestratorError(
                    "equivalent reuse requires a minimal derivation path"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "explanation": self.explanation,
            "prior_key_id": self.prior_key_id,
            "current_key_id": self.current_key_id,
            "prior_closure_id": self.prior_closure_id,
            "current_closure_id": self.current_closure_id,
            "path": list(self.path),
            "derivation_id": self.derivation_id,
        }

    @property
    def derivation_id(self) -> str:
        return _content_id(
            "facp-cache-reuse",
            {
                "kind": self.kind.value,
                "explanation": self.explanation,
                "prior_key_id": self.prior_key_id,
                "current_key_id": self.current_key_id,
                "prior_closure_id": self.prior_closure_id,
                "current_closure_id": self.current_closure_id,
                "path": list(self.path),
            },
        )


@dataclass(frozen=True)
class EscalationReceipt:
    """Reason and measured cost for escalating to a stronger ladder stage."""

    from_stage: EscalationStage
    to_stage: EscalationStage
    reason: str
    cost: int
    cumulative_cost: int
    schema: str = ESCALATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "from_stage", _enum(self.from_stage, EscalationStage, "from_stage")
        )
        object.__setattr__(
            self, "to_stage", _enum(self.to_stage, EscalationStage, "to_stage")
        )
        if self.to_stage.rank <= self.from_stage.rank:
            raise OrchestratorError(
                "escalation must move to a strictly stronger ladder stage"
            )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "cost", _nonneg_int(self.cost, "cost"))
        object.__setattr__(
            self,
            "cumulative_cost",
            _nonneg_int(self.cumulative_cost, "cumulative_cost"),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != ESCALATION_SCHEMA:
            raise OrchestratorError(f"unsupported escalation schema: {self.schema}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "reason": self.reason,
            "cost": self.cost,
            "cumulative_cost": self.cumulative_cost,
            "escalation_id": self.escalation_id,
        }

    @property
    def escalation_id(self) -> str:
        return _content_id(
            "facp-escalation",
            {
                "from_stage": self.from_stage.value,
                "to_stage": self.to_stage.value,
                "reason": self.reason,
                "cost": self.cost,
                "cumulative_cost": self.cumulative_cost,
            },
        )


@dataclass(frozen=True)
class SolverConflictRecord:
    """Explicit conflict when stages or authorities disagree."""

    kind: ConflictKind
    obligation_id: str
    left_stage: str
    right_stage: str
    left_outcome: str
    right_outcome: str
    explanation: str
    assumptions: tuple[str, ...] = ()
    verifier: str = TOOLCHAIN_ID
    toolchain: str = TOOLCHAIN_ID
    schema: str = SOLVER_CONFLICT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ConflictKind, "kind"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "left_stage", _text(self.left_stage, "left_stage"))
        object.__setattr__(self, "right_stage", _text(self.right_stage, "right_stage"))
        object.__setattr__(
            self, "left_outcome", _text(self.left_outcome, "left_outcome")
        )
        object.__setattr__(
            self, "right_outcome", _text(self.right_outcome, "right_outcome")
        )
        object.__setattr__(
            self, "explanation", _text(self.explanation, "explanation")
        )
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(self, "verifier", _text(self.verifier, "verifier"))
        object.__setattr__(self, "toolchain", _text(self.toolchain, "toolchain"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != SOLVER_CONFLICT_SCHEMA:
            raise OrchestratorError(
                f"unsupported solver-conflict schema: {self.schema}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "obligation_id": self.obligation_id,
            "left_stage": self.left_stage,
            "right_stage": self.right_stage,
            "left_outcome": self.left_outcome,
            "right_outcome": self.right_outcome,
            "explanation": self.explanation,
            "assumptions": list(self.assumptions),
            "verifier": self.verifier,
            "toolchain": self.toolchain,
            "conflict_id": self.conflict_id,
            "verdict": OrchestratorVerdict.CONFLICT.value,
        }

    @property
    def conflict_id(self) -> str:
        return _content_id(
            "facp-solver-conflict",
            {
                "kind": self.kind.value,
                "obligation_id": self.obligation_id,
                "left_stage": self.left_stage,
                "right_stage": self.right_stage,
                "left_outcome": self.left_outcome,
                "right_outcome": self.right_outcome,
                "explanation": self.explanation,
            },
        )


@dataclass(frozen=True)
class StageAttempt:
    """One recorded attempt on the escalation ladder."""

    stage: EscalationStage
    outcome: StageOutcome
    verifier: str
    cost: int
    detail: str = ""
    authoritative: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, EscalationStage, "stage"))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, StageOutcome, "outcome")
        )
        object.__setattr__(self, "verifier", _text(self.verifier, "verifier"))
        object.__setattr__(self, "cost", _nonneg_int(self.cost, "cost"))
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        object.__setattr__(self, "authoritative", bool(self.authoritative))
        object.__setattr__(
            self, "evidence", MappingProxyType(_mapping(self.evidence, "evidence"))
        )
        # Candidate / unknown / unavailable never flip authoritative.
        if self.outcome in (
            StageOutcome.UNKNOWN,
            StageOutcome.UNAVAILABLE,
            StageOutcome.CANDIDATE,
        ):
            object.__setattr__(self, "authoritative", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "verifier": self.verifier,
            "cost": self.cost,
            "detail": self.detail,
            "authoritative": self.authoritative,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class OrchestrationResult:
    """Authoritative orchestration outcome; always names identity bindings."""

    verdict: OrchestratorVerdict
    obligation_id: str
    assumptions: tuple[str, ...]
    verifier: str
    toolchain: str
    cache_key: CapsuleBoundProofCacheKey
    stage: EscalationStage
    assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    attempts: tuple[StageAttempt, ...] = ()
    escalations: tuple[EscalationReceipt, ...] = ()
    conflict: Optional[SolverConflictRecord] = None
    cache_reuse: Optional[CacheReuseDerivation] = None
    reason: str = ""
    cumulative_cost: int = 0
    schema: str = RESULT_SCHEMA
    router_schema: str = PROOF_ROUTER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum(self.verdict, OrchestratorVerdict, "verdict")
        )
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(self, "verifier", _text(self.verifier, "verifier"))
        object.__setattr__(self, "toolchain", _text(self.toolchain, "toolchain"))
        if not isinstance(self.cache_key, CapsuleBoundProofCacheKey):
            object.__setattr__(
                self, "cache_key", CapsuleBoundProofCacheKey.from_dict(self.cache_key)
            )
        object.__setattr__(self, "stage", _enum(self.stage, EscalationStage, "stage"))
        object.__setattr__(
            self, "assurance", _enum(self.assurance, AssuranceLevel, "assurance")
        )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "escalations", tuple(self.escalations))
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "cumulative_cost",
            _nonneg_int(self.cumulative_cost, "cumulative_cost"),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "router_schema", _text(self.router_schema, "router_schema")
        )
        if self.schema != RESULT_SCHEMA:
            raise OrchestratorError(f"unsupported result schema: {self.schema}")
        if self.router_schema != PROOF_ROUTER_SCHEMA:
            raise OrchestratorError(
                f"unsupported proof-router schema: {self.router_schema}"
            )
        # Fail-closed promotions.
        if self.verdict is OrchestratorVerdict.VERIFIED:
            if self.assurance.rank < AssuranceLevel.SOLVER_CHECKED.rank:
                raise OrchestratorError(
                    "verified verdict requires at least solver_checked assurance"
                )
            if any(
                attempt.outcome
                in (StageOutcome.UNKNOWN, StageOutcome.UNAVAILABLE)
                and attempt.authoritative
                for attempt in self.attempts
            ):
                raise OrchestratorError(
                    "unknown/unavailable attempts cannot be authoritative for verified"
                )
        if self.verdict is OrchestratorVerdict.CONFLICT and self.conflict is None:
            raise OrchestratorError("conflict verdict requires a conflict record")
        if self.verdict is OrchestratorVerdict.CACHE_HIT and self.cache_reuse is None:
            raise OrchestratorError("cache_hit verdict requires a reuse derivation")

    @property
    def nonverified(self) -> bool:
        return self.verdict in {
            OrchestratorVerdict.NONVERIFIED,
            OrchestratorVerdict.CONFLICT,
            OrchestratorVerdict.UNSUPPORTED,
        }

    @property
    def result_id(self) -> str:
        return _content_id(
            "facp-orchestration-result",
            {
                "verdict": self.verdict.value,
                "obligation_id": self.obligation_id,
                "assumptions": list(self.assumptions),
                "verifier": self.verifier,
                "toolchain": self.toolchain,
                "cache_key_id": self.cache_key.key_id,
                "stage": self.stage.value,
                "assurance": self.assurance.value,
                "reason": self.reason,
                "cumulative_cost": self.cumulative_cost,
                "conflict_id": None if self.conflict is None else self.conflict.conflict_id,
                "cache_reuse_id": (
                    None if self.cache_reuse is None else self.cache_reuse.derivation_id
                ),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "router_schema": self.router_schema,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "interface": INTERFACE,
            "analyzer_version": ANALYZER_VERSION,
            "result_id": self.result_id,
            "verdict": self.verdict.value,
            "obligation_id": self.obligation_id,
            "assumptions": list(self.assumptions),
            "verifier": self.verifier,
            "toolchain": self.toolchain,
            "cache_key": self.cache_key.to_dict(),
            "stage": self.stage.value,
            "assurance": self.assurance.value,
            "attempts": [item.to_dict() for item in self.attempts],
            "escalations": [item.to_dict() for item in self.escalations],
            "conflict": None if self.conflict is None else self.conflict.to_dict(),
            "cache_reuse": (
                None if self.cache_reuse is None else self.cache_reuse.to_dict()
            ),
            "reason": self.reason,
            "cumulative_cost": self.cumulative_cost,
            "nonverified": self.nonverified,
        }


# ---------------------------------------------------------------------------
# In-memory capsule-bound proof cache (composes with legacy key projection)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    key: CapsuleBoundProofCacheKey
    result: OrchestrationResult
    legacy_key_id: str


class CapsuleBoundProofCache:
    """Process-local cache that refuses reuse on changed semantic closures."""

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._by_closure: dict[str, set[str]] = {}

    def clear(self) -> None:
        self._entries.clear()
        self._by_closure.clear()

    def store(self, result: OrchestrationResult) -> str:
        if result.verdict in (
            OrchestratorVerdict.NONVERIFIED,
            OrchestratorVerdict.CONFLICT,
            OrchestratorVerdict.UNSUPPORTED,
        ):
            # Non-conclusive outcomes are never cached as reusable proofs.
            return ""
        if result.verdict is OrchestratorVerdict.CACHE_HIT:
            raise OrchestratorError("cannot store a cache_hit as a new cache entry")
        key = result.cache_key
        legacy = key.to_legacy_proof_cache_key()
        entry = _CacheEntry(key=key, result=result, legacy_key_id=legacy.key_id)
        self._entries[key.key_id] = entry
        self._by_closure.setdefault(key.semantic_closure_id, set()).add(key.key_id)
        return key.key_id

    def lookup(
        self,
        key: CapsuleBoundProofCacheKey,
        *,
        equivalence: Optional[CacheReuseDerivation] = None,
    ) -> tuple[CacheLookupStatus, Optional[OrchestrationResult], Optional[CacheReuseDerivation]]:
        exact = self._entries.get(key.key_id)
        if exact is not None:
            if exact.key.semantic_closure_id != key.semantic_closure_id:
                return CacheLookupStatus.REJECTED, None, None
            derivation = CacheReuseDerivation(
                kind=CacheReuseKind.UNCHANGED,
                explanation=(
                    "semantic closure, claim/spec/code, assumptions, environment, "
                    "solver, revision, and tactic identities are unchanged"
                ),
                prior_key_id=exact.key.key_id,
                current_key_id=key.key_id,
                prior_closure_id=exact.key.semantic_closure_id,
                current_closure_id=key.semantic_closure_id,
                path=("cache_key", "semantic_closure"),
            )
            reused = self._as_cache_hit(exact.result, key, derivation)
            return CacheLookupStatus.HIT, reused, derivation

        if equivalence is None:
            # Distinct key without a formal equivalence derivation never reuses
            # a prior entry — including when the capsule/closure changed.
            return CacheLookupStatus.MISS, None, None

        if equivalence.kind is not CacheReuseKind.EQUIVALENT:
            raise OrchestratorError(
                "non-exact lookup requires an equivalent reuse derivation"
            )
        if equivalence.current_key_id != key.key_id:
            raise OrchestratorError(
                "equivalence derivation current_key_id must match lookup key"
            )
        prior = self._entries.get(equivalence.prior_key_id)
        if prior is None:
            return CacheLookupStatus.MISS, None, None
        if prior.key.semantic_closure_id != equivalence.prior_closure_id:
            return CacheLookupStatus.REJECTED, None, None
        if key.semantic_closure_id != equivalence.current_closure_id:
            return CacheLookupStatus.REJECTED, None, None
        # Refuse silent reuse across a changed closure without an equivalence path
        # that actually names the closure transition.
        if (
            prior.key.semantic_closure_id != key.semantic_closure_id
            and "semantic_closure" not in equivalence.path
            and "closure" not in equivalence.path
        ):
            return CacheLookupStatus.REJECTED, None, None
        # Equivalence may rewrite presentation identity but not weaken claim/code.
        if prior.key.claim_id != key.claim_id or prior.key.code_id != key.code_id:
            return CacheLookupStatus.REJECTED, None, None
        reused = self._as_cache_hit(prior.result, key, equivalence)
        return CacheLookupStatus.HIT, reused, equivalence

    @staticmethod
    def _as_cache_hit(
        prior: OrchestrationResult,
        current_key: CapsuleBoundProofCacheKey,
        derivation: CacheReuseDerivation,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            verdict=OrchestratorVerdict.CACHE_HIT,
            obligation_id=prior.obligation_id,
            assumptions=prior.assumptions,
            verifier=prior.verifier,
            toolchain=prior.toolchain,
            cache_key=current_key,
            stage=prior.stage,
            assurance=prior.assurance,
            attempts=prior.attempts,
            escalations=prior.escalations,
            conflict=None,
            cache_reuse=derivation,
            reason=f"cache reuse ({derivation.kind.value}): {derivation.explanation}",
            cumulative_cost=prior.cumulative_cost,
        )


# ---------------------------------------------------------------------------
# Stage runners / routing
# ---------------------------------------------------------------------------

StageRunner = Callable[[EscalationStage, PropertyObligation, CapsuleBoundProofCacheKey], StageAttempt]


def _default_stage_runner(
    stage: EscalationStage,
    obligation: PropertyObligation,
    key: CapsuleBoundProofCacheKey,
) -> StageAttempt:
    """Hermetic default: schema validates; deeper stages report unavailable."""

    del key  # identity already bound by the caller
    if stage is EscalationStage.SCHEMA:
        if not obligation.statement.strip():
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.UNSUPPORTED,
                verifier="schema-validator",
                cost=stage.default_cost,
                detail="empty obligation statement",
                authoritative=False,
            )
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.CANDIDATE,
            verifier="schema-validator",
            cost=stage.default_cost,
            detail="schema-valid; not an authority proof",
            authoritative=False,
        )
    if stage is EscalationStage.HUMAN:
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.UNKNOWN,
            verifier="human-review-gate",
            cost=stage.default_cost,
            detail="human review required; remains nonverified until recorded",
            authoritative=False,
        )
    return StageAttempt(
        stage=stage,
        outcome=StageOutcome.UNAVAILABLE,
        verifier=f"{stage.value}-adapter",
        cost=stage.default_cost,
        detail=f"{stage.value} backend unavailable in hermetic orchestrator",
        authoritative=False,
    )


def _assurance_for_verdict(
    verdict: OrchestratorVerdict, stage: EscalationStage
) -> AssuranceLevel:
    if verdict is OrchestratorVerdict.VERIFIED:
        if stage is EscalationStage.LEAN:
            return AssuranceLevel.KERNEL_VERIFIED
        if stage in _AUTHORITATIVE_STAGES:
            return AssuranceLevel.SOLVER_CHECKED
        return AssuranceLevel.UNVERIFIED
    if verdict is OrchestratorVerdict.DISPROVED:
        return AssuranceLevel.SOLVER_CHECKED
    if verdict is OrchestratorVerdict.CACHE_HIT:
        return AssuranceLevel.SOLVER_CHECKED
    return AssuranceLevel.UNVERIFIED


def _conclusive_outcomes() -> frozenset[StageOutcome]:
    return frozenset({StageOutcome.VERIFIED, StageOutcome.DISPROVED})


def build_conflict(
    *,
    obligation_id: str,
    left: StageAttempt,
    right: StageAttempt,
    assumptions: Sequence[str],
    kind: ConflictKind = ConflictKind.STAGE_DISAGREEMENT,
    explanation: str = "",
) -> SolverConflictRecord:
    detail = explanation or (
        f"disagreement between {left.stage.value} ({left.outcome.value}) and "
        f"{right.stage.value} ({right.outcome.value})"
    )
    return SolverConflictRecord(
        kind=kind,
        obligation_id=obligation_id,
        left_stage=left.stage.value,
        right_stage=right.stage.value,
        left_outcome=left.outcome.value,
        right_outcome=right.outcome.value,
        explanation=detail,
        assumptions=tuple(assumptions),
        verifier=right.verifier or left.verifier or TOOLCHAIN_ID,
        toolchain=TOOLCHAIN_ID,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FormalAssuranceOrchestrator:
    """Route obligations through cache + cheapest-sound escalation ladder."""

    def __init__(
        self,
        *,
        cache: CapsuleBoundProofCache | None = None,
        router: MultiProverRouter | None = None,
        stage_runner: StageRunner | None = None,
        toolchain: str = TOOLCHAIN_ID,
    ) -> None:
        self._cache = cache if cache is not None else CapsuleBoundProofCache()
        self._router = router if router is not None else MultiProverRouter()
        self._stage_runner = stage_runner or _default_stage_runner
        self._toolchain = _text(toolchain, "toolchain")

    @property
    def cache(self) -> CapsuleBoundProofCache:
        return self._cache

    @property
    def router(self) -> MultiProverRouter:
        return self._router

    @property
    def toolchain(self) -> str:
        return self._toolchain

    def plan_route(
        self,
        obligation: PropertyObligation | Mapping[str, Any],
        *,
        minimum_stage: EscalationStage | str | None = None,
    ) -> dict[str, Any]:
        """Return the ordered ladder plan for an obligation (proof-router@1)."""

        obl = (
            obligation
            if isinstance(obligation, PropertyObligation)
            else PropertyObligation.from_dict(obligation)
        )
        entry = stage_for_property(obl.property_kind)
        if minimum_stage is not None:
            floor = _enum(minimum_stage, EscalationStage, "minimum_stage")
            if floor.rank > entry.rank:
                entry = floor
        stages = [stage for stage in _STAGE_ORDER if stage.rank >= entry.rank]
        payload = {
            "schema": PROOF_ROUTER_SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "obligation_id": obl.obligation_id,
            "property_kind": obl.property_kind.value,
            "entry_stage": entry.value,
            "ladder": [stage.value for stage in stages],
            "prohibited_stages": sorted(PROHIBITED_ASSURANCE_STAGES),
            "toolchain": self._toolchain,
            "analyzer_version": ANALYZER_VERSION,
        }
        payload["plan_id"] = _content_id("facp-proof-router", payload)
        return payload

    def escalate(
        self,
        from_stage: EscalationStage | str,
        *,
        reason: str,
        cumulative_cost: int = 0,
        cost: int | None = None,
        to_stage: EscalationStage | str | None = None,
    ) -> EscalationReceipt:
        """Build a stronger-stage escalation receipt with reason and cost."""

        current = _enum(from_stage, EscalationStage, "from_stage")
        if to_stage is None:
            stronger = next_stronger_stage(current)
            if stronger is None:
                raise OrchestratorError(
                    f"no stronger stage above {current.value}"
                )
        else:
            stronger = _enum(to_stage, EscalationStage, "to_stage")
            _reject_prohibited_stage(stronger.value)
        delta = _nonneg_int(
            cost if cost is not None else stronger.default_cost, "cost"
        )
        total = _nonneg_int(cumulative_cost, "cumulative_cost") + delta
        return EscalationReceipt(
            from_stage=current,
            to_stage=stronger,
            reason=reason,
            cost=delta,
            cumulative_cost=total,
        )

    def execute(
        self,
        *,
        obligation: PropertyObligation | Mapping[str, Any],
        cache_key: CapsuleBoundProofCacheKey | Mapping[str, Any],
        stage_runner: StageRunner | None = None,
        equivalence: Optional[CacheReuseDerivation] = None,
        stop_on_verified: bool = True,
        max_stage: EscalationStage | str | None = None,
        store_on_success: bool = True,
    ) -> OrchestrationResult:
        """Lookup cache, then escalate through the ladder until conclusive."""

        obl = (
            obligation
            if isinstance(obligation, PropertyObligation)
            else PropertyObligation.from_dict(obligation)
        )
        key = (
            cache_key
            if isinstance(cache_key, CapsuleBoundProofCacheKey)
            else CapsuleBoundProofCacheKey.from_dict(cache_key)
        )
        assumptions = _strings(
            tuple(obl.premise_ids) + tuple(key.assumptions), "assumptions"
        )
        runner = stage_runner or self._stage_runner

        status, cached, derivation = self._cache.lookup(key, equivalence=equivalence)
        if status is CacheLookupStatus.HIT and cached is not None:
            return cached
        if status is CacheLookupStatus.REJECTED:
            conflict = SolverConflictRecord(
                kind=ConflictKind.CACHE_CLOSURE_MISMATCH,
                obligation_id=obl.obligation_id,
                left_stage="cache",
                right_stage="request",
                left_outcome="stale_or_changed_closure",
                right_outcome="current_closure",
                explanation=(
                    "cache reuse refused because the semantic closure changed "
                    "and no formal equivalence derivation was supplied"
                ),
                assumptions=assumptions,
                verifier=self._toolchain,
                toolchain=self._toolchain,
            )
            return OrchestrationResult(
                verdict=OrchestratorVerdict.CONFLICT,
                obligation_id=obl.obligation_id,
                assumptions=assumptions,
                verifier=self._toolchain,
                toolchain=self._toolchain,
                cache_key=key,
                stage=stage_for_property(obl.property_kind),
                assurance=AssuranceLevel.UNVERIFIED,
                conflict=conflict,
                reason=conflict.explanation,
                cumulative_cost=0,
            )

        plan = self.plan_route(obl)
        ceiling = (
            _enum(max_stage, EscalationStage, "max_stage")
            if max_stage is not None
            else EscalationStage.HUMAN
        )
        attempts: list[StageAttempt] = []
        escalations: list[EscalationReceipt] = []
        cumulative = 0
        prior_conclusive: StageAttempt | None = None
        last_stage = _enum(plan["entry_stage"], EscalationStage, "entry_stage")

        for stage_name in plan["ladder"]:
            stage = _enum(stage_name, EscalationStage, "stage")
            if stage.rank > ceiling.rank:
                break
            if attempts:
                receipt = self.escalate(
                    last_stage,
                    to_stage=stage,
                    reason=(
                        f"{last_stage.value} inconclusive "
                        f"({attempts[-1].outcome.value}); escalate for soundness"
                    ),
                    cumulative_cost=cumulative,
                    cost=stage.default_cost,
                )
                escalations.append(receipt)
                cumulative = receipt.cumulative_cost
            else:
                cumulative += stage.default_cost

            attempt = runner(stage, obl, key)
            if not isinstance(attempt, StageAttempt):
                raise OrchestratorError("stage_runner must return StageAttempt")
            if attempt.stage is not stage:
                raise OrchestratorError("stage_runner returned mismatched stage")
            # Unknown / unavailable / candidate stay non-authoritative.
            if attempt.outcome in (
                StageOutcome.UNKNOWN,
                StageOutcome.UNAVAILABLE,
                StageOutcome.CANDIDATE,
            ):
                attempt = StageAttempt(
                    stage=attempt.stage,
                    outcome=attempt.outcome,
                    verifier=attempt.verifier,
                    cost=attempt.cost,
                    detail=attempt.detail,
                    authoritative=False,
                    evidence=dict(attempt.evidence),
                )
            elif attempt.authoritative and not stage.authoritative:
                raise OrchestratorError(
                    f"stage {stage.value} cannot claim authoritative verification"
                )
            attempts.append(attempt)
            last_stage = stage

            if attempt.outcome in _conclusive_outcomes():
                if prior_conclusive is not None and (
                    prior_conclusive.outcome != attempt.outcome
                ):
                    conflict = build_conflict(
                        obligation_id=obl.obligation_id,
                        left=prior_conclusive,
                        right=attempt,
                        assumptions=assumptions,
                        kind=ConflictKind.AUTHORITY_DISAGREEMENT,
                    )
                    return OrchestrationResult(
                        verdict=OrchestratorVerdict.CONFLICT,
                        obligation_id=obl.obligation_id,
                        assumptions=assumptions,
                        verifier=attempt.verifier,
                        toolchain=self._toolchain,
                        cache_key=key,
                        stage=stage,
                        assurance=AssuranceLevel.UNVERIFIED,
                        attempts=tuple(attempts),
                        escalations=tuple(escalations),
                        conflict=conflict,
                        reason=conflict.explanation,
                        cumulative_cost=cumulative,
                    )
                if not attempt.authoritative or not stage.authoritative:
                    # Solver/candidate-style conclusive report without authority.
                    if attempt.outcome is StageOutcome.VERIFIED:
                        # Cannot self-promote.
                        continue
                prior_conclusive = attempt
                if stop_on_verified and attempt.authoritative and stage.authoritative:
                    verdict = (
                        OrchestratorVerdict.VERIFIED
                        if attempt.outcome is StageOutcome.VERIFIED
                        else OrchestratorVerdict.DISPROVED
                    )
                    result = OrchestrationResult(
                        verdict=verdict,
                        obligation_id=obl.obligation_id,
                        assumptions=assumptions,
                        verifier=attempt.verifier,
                        toolchain=self._toolchain,
                        cache_key=key,
                        stage=stage,
                        assurance=_assurance_for_verdict(verdict, stage),
                        attempts=tuple(attempts),
                        escalations=tuple(escalations),
                        reason=attempt.detail or f"{stage.value} conclusive",
                        cumulative_cost=cumulative,
                    )
                    if store_on_success:
                        self._cache.store(result)
                    return result

        # No authoritative conclusive result.
        reason = "ladder exhausted without authoritative conclusive result"
        if attempts and attempts[-1].outcome is StageOutcome.UNSUPPORTED:
            verdict = OrchestratorVerdict.UNSUPPORTED
            reason = attempts[-1].detail or reason
        else:
            verdict = OrchestratorVerdict.NONVERIFIED
            if attempts:
                reason = (
                    f"terminal outcome {attempts[-1].outcome.value} remains nonverified"
                )
        return OrchestrationResult(
            verdict=verdict,
            obligation_id=obl.obligation_id,
            assumptions=assumptions,
            verifier=attempts[-1].verifier if attempts else self._toolchain,
            toolchain=self._toolchain,
            cache_key=key,
            stage=last_stage,
            assurance=AssuranceLevel.UNVERIFIED,
            attempts=tuple(attempts),
            escalations=tuple(escalations),
            reason=reason,
            cumulative_cost=cumulative,
        )

    def execute_with_portfolio(
        self,
        *,
        obligation: PropertyObligation | Mapping[str, Any],
        cache_key: CapsuleBoundProofCacheKey | Mapping[str, Any],
        portfolio_runner: Callable[..., ProverOutput],
    ) -> OrchestrationResult:
        """Compose :class:`MultiProverRouter` without treating candidates as proofs."""

        obl = (
            obligation
            if isinstance(obligation, PropertyObligation)
            else PropertyObligation.from_dict(obligation)
        )
        key = (
            cache_key
            if isinstance(cache_key, CapsuleBoundProofCacheKey)
            else CapsuleBoundProofCacheKey.from_dict(cache_key)
        )
        assumptions = _strings(
            tuple(obl.premise_ids) + tuple(key.assumptions), "assumptions"
        )
        portfolio: PortfolioResult = self._router.execute(obl, portfolio_runner)

        # Detect authority-reported disagreement even when the router demotes one
        # side's effective outcome; FACP-050 requires an explicit conflict receipt.
        authority_attempts = [
            item
            for item in portfolio.attempts
            if item.role.authoritative or item.authoritative
        ]
        verified_reports = [
            item
            for item in authority_attempts
            if item.reported_outcome is AttemptOutcome.VERIFIED
        ]
        counterexample_reports = [
            item
            for item in authority_attempts
            if item.reported_outcome is AttemptOutcome.COUNTEREXAMPLE
        ]
        if (
            portfolio.disagreement
            or (verified_reports and counterexample_reports)
        ):
            left = verified_reports[0] if verified_reports else authority_attempts[0]
            right = (
                counterexample_reports[0]
                if counterexample_reports
                else authority_attempts[-1]
            )
            conflict = SolverConflictRecord(
                kind=ConflictKind.AUTHORITY_DISAGREEMENT,
                obligation_id=obl.obligation_id,
                left_stage=left.prover_id,
                right_stage=right.prover_id,
                left_outcome=left.reported_outcome.value,
                right_outcome=right.reported_outcome.value,
                explanation=(
                    portfolio.reason
                    or "multi-prover authority lanes reported disagreeing conclusive outcomes"
                ),
                assumptions=assumptions,
                verifier="multi-prover-router",
                toolchain=self._toolchain,
            )
            return OrchestrationResult(
                verdict=OrchestratorVerdict.CONFLICT,
                obligation_id=obl.obligation_id,
                assumptions=assumptions,
                verifier="multi-prover-router",
                toolchain=self._toolchain,
                cache_key=key,
                stage=stage_for_property(obl.property_kind),
                assurance=AssuranceLevel.UNVERIFIED,
                conflict=conflict,
                reason=conflict.explanation,
                cumulative_cost=int(portfolio.duration_ms or 0),
            )

        # Map portfolio into orchestration vocabulary — candidates stay nonverified.
        if portfolio.verdict is PortfolioVerdict.PROVED:
            stage = EscalationStage.LEAN
            for attempt in portfolio.attempts:
                if (
                    attempt.role in (ProverRole.KERNEL, ProverRole.MODEL_CHECKER)
                    and attempt.effective_outcome is AttemptOutcome.VERIFIED
                ):
                    stage = (
                        EscalationStage.LEAN
                        if attempt.role is ProverRole.KERNEL
                        else EscalationStage.TLA
                    )
                    break
            result = OrchestrationResult(
                verdict=OrchestratorVerdict.VERIFIED,
                obligation_id=obl.obligation_id,
                assumptions=assumptions,
                verifier=portfolio.authority_attempt_ids[0]
                if portfolio.authority_attempt_ids
                else "multi-prover-router",
                toolchain=self._toolchain,
                cache_key=key,
                stage=stage,
                assurance=portfolio.assurance
                if isinstance(portfolio.assurance, AssuranceLevel)
                else AssuranceLevel.SOLVER_CHECKED,
                reason=portfolio.reason or "portfolio proved under reconstruction",
                cumulative_cost=int(portfolio.duration_ms or 0),
            )
            self._cache.store(result)
            return result

        if portfolio.verdict is PortfolioVerdict.DISPROVED:
            return OrchestrationResult(
                verdict=OrchestratorVerdict.DISPROVED,
                obligation_id=obl.obligation_id,
                assumptions=assumptions,
                verifier="multi-prover-router",
                toolchain=self._toolchain,
                cache_key=key,
                stage=EscalationStage.SMT,
                assurance=AssuranceLevel.SOLVER_CHECKED,
                reason=portfolio.reason or "portfolio disproved",
                cumulative_cost=int(portfolio.duration_ms or 0),
            )

        verdict = (
            OrchestratorVerdict.UNSUPPORTED
            if portfolio.verdict is PortfolioVerdict.UNSUPPORTED
            else OrchestratorVerdict.NONVERIFIED
        )
        return OrchestrationResult(
            verdict=verdict,
            obligation_id=obl.obligation_id,
            assumptions=assumptions,
            verifier="multi-prover-router",
            toolchain=self._toolchain,
            cache_key=key,
            stage=stage_for_property(obl.property_kind),
            assurance=AssuranceLevel.UNVERIFIED,
            reason=portfolio.reason or "portfolio inconclusive; remains nonverified",
            cumulative_cost=int(portfolio.duration_ms or 0),
        )


def orchestrate_obligation(
    *,
    obligation: PropertyObligation | Mapping[str, Any],
    cache_key: CapsuleBoundProofCacheKey | Mapping[str, Any],
    stage_runner: StageRunner | None = None,
    equivalence: Optional[CacheReuseDerivation] = None,
) -> OrchestrationResult:
    """Module-level convenience entry point for FACP-050 orchestration."""

    return FormalAssuranceOrchestrator().execute(
        obligation=obligation,
        cache_key=cache_key,
        stage_runner=stage_runner,
        equivalence=equivalence,
    )
