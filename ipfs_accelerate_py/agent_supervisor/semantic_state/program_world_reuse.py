"""Exact program-world reuse gates for SAWM-016.

Reuse is granted only for a current exact key match or independently
proof-backed scoped evidence. Similarity/ANN is context-only. Stale,
unsafe, incomplete, typed-unavailable, or environment-incompatible
evidence fails closed and explains why.

This module cannot self-admit operational acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ANN_REASON_CODES,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    ProgramWorldAdmissionError,
    ProgramWorldReuseDecision,
    ReuseVerdict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


ADAPTER_ID: Final[str] = "ipfs-accelerate.semantic-state.program-world-reuse"
PROGRAM_WORLD_REUSE_KEY_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-key@1"
)
NEGATIVE_MEMORY_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-negative-memory@1"
)

BINDING_FIELDS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
    "procedure_revision_cid",
)

REASON_EXACT_HIT: Final[str] = "exact_identity_match"
REASON_BINDING_MISMATCH: Final[str] = "single_binding_mismatch"
REASON_STALE: Final[str] = "stale_evidence_revoked"
REASON_UNSAFE_ABSTRACTION: Final[str] = "unsafe_abstraction"
REASON_PROOF_CACHE: Final[str] = "proof_cache_fresh_admission"
REASON_SIMILARITY: Final[str] = "similarity_is_not_reuse"
REASON_TYPED_UNAVAILABLE: Final[str] = "typed_surface_unavailable"
REASON_RAW_FALLBACK: Final[str] = "raw_source_fallback"
REASON_NEGATIVE_MEMORY: Final[str] = "negative_memory_invalidation"
REASON_INCOMPLETE: Final[str] = "incomplete_reuse_key"
REASON_ENVIRONMENT: Final[str] = "environment_incompatible"

_MAX_BINDINGS: Final[int] = 64


class ProgramWorldReuseError(HarnessError):
    """Closed reuse-gate contract violation."""


class ProgramWorldReuseRejection(ProgramWorldReuseError):
    """Deterministic closed-form refusal of a reuse query."""

    def __init__(self, reason_code: str, *, mismatches: Sequence[str] = ()) -> None:
        self.reason_code = str(reason_code)
        self.mismatches = tuple(mismatches)
        super().__init__(self.reason_code)


def _cid(value: Any, name: str) -> str:
    return validate_opaque_cid(value, name)


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _cid_tuple(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProgramWorldReuseError(f"{name} must be a sequence of CIDs")
    items = [_cid(item, name) for item in values]
    if len(items) > _MAX_BINDINGS:
        raise ProgramWorldReuseError(f"{name} exceeds {_MAX_BINDINGS} items")
    ordered = tuple(sorted(set(items)))
    return ordered


def _generation(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgramWorldReuseError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseKey:
    """Exact reuse key. Every listed binding participates in identity."""

    state_cid: str
    goal_cid: str
    policy_cid: str
    environment_cid: str
    toolchain_cid: str
    procedure_revision_cid: str
    obligation_cids: tuple[str, ...] = ()
    selection_cids: tuple[str, ...] = ()
    validation_dependency_cids: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_KEY_SCHEMA

    def __post_init__(self) -> None:
        for name in BINDING_FIELDS:
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        object.__setattr__(self, "obligation_cids", _cid_tuple(self.obligation_cids, "obligation_cids"))
        object.__setattr__(self, "selection_cids", _cid_tuple(self.selection_cids, "selection_cids"))
        object.__setattr__(
            self,
            "validation_dependency_cids",
            _cid_tuple(self.validation_dependency_cids, "validation_dependency_cids"),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "procedure_revision_cid": self.procedure_revision_cid,
            "obligation_cids": list(self.obligation_cids),
            "selection_cids": list(self.selection_cids),
            "validation_dependency_cids": list(self.validation_dependency_cids),
        }

    @property
    def key_cid(self) -> str:
        return cid_for_payload(self.identity_payload())

    def mismatches(self, other: "ProgramWorldReuseKey") -> tuple[str, ...]:
        found: list[str] = []
        for name in BINDING_FIELDS:
            if getattr(self, name) != getattr(other, name):
                found.append(name)
        for name in ("obligation_cids", "selection_cids", "validation_dependency_cids"):
            if getattr(self, name) != getattr(other, name):
                found.append(name)
        return tuple(found)


@dataclass(frozen=True, slots=True)
class NegativeMemoryRecord:
    """Scoped invalidation of a previously seen reuse key."""

    key_cid: str
    reason_code: str
    generation: int
    mismatches: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = NEGATIVE_MEMORY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", _cid(self.key_cid, "key_cid"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))
        object.__setattr__(self, "generation", _generation(self.generation, "generation"))
        object.__setattr__(
            self,
            "mismatches",
            tuple(_text(item, "mismatch") for item in self.mismatches),
        )


@dataclass(frozen=True, slots=True)
class CachedReuseEvidence:
    """Landed cache row. Never self-admits."""

    key: ProgramWorldReuseKey
    generation: int
    proof_cid: str | None = None
    abstraction_safe: bool = True
    typed_available: bool = True
    raw_source: bool = False
    environment_compatible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProgramWorldReuseKey):
            raise ProgramWorldReuseError("cached evidence requires a reuse key")
        object.__setattr__(self, "generation", _generation(self.generation, "generation"))
        object.__setattr__(self, "proof_cid", _optional_cid(self.proof_cid, "proof_cid"))
        object.__setattr__(self, "abstraction_safe", _bool(self.abstraction_safe, "abstraction_safe"))
        object.__setattr__(self, "typed_available", _bool(self.typed_available, "typed_available"))
        object.__setattr__(self, "raw_source", _bool(self.raw_source, "raw_source"))
        object.__setattr__(
            self,
            "environment_compatible",
            _bool(self.environment_compatible, "environment_compatible"),
        )


@dataclass
class ProgramWorldReuseGate:
    """Deterministic reuse evaluator with negative memory."""

    current_generation: int = 0
    _cache: dict[str, CachedReuseEvidence] = field(default_factory=dict)
    _negative: dict[str, NegativeMemoryRecord] = field(default_factory=dict)

    def remember(self, evidence: CachedReuseEvidence) -> None:
        self._cache[evidence.key.key_cid] = evidence

    def invalidate(self, record: NegativeMemoryRecord) -> None:
        self._negative[record.key_cid] = record
        self._cache.pop(record.key_cid, None)

    def evaluate_program_world_reuse(
        self,
        query: ProgramWorldReuseKey,
        *,
        typed_available: bool = True,
        similarity_candidates: Sequence[Any] = (),
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
        current_generation: int | None = None,
    ) -> ProgramWorldReuseDecision:
        generation = self.current_generation if current_generation is None else current_generation
        generation = _generation(generation, "current_generation")
        if admission_authority == ADAPTER_ID:
            raise ProgramWorldAdmissionError("reuse gate cannot admit its own output")
        if not typed_available:
            return self._reject(
                query,
                REASON_TYPED_UNAVAILABLE,
                fallback=True,
                limitations=("typed_surface_unavailable", "raw_source_is_not_reuse"),
            )
        if similarity_candidates:
            reason = REASON_SIMILARITY
            if any(
                isinstance(item, Mapping)
                and (set(item) & {"score", "ann_score", "nearest", "similarity"})
                for item in similarity_candidates
            ):
                reason = "ann_not_authoritative"
            return self._reject(
                query,
                reason,
                limitations=("similarity_is_context_only", "ann_advisory_only"),
            )
        negative = self._negative.get(query.key_cid)
        if negative is not None:
            return self._reject(
                query,
                REASON_NEGATIVE_MEMORY,
                limitations=("negative_memory_invalidation", negative.reason_code),
            )
        cached = self._cache.get(query.key_cid)
        if cached is None:
            nearest = self._nearest_mismatch(query)
            if nearest is not None:
                cached_key, mismatches = nearest
                if len(mismatches) == 1:
                    return self._reject(
                        query,
                        REASON_BINDING_MISMATCH,
                        limitations=mismatches,
                    )
                if "environment_cid" in mismatches:
                    return self._reject(
                        query,
                        REASON_ENVIRONMENT,
                        limitations=("environment_incompatible",),
                    )
            return self._reject(
                query,
                REASON_INCOMPLETE,
                limitations=("exact_reuse_required",),
            )
        if cached.generation > generation:
            raise ProgramWorldReuseError("cached evidence generation is from the future")
        if cached.generation < generation:
            self.invalidate(
                NegativeMemoryRecord(
                    key_cid=query.key_cid,
                    reason_code=REASON_STALE,
                    generation=generation,
                )
            )
            return self._reject(query, REASON_STALE, limitations=("freshness_revoked",))
        if not cached.abstraction_safe:
            return self._reject(
                query,
                REASON_UNSAFE_ABSTRACTION,
                limitations=("unsafe_abstraction",),
            )
        if not cached.environment_compatible:
            return self._reject(
                query,
                REASON_ENVIRONMENT,
                limitations=("environment_incompatible",),
            )
        if not cached.typed_available:
            return self._reject(
                query,
                REASON_TYPED_UNAVAILABLE,
                fallback=True,
                limitations=("typed_surface_unavailable",),
            )
        if cached.raw_source and cached.proof_cid is None:
            return self._reject(
                query,
                REASON_RAW_FALLBACK,
                fallback=True,
                limitations=("raw_source_is_not_reuse",),
            )
        exact = cached.key.mismatches(query) == ()
        if not exact:
            return self._reject(
                query,
                REASON_BINDING_MISMATCH,
                limitations=cached.key.mismatches(query),
            )
        proof_backed = cached.proof_cid is not None
        reason = REASON_PROOF_CACHE if proof_backed else REASON_EXACT_HIT
        admitted = admission_authority is not None
        return ProgramWorldReuseDecision(
            verdict=ReuseVerdict.REUSE,
            state_cid=query.state_cid,
            goal_cid=query.goal_cid,
            policy_cid=query.policy_cid,
            environment_cid=query.environment_cid,
            toolchain_cid=query.toolchain_cid,
            procedure_revision_cid=query.procedure_revision_cid,
            exact_match=True,
            reason_code=reason,
            proposal_only=not admitted,
            admitted=admitted,
            admission_authority=admission_authority,
            admission_evidence_cid=admission_evidence_cid,
            limitations=("reuse_is_proposal_until_supervisor_admission",),
        )

    def explain_program_world_reuse(
        self,
        query: ProgramWorldReuseKey,
        **kwargs: Any,
    ) -> dict[str, Any]:
        decision = self.evaluate_program_world_reuse(query, **kwargs)
        nearest = self._nearest_mismatch(query)
        return {
            "verdict": decision.verdict,
            "reason_code": decision.reason_code,
            "exact_match": decision.exact_match,
            "admitted": decision.admitted,
            "proposal_only": decision.proposal_only,
            "key_cid": query.key_cid,
            "mismatches": list(nearest[1]) if nearest is not None else [],
            "negative_memory": query.key_cid in self._negative,
            "limitations": list(decision.limitations),
            "ann_authoritative": decision.ann_authoritative,
        }

    def _nearest_mismatch(
        self, query: ProgramWorldReuseKey
    ) -> tuple[ProgramWorldReuseKey, tuple[str, ...]] | None:
        best: tuple[ProgramWorldReuseKey, tuple[str, ...]] | None = None
        for evidence in self._cache.values():
            mismatches = evidence.key.mismatches(query)
            if not mismatches:
                continue
            if best is None or len(mismatches) < len(best[1]):
                best = (evidence.key, mismatches)
        return best

    def _reject(
        self,
        query: ProgramWorldReuseKey,
        reason_code: str,
        *,
        fallback: bool = False,
        limitations: Sequence[str] = (),
    ) -> ProgramWorldReuseDecision:
        if reason_code in ANN_REASON_CODES or reason_code == REASON_SIMILARITY:
            verdict = ReuseVerdict.REJECT
        elif reason_code in {REASON_TYPED_UNAVAILABLE, REASON_RAW_FALLBACK}:
            verdict = ReuseVerdict.UNAVAILABLE
            fallback = True
        else:
            verdict = ReuseVerdict.REJECT
        return ProgramWorldReuseDecision(
            verdict=verdict,
            state_cid=query.state_cid,
            goal_cid=query.goal_cid,
            policy_cid=query.policy_cid,
            environment_cid=query.environment_cid,
            toolchain_cid=query.toolchain_cid,
            procedure_revision_cid=query.procedure_revision_cid,
            exact_match=False,
            reason_code=reason_code,
            fallback=fallback,
            limitations=tuple(limitations),
        )


def evaluate_program_world_reuse(
    query: ProgramWorldReuseKey,
    *,
    gate: ProgramWorldReuseGate | None = None,
    **kwargs: Any,
) -> ProgramWorldReuseDecision:
    evaluator = gate if gate is not None else ProgramWorldReuseGate()
    return evaluator.evaluate_program_world_reuse(query, **kwargs)


def explain_program_world_reuse(
    query: ProgramWorldReuseKey,
    *,
    gate: ProgramWorldReuseGate | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    evaluator = gate if gate is not None else ProgramWorldReuseGate()
    return evaluator.explain_program_world_reuse(query, **kwargs)
