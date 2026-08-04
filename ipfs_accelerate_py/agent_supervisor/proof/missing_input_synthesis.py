"""Prove, refute, and reconstruct missing-value and behavior mappings.

RPR-036 / ``MissingInputSynthesizer@1``

This module is an adapter over capability-admitted ``ipfs_datasets_py``
LogicIR / TDFOL / CEC / SMT / Hammer routes.  Compiled change-propagation
obligations (RPR-035) and nominated value candidates are candidates only:
solver order, premise ranking, vector hits, and unreconstructed cache rows
never grant code authority.

Authority requires independent kernel reconstruction under the exact
premises, translator, toolchain, kernel, and policy roots.  Cached receipts
are revalidated against every invalidator before reuse.  Zero, one, or many
independently proved candidates yield refutation, unique proof, or
ambiguity respectively; search order never breaks ties.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    MissingInputRequirement,
    PropagationAuthorityRoots,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
)
from .change_propagation_obligations import (
    BehaviorRefinementClaim,
    ChangePropagationObligation,
    ChangePropagationObligationCompilation,
    ObligationKind,
    ValueMappingClaim,
)
from .formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_cache import (
    CacheLookupStatus,
    FormalVerificationCache,
    ProofCacheKey,
)
from .formal_verification_capabilities import ProofProviderOperation
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    canonical_json,
    content_identity,
)
from .formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from .kernel_verification import (
    KernelVerificationResult,
    build_kernel_verified_receipt,
)


MISSING_INPUT_SYNTHESIS_INTERFACE: Final = "MissingInputSynthesizer@1"
MISSING_INPUT_SYNTHESIS_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-synthesis-receipt@1"
)
VALUE_MAPPING_PROOF_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/value-mapping-proof@1"
)
BEHAVIOR_PROOF_SET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/behavior-proof-set@1"
)
CANDIDATE_FACET_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/candidate-facet-result@1"
)
UPSTREAM_THREAD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/upstream-thread-requirement@1"
)
IPFS_DATASETS_LOGIC_PROVIDER_ID: Final = "hammer"
LOGIC_CAPABILITY_IDS: Final[frozenset[str]] = frozenset(
    {
        "hammer",
        "smt",
        "tdfol",
        "dcec",
        "cec",
        "logic-ir",
        "datasets.logic_ir",
    }
)

_VALUE_MAPPING_FACETS: Final[frozenset[ObligationKind]] = frozenset(
    {
        ObligationKind.SOURCE_SCOPE_PATH_AVAILABILITY,
        ObligationKind.TYPE_SCHEMA_RANGE_NULLABILITY,
        ObligationKind.INFORMATION_SUFFICIENCY,
        ObligationKind.CONVERSION_CONSTRUCTOR_TOTALITY,
        ObligationKind.ERROR_COMPATIBILITY,
        ObligationKind.EFFECT_COMPATIBILITY,
        ObligationKind.CAPABILITY_COMPATIBILITY,
        ObligationKind.AUTHORIZATION_COMPATIBILITY,
        ObligationKind.TRUST_COMPATIBILITY,
        ObligationKind.RESOURCE_COMPATIBILITY,
        ObligationKind.OWNERSHIP_LIFETIME,
        ObligationKind.MUTATION_CONCURRENCY,
        ObligationKind.DEPENDENCY_CYCLE_ABSENCE,
        ObligationKind.PARAMETER_THREADING,
    }
)

_BEHAVIOR_FACETS: Final[frozenset[ObligationKind]] = frozenset(
    {
        ObligationKind.BEHAVIOR_INVARIANTS,
        ObligationKind.STATE_TRANSITIONS,
        ObligationKind.SERIALIZATION_MIGRATION,
        ObligationKind.PLACEMENT,
    }
)

_NOMINATION_ONLY_KINDS: Final[frozenset[ValueCandidateKind]] = frozenset(
    {
        ValueCandidateKind.VECTOR_NOMINATION,
        ValueCandidateKind.GRAPH_NOMINATION,
        ValueCandidateKind.HISTORY,
    }
)

_NON_CONCLUSIVE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "unknown",
        "timeout",
        "proof_timed_out",
        "missing_backend",
        "incomplete_premise_slice",
        "incomplete_slice",
        "unsupported",
        "unsupported_semantics",
        "stale",
        "stale_cache_entry",
        "failed_reconstruction",
        "malformed_or_wrong_theorem_reconstruction",
        "independent_reconstruction_unavailable",
        "backend_unavailable",
        "backend_non_conclusive",
        "unknown_backend_result",
        "wrong_candidate_or_theorem",
        "unverified_counterexample",
        "cache_invalidator_mismatch",
        "root_mismatch",
    }
)


class MissingInputSynthesisError(ValueError):
    """Caller supplied malformed input or attempted to weaken a proof boundary."""


class SynthesisDisposition(str, Enum):
    """Analytical disposition for one missing-value or behavior clause.

    Non-success dispositions never produce code authority.  Search order of
    candidates or solvers is never used to invent uniqueness.
    """

    UNIQUE_PROVED = "unique_proved"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class FacetDisposition(str, Enum):
    """Per-obligation / per-candidate facet outcome before aggregation."""

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise MissingInputSynthesisError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise MissingInputSynthesisError(f"{name} is required")
    return result


def _ids(values: Sequence[Any], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MissingInputSynthesisError(f"{name} must be a sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise MissingInputSynthesisError(f"{name} must contain non-empty identifiers")
        result.add(value.strip())
    ordered = tuple(sorted(result))
    if required and not ordered:
        raise MissingInputSynthesisError(f"{name} must not be empty")
    return ordered


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MissingInputSynthesisError(f"{name} must be an object")
    try:
        import json

        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MissingInputSynthesisError(f"{name} must contain canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise MissingInputSynthesisError(f"{name} must be an object")
    return normalized


def _failure_reason(code: ProviderFailureCode | None) -> str:
    return {
        ProviderFailureCode.TIMED_OUT: "proof_timed_out",
        ProviderFailureCode.UNSUPPORTED: "unsupported_semantics",
        ProviderFailureCode.UNAVAILABLE: "backend_unavailable",
        ProviderFailureCode.MALFORMED_RESPONSE: "malformed_backend_response",
        ProviderFailureCode.MALFORMED_REQUEST: "malformed_backend_request",
    }.get(code, "backend_non_conclusive")


def _reason_to_facet_disposition(reason: str) -> FacetDisposition:
    key = reason.strip().casefold()
    if key in {"proof_timed_out", "timeout"}:
        return FacetDisposition.TIMEOUT
    if key in {
        "missing_backend",
        "unsupported_semantics",
        "unsupported",
        "backend_unavailable",
        "independent_reconstruction_unavailable",
    }:
        return FacetDisposition.UNSUPPORTED
    return FacetDisposition.UNKNOWN


def _aggregate_facet_dispositions(
    dispositions: Sequence[FacetDisposition],
) -> FacetDisposition:
    """Worst non-success wins; only all-proved yields PROVED."""

    if not dispositions:
        return FacetDisposition.UNKNOWN
    if all(item is FacetDisposition.PROVED for item in dispositions):
        return FacetDisposition.PROVED
    if any(item is FacetDisposition.TIMEOUT for item in dispositions):
        return FacetDisposition.TIMEOUT
    if any(item is FacetDisposition.UNSUPPORTED for item in dispositions):
        return FacetDisposition.UNSUPPORTED
    if any(item is FacetDisposition.UNKNOWN for item in dispositions):
        return FacetDisposition.UNKNOWN
    if any(item is FacetDisposition.REFUTED for item in dispositions):
        return FacetDisposition.REFUTED
    return FacetDisposition.UNKNOWN


def _facet_to_synthesis(disposition: FacetDisposition) -> SynthesisDisposition:
    return {
        FacetDisposition.PROVED: SynthesisDisposition.UNIQUE_PROVED,
        FacetDisposition.REFUTED: SynthesisDisposition.REFUTED,
        FacetDisposition.TIMEOUT: SynthesisDisposition.TIMEOUT,
        FacetDisposition.UNSUPPORTED: SynthesisDisposition.UNSUPPORTED,
        FacetDisposition.UNKNOWN: SynthesisDisposition.UNKNOWN,
    }.get(disposition, SynthesisDisposition.UNKNOWN)


@dataclass(frozen=True)
class UpstreamThreadRequirement(CanonicalContract):
    """A new upstream missing-input demand threaded from a proved origin only."""

    SCHEMA: ClassVar[str] = UPSTREAM_THREAD_SCHEMA

    origin_requirement_id: str
    origin_consumer_id: str
    origin_obligation_id: str
    parameter_name: str
    type_ref: str
    reason_codes: tuple[str, ...]
    threaded_requirement_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "origin_requirement_id",
            "origin_consumer_id",
            "origin_obligation_id",
            "parameter_name",
            "type_ref",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True))
        threaded = _text(self.threaded_requirement_id, "threaded_requirement_id", required=False)
        if not threaded:
            threaded = content_identity(
                {
                    "origin": self.origin_requirement_id,
                    "consumer": self.origin_consumer_id,
                    "parameter": self.parameter_name,
                    "type_ref": self.type_ref,
                }
            )[:48]
            threaded = f"upstream:{threaded}"
        object.__setattr__(self, "threaded_requirement_id", threaded)

    def _payload(self) -> dict[str, Any]:
        return {
            "origin_requirement_id": self.origin_requirement_id,
            "origin_consumer_id": self.origin_consumer_id,
            "origin_obligation_id": self.origin_obligation_id,
            "parameter_name": self.parameter_name,
            "type_ref": self.type_ref,
            "reason_codes": list(self.reason_codes),
            "threaded_requirement_id": self.threaded_requirement_id,
        }


@dataclass(frozen=True)
class CandidateFacetResult:
    """One obligation outcome for one value or behavior candidate."""

    obligation_id: str
    obligation_kind: ObligationKind
    candidate_id: str
    receipt: ProofReceipt
    disposition: FacetDisposition
    reason_codes: tuple[str, ...]
    cache_key_id: str
    counterexample: FormalCounterexample | None = None
    unsatisfied_clauses: tuple[str, ...] = ()
    from_cache: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "obligation_kind", ObligationKind(self.obligation_kind))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not isinstance(self.receipt, ProofReceipt):
            raise MissingInputSynthesisError("facet result requires a typed ProofReceipt")
        object.__setattr__(self, "disposition", FacetDisposition(self.disposition))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True))
        object.__setattr__(self, "cache_key_id", _text(self.cache_key_id, "cache_key_id"))
        object.__setattr__(
            self,
            "unsatisfied_clauses",
            _ids(self.unsatisfied_clauses, "unsatisfied_clauses"),
        )
        if not isinstance(self.from_cache, bool):
            raise MissingInputSynthesisError("from_cache must be boolean")
        if self.counterexample is not None and not isinstance(
            self.counterexample, FormalCounterexample
        ):
            raise MissingInputSynthesisError("counterexample must be a FormalCounterexample")
        if self.disposition is FacetDisposition.PROVED:
            if not self.receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED):
                raise MissingInputSynthesisError(
                    "proved facet requires current independent reconstruction"
                )
            if self.counterexample is not None:
                raise MissingInputSynthesisError("proved facet cannot carry a counterexample")
        elif self.disposition is FacetDisposition.REFUTED:
            if self.receipt.authoritative_verdict is not ProofVerdict.DISPROVED:
                raise MissingInputSynthesisError(
                    "refuted facet requires independently verified counterexample"
                )
            if self.counterexample is None:
                raise MissingInputSynthesisError("refuted facet requires a minimal counterexample")

    @property
    def authoritative(self) -> bool:
        return (
            self.disposition is FacetDisposition.PROVED
            and self.receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED)
        )

    @property
    def code_authority(self) -> bool:
        """Single facets never authorize code; only aggregated unique proofs may."""

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_FACET_RESULT_SCHEMA,
            "obligation_id": self.obligation_id,
            "obligation_kind": self.obligation_kind.value,
            "candidate_id": self.candidate_id,
            "receipt": self.receipt.to_dict(),
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "cache_key_id": self.cache_key_id,
            "counterexample_id": (
                self.counterexample.counterexample_id if self.counterexample else ""
            ),
            "unsatisfied_clauses": list(self.unsatisfied_clauses),
            "from_cache": self.from_cache,
            "authoritative": self.authoritative,
            "code_authority": self.code_authority,
        }


@dataclass(frozen=True)
class ValueMappingProof(CanonicalContract):
    """Aggregated prove/refute result for one missing-input requirement."""

    SCHEMA: ClassVar[str] = VALUE_MAPPING_PROOF_SCHEMA

    requirement_id: str
    consumer_id: str
    disposition: SynthesisDisposition
    facet_results: tuple[CandidateFacetResult, ...]
    proved_candidate_ids: tuple[str, ...] = ()
    refuted_candidate_ids: tuple[str, ...] = ()
    inconclusive_candidate_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    unsatisfied_clauses: tuple[str, ...] = ()
    expression_ref: str = ""
    type_ref: str = ""
    mapping_claim_id: str = ""
    upstream_thread: UpstreamThreadRequirement | None = None
    repository_id: str = ""
    tree_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _text(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "disposition", SynthesisDisposition(self.disposition))
        if not all(isinstance(item, CandidateFacetResult) for item in self.facet_results):
            raise MissingInputSynthesisError("facet_results must be CandidateFacetResult values")
        # Content-ordered candidate sets — never preserve nomination/search order.
        object.__setattr__(
            self,
            "proved_candidate_ids",
            _ids(self.proved_candidate_ids, "proved_candidate_ids"),
        )
        object.__setattr__(
            self,
            "refuted_candidate_ids",
            _ids(self.refuted_candidate_ids, "refuted_candidate_ids"),
        )
        object.__setattr__(
            self,
            "inconclusive_candidate_ids",
            _ids(self.inconclusive_candidate_ids, "inconclusive_candidate_ids"),
        )
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(
            self,
            "unsatisfied_clauses",
            _ids(self.unsatisfied_clauses, "unsatisfied_clauses"),
        )
        for name in (
            "expression_ref",
            "type_ref",
            "mapping_claim_id",
            "repository_id",
            "tree_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.upstream_thread is not None and not isinstance(
            self.upstream_thread, UpstreamThreadRequirement
        ):
            raise MissingInputSynthesisError(
                "upstream_thread must be UpstreamThreadRequirement"
            )
        # Fail closed: uniqueness is never implied by a single listed id when
        # disposition says otherwise, and ambiguous sets may never claim code.
        if self.disposition is SynthesisDisposition.UNIQUE_PROVED:
            if len(self.proved_candidate_ids) != 1:
                raise MissingInputSynthesisError(
                    "unique_proved requires exactly one independently proved candidate"
                )
            if self.inconclusive_candidate_ids:
                raise MissingInputSynthesisError(
                    "unique_proved cannot retain inconclusive candidates"
                )
        if self.disposition is SynthesisDisposition.AMBIGUOUS:
            if len(self.proved_candidate_ids) < 2:
                raise MissingInputSynthesisError(
                    "ambiguous requires two or more independently proved candidates"
                )
        if self.disposition is SynthesisDisposition.REFUTED:
            if self.proved_candidate_ids:
                raise MissingInputSynthesisError("refuted cannot list proved candidates")
        if self.upstream_thread is not None:
            if self.upstream_thread.origin_requirement_id != self.requirement_id:
                raise MissingInputSynthesisError(
                    "upstream thread must preserve the exact origin requirement"
                )
            if self.disposition is SynthesisDisposition.UNIQUE_PROVED and not self.proved_candidate_ids:
                raise MissingInputSynthesisError(
                    "proved value may only open an upstream thread with origin binding"
                )

    @property
    def proof_id(self) -> str:
        return self.content_id

    @property
    def unique_candidate_id(self) -> str:
        if self.disposition is not SynthesisDisposition.UNIQUE_PROVED:
            return ""
        return self.proved_candidate_ids[0] if self.proved_candidate_ids else ""

    @property
    def code_authority(self) -> bool:
        """True only for a unique reconstructed mapping under current roots."""

        if self.disposition is not SynthesisDisposition.UNIQUE_PROVED:
            return False
        if len(self.proved_candidate_ids) != 1:
            return False
        proved_id = self.proved_candidate_ids[0]
        relevant = [
            item
            for item in self.facet_results
            if item.candidate_id == proved_id
            and item.obligation_kind in _VALUE_MAPPING_FACETS
        ]
        if not relevant:
            return False
        return all(item.authoritative for item in relevant)

    def _payload(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "consumer_id": self.consumer_id,
            "disposition": self.disposition.value,
            "facet_results": [item.to_dict() for item in self.facet_results],
            "proved_candidate_ids": list(self.proved_candidate_ids),
            "refuted_candidate_ids": list(self.refuted_candidate_ids),
            "inconclusive_candidate_ids": list(self.inconclusive_candidate_ids),
            "reason_codes": list(self.reason_codes),
            "unsatisfied_clauses": list(self.unsatisfied_clauses),
            "expression_ref": self.expression_ref,
            "type_ref": self.type_ref,
            "mapping_claim_id": self.mapping_claim_id,
            "upstream_thread": (
                self.upstream_thread.to_dict() if self.upstream_thread is not None else None
            ),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "code_authority": self.code_authority,
        }


@dataclass(frozen=True)
class BehaviorProofSet(CanonicalContract):
    """Aggregated prove/refute result for one required-behavior contract."""

    SCHEMA: ClassVar[str] = BEHAVIOR_PROOF_SET_SCHEMA

    behavior_id: str
    consumer_id: str
    disposition: SynthesisDisposition
    facet_results: tuple[CandidateFacetResult, ...]
    reason_codes: tuple[str, ...] = ()
    unsatisfied_clauses: tuple[str, ...] = ()
    refinement_claim_id: str = ""
    placement_decision_ref: str = ""
    repository_id: str = ""
    tree_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior_id", _text(self.behavior_id, "behavior_id"))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "disposition", SynthesisDisposition(self.disposition))
        if not all(isinstance(item, CandidateFacetResult) for item in self.facet_results):
            raise MissingInputSynthesisError("facet_results must be CandidateFacetResult values")
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(
            self,
            "unsatisfied_clauses",
            _ids(self.unsatisfied_clauses, "unsatisfied_clauses"),
        )
        for name in (
            "refinement_claim_id",
            "placement_decision_ref",
            "repository_id",
            "tree_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.disposition is SynthesisDisposition.AMBIGUOUS:
            raise MissingInputSynthesisError(
                "behavior proof sets do not use multi-candidate ambiguity"
            )

    @property
    def proof_id(self) -> str:
        return self.content_id

    @property
    def code_authority(self) -> bool:
        if self.disposition is not SynthesisDisposition.UNIQUE_PROVED:
            return False
        relevant = [
            item
            for item in self.facet_results
            if item.obligation_kind in _BEHAVIOR_FACETS
        ]
        if not relevant:
            return False
        return all(item.authoritative for item in relevant)

    def _payload(self) -> dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "consumer_id": self.consumer_id,
            "disposition": self.disposition.value,
            "facet_results": [item.to_dict() for item in self.facet_results],
            "reason_codes": list(self.reason_codes),
            "unsatisfied_clauses": list(self.unsatisfied_clauses),
            "refinement_claim_id": self.refinement_claim_id,
            "placement_decision_ref": self.placement_decision_ref,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "code_authority": self.code_authority,
        }


@dataclass(frozen=True)
class MissingInputSynthesisReceipt(CanonicalContract):
    """Complete analytical synthesis for one consumer obligation compilation."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_SYNTHESIS_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    delta_id: str
    consumer_id: str
    migration_obligation_id: str
    value_mapping_proofs: tuple[ValueMappingProof, ...]
    behavior_proof_sets: tuple[BehaviorProofSet, ...]
    backend_id: str
    backend_version: str
    reason_codes: tuple[str, ...] = ()
    invalidators: tuple[str, ...] = ()
    synthesizer_id: str = MISSING_INPUT_SYNTHESIS_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise MissingInputSynthesisError(
                "receipt requires PropagationAuthorityRoots"
            )
        for name in ("delta_id", "consumer_id", "migration_obligation_id", "backend_id", "backend_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "synthesizer_id", _text(self.synthesizer_id, "synthesizer_id")
        )
        if not all(isinstance(item, ValueMappingProof) for item in self.value_mapping_proofs):
            raise MissingInputSynthesisError(
                "value_mapping_proofs must be ValueMappingProof values"
            )
        if not all(isinstance(item, BehaviorProofSet) for item in self.behavior_proof_sets):
            raise MissingInputSynthesisError(
                "behavior_proof_sets must be BehaviorProofSet values"
            )
        object.__setattr__(
            self,
            "value_mapping_proofs",
            tuple(sorted(self.value_mapping_proofs, key=lambda item: item.requirement_id)),
        )
        object.__setattr__(
            self,
            "behavior_proof_sets",
            tuple(sorted(self.behavior_proof_sets, key=lambda item: item.behavior_id)),
        )
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(self, "invalidators", _ids(self.invalidators, "invalidators"))

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def code_authority(self) -> bool:
        """True only when every automated clause has unique reconstructed proof."""

        if not self.value_mapping_proofs and not self.behavior_proof_sets:
            return False
        values_ok = all(item.code_authority for item in self.value_mapping_proofs)
        behaviors_ok = all(item.code_authority for item in self.behavior_proof_sets)
        if self.value_mapping_proofs and not values_ok:
            return False
        if self.behavior_proof_sets and not behaviors_ok:
            return False
        return bool(self.value_mapping_proofs or self.behavior_proof_sets) and (
            (not self.value_mapping_proofs or values_ok)
            and (not self.behavior_proof_sets or behaviors_ok)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": MISSING_INPUT_SYNTHESIS_INTERFACE,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "consumer_id": self.consumer_id,
            "migration_obligation_id": self.migration_obligation_id,
            "value_mapping_proofs": [item.to_dict() for item in self.value_mapping_proofs],
            "behavior_proof_sets": [item.to_dict() for item in self.behavior_proof_sets],
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "reason_codes": list(self.reason_codes),
            "invalidators": list(self.invalidators),
            "synthesizer_id": self.synthesizer_id,
            "code_authority": self.code_authority,
        }


CounterexampleVerifier = Callable[
    [ChangePropagationObligation, Mapping[str, Any]],
    FormalCounterexample | Mapping[str, Any] | None,
]


class MissingInputSynthesizer:
    """Route finite change-propagation obligations through admitted logic backends."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        cache: FormalVerificationCache | None = None,
        resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
        counterexample_verifier: CounterexampleVerifier | None = None,
        multi_prover_router: Any | None = None,
    ) -> None:
        if cache is not None and not isinstance(cache, FormalVerificationCache):
            raise MissingInputSynthesisError("cache must be a FormalVerificationCache")
        if counterexample_verifier is not None and not callable(counterexample_verifier):
            raise MissingInputSynthesisError("counterexample_verifier must be callable")
        self.backend = backend if backend is not None else self._default_backend()
        self.cache = cache
        self.resource_budget = self._budget(resource_budget)
        self.counterexample_verifier = counterexample_verifier
        # Optional MultiProverRouter is retained for portfolio planning only; it
        # never grants authority without independent reconstruction.
        self.multi_prover_router = multi_prover_router

    @staticmethod
    def _default_backend() -> Any:
        from ..integrations.ipfs_datasets_logic_provider import IpfsDatasetsLogicProvider

        return IpfsDatasetsLogicProvider()

    @staticmethod
    def _budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
        if value is None:
            return ResourceBudget()
        if isinstance(value, ResourceBudget):
            return value
        if isinstance(value, Mapping):
            return ResourceBudget.from_dict(value)
        raise MissingInputSynthesisError(
            "resource_budget must be a ResourceBudget or object"
        )

    def _backend_identity(self) -> tuple[str, str]:
        provider_id = str(getattr(self.backend, "provider_id", "")).strip()
        provider_version = str(getattr(self.backend, "provider_version", "")).strip()
        if not provider_id or not provider_version:
            return "unavailable", "unavailable"
        return provider_id, provider_version

    def _backend_supports(self, operation: ProofProviderOperation) -> bool:
        provider_id, _ = self._backend_identity()
        if provider_id not in LOGIC_CAPABILITY_IDS and provider_id != IPFS_DATASETS_LOGIC_PROVIDER_ID:
            return False
        capability_method = getattr(self.backend, "capabilities", None)
        if not callable(capability_method):
            return False
        try:
            capability = capability_method()
            operations = getattr(capability, "operations", ())
            return operation in operations
        except (TypeError, ValueError, AttributeError):
            return False

    def _cache_key(
        self,
        obligation: ChangePropagationObligation,
        premises: tuple[dict[str, Any], ...],
        *,
        candidate_id: str,
        invalidators: Sequence[str],
    ) -> ProofCacheKey:
        claim = obligation.claim
        backend_id, backend_version = self._backend_identity()
        return ProofCacheKey(
            obligation={
                "change_propagation_obligation": obligation.to_dict(),
                "logic_ir": claim.to_logic_ir(),
                "candidate_id": candidate_id,
            },
            premises=premises,
            translator={
                "id": claim.translator_id,
                "capability": claim.capability_id,
                "revision": claim.capability_revision,
            },
            solver={"provider_id": backend_id, "provider_version": backend_version},
            kernel={
                "required": "independent-reconstruction",
                "capability": claim.capability_id,
            },
            toolchain={"id": claim.toolchain_id, "backend_version": backend_version},
            theorem_registry={
                "source_ids": list(claim.source_ids),
                "assumption_ids": list(claim.assumption_ids),
            },
            policy={
                "id": claim.policy_id,
                "required_assurance": AssuranceLevel.KERNEL_VERIFIED.value,
                "invalidators": list(sorted(set(invalidators))),
            },
            resource_budget=self.resource_budget.to_dict(),
            candidate_tree={
                "repository_id": claim.repository_id,
                "tree_id": claim.tree_id,
                "candidate_id": candidate_id,
                "consumer_id": obligation.consumer_id,
            },
        )

    @staticmethod
    def _premises_for(
        obligation: ChangePropagationObligation,
        premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(premises, Mapping):
            raw = []
            for premise_id in obligation.claim.premise_ids:
                value = premises.get(premise_id)
                if value is None:
                    raise MissingInputSynthesisError("incomplete_premise_slice")
                if not isinstance(value, Mapping):
                    raise MissingInputSynthesisError("premise records must be objects")
                item = dict(value)
                item.setdefault("premise_id", premise_id)
                raw.append(item)
        elif isinstance(premises, Sequence) and not isinstance(
            premises, (str, bytes, bytearray)
        ):
            by_id = {
                str(item.get("premise_id", "")): item
                for item in premises
                if isinstance(item, Mapping)
            }
            raw = [
                dict(by_id[premise_id])
                for premise_id in obligation.claim.premise_ids
                if premise_id in by_id
            ]
            if len(raw) != len(obligation.claim.premise_ids):
                raise MissingInputSynthesisError("incomplete_premise_slice")
        else:
            raise MissingInputSynthesisError("premises must be a mapping or sequence")
        normalized = tuple(_canonical_mapping(item, "premise") for item in raw)
        if {str(item.get("premise_id", "")) for item in normalized} != set(
            obligation.claim.premise_ids
        ):
            raise MissingInputSynthesisError("premises do not bind the exact claim")
        return tuple(sorted(normalized, key=lambda item: str(item["premise_id"])))

    def _receipt_matches_invalidators(
        self,
        receipt: ProofReceipt,
        obligation: ChangePropagationObligation,
        invalidators: Sequence[str],
    ) -> bool:
        claim = obligation.claim
        if receipt.obligation_id != obligation.code_obligation.obligation_id:
            return False
        if receipt.repository_tree_id != claim.tree_id:
            return False
        if receipt.toolchain_id != claim.toolchain_id:
            return False
        if receipt.policy_id != claim.policy_id:
            return False
        if receipt.translator_id != claim.translator_id:
            return False
        if receipt.freshness is not EvidenceFreshness.CURRENT:
            return False
        if not receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED):
            return False
        # Every listed invalidator must still be present on the receipt premise
        # set or metadata so a root/graph/index rotation cannot reuse stale work.
        receipt_tokens = set(receipt.premise_ids)
        receipt_tokens.update(str(item) for item in receipt.metadata.get("invalidators", []) if item)
        receipt_tokens.add(receipt.repository_tree_id)
        receipt_tokens.add(receipt.toolchain_id)
        receipt_tokens.add(receipt.policy_id)
        receipt_tokens.add(receipt.translator_id)
        for item in invalidators:
            if item and item not in receipt_tokens and item not in receipt.premise_ids:
                # Invalidators that are pure root ids are checked above; free-form
                # invalidators must appear in the receipt binding surface.
                meta_invalidators = {
                    str(value)
                    for value in (receipt.metadata.get("invalidators") or [])
                    if value
                }
                if item not in meta_invalidators:
                    return False
        return True

    def _non_conclusive_receipt(
        self,
        obligation: ChangePropagationObligation,
        *,
        verdict: ProofVerdict,
        reason: str,
        backend_id: str,
        candidate_id: str,
        invalidators: Sequence[str],
    ) -> ProofReceipt:
        code = obligation.code_obligation
        return ProofReceipt(
            obligation_id=code.obligation_id,
            plan_id=content_identity(
                {
                    "interface": MISSING_INPUT_SYNTHESIS_INTERFACE,
                    "obligation": obligation.obligation_id,
                    "candidate_id": candidate_id,
                }
            ),
            attempt_id=content_identity(
                {
                    "reason": reason,
                    "obligation": obligation.obligation_id,
                    "candidate_id": candidate_id,
                }
            ),
            repository_id=code.repository_id,
            repository_tree_id=code.repository_tree_id,
            ast_scope_ids=code.ast_scope_ids,
            premise_ids=code.premise_ids,
            translator_id=obligation.claim.translator_id,
            solver_id=backend_id or "unavailable",
            kernel_id="independent-reconstruction-required",
            toolchain_id=obligation.claim.toolchain_id,
            policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget,
            verdict=verdict,
            freshness=EvidenceFreshness.CURRENT,
            metadata={
                "reason_codes": [reason],
                "change_propagation_obligation_id": obligation.obligation_id,
                "candidate_id": candidate_id,
                "invalidators": list(sorted(set(invalidators))),
            },
        )

    def _result(
        self,
        obligation: ChangePropagationObligation,
        *,
        candidate_id: str,
        receipt: ProofReceipt,
        disposition: FacetDisposition,
        reasons: Sequence[str],
        key: ProofCacheKey,
        counterexample: FormalCounterexample | None = None,
        unsatisfied_clauses: Sequence[str] = (),
        from_cache: bool = False,
    ) -> CandidateFacetResult:
        return CandidateFacetResult(
            obligation_id=obligation.obligation_id,
            obligation_kind=obligation.kind,
            candidate_id=candidate_id,
            receipt=receipt,
            disposition=disposition,
            reason_codes=tuple(reasons),
            cache_key_id=key.key_id,
            counterexample=counterexample,
            unsatisfied_clauses=tuple(unsatisfied_clauses),
            from_cache=from_cache,
        )

    def _reconstruction_receipt(
        self,
        obligation: ChangePropagationObligation,
        result: Mapping[str, Any],
        *,
        request_id: str,
        candidate_id: str,
        invalidators: Sequence[str],
    ) -> ProofReceipt | None:
        raw = result.get("kernel_verification")
        if not isinstance(raw, Mapping):
            return None
        try:
            verification = KernelVerificationResult.from_dict(raw)
        except (TypeError, ValueError):
            return None
        code = obligation.code_obligation
        if (
            verification.obligation_id != code.obligation_id
            or verification.request_id != request_id
            or verification.candidate_id != candidate_id
            or verification.toolchain_id != obligation.claim.toolchain_id
            or verification.verdict is not ProofVerdict.PROVED
        ):
            return None
        receipt = build_kernel_verified_receipt(
            verification,
            obligation=code,
            plan_id=content_identity(
                {
                    "interface": MISSING_INPUT_SYNTHESIS_INTERFACE,
                    "obligation": obligation.obligation_id,
                    "candidate_id": candidate_id,
                }
            ),
            attempt_id=verification.request_id,
            translator_id=obligation.claim.translator_id,
            solver_id=self._backend_identity()[0],
            policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget,
            provider_id=self._backend_identity()[0],
            theorem_registry_id=content_identity(
                {"sources": list(obligation.claim.source_ids)}
            ),
            metadata={
                "change_propagation_obligation_id": obligation.obligation_id,
                "claim_id": obligation.claim.content_id,
                "candidate_id": candidate_id,
                "invalidators": list(sorted(set(invalidators))),
            },
        )
        if not self._receipt_matches_invalidators(receipt, obligation, invalidators):
            return None
        return receipt

    def _candidate_counterexample(
        self,
        obligation: ChangePropagationObligation,
        raw: Mapping[str, Any],
    ) -> FormalCounterexample | None:
        try:
            return normalize_counterexample(
                raw,
                kind=CounterexampleKind.SMT_MODEL,
                bindings=CounterexampleBindings(
                    tree_ids=(obligation.claim.tree_id,),
                    obligation_ids=(obligation.code_obligation.obligation_id,),
                    provider_ids=(self._backend_identity()[0],),
                    policy_ids=(obligation.claim.policy_id,),
                ),
                violated_property=obligation.claim.predicate,
            )
        except (TypeError, ValueError):
            return None

    def _verified_counterexample(
        self,
        obligation: ChangePropagationObligation,
        raw: Mapping[str, Any],
    ) -> FormalCounterexample | None:
        if self.counterexample_verifier is None:
            return None
        try:
            candidate = self.counterexample_verifier(obligation, raw)
            if candidate is None:
                return None
            if isinstance(candidate, FormalCounterexample):
                result = candidate
            elif isinstance(candidate, Mapping):
                result = normalize_counterexample(
                    candidate,
                    kind=CounterexampleKind.SMT_MODEL,
                    bindings=CounterexampleBindings(
                        tree_ids=(obligation.claim.tree_id,),
                        obligation_ids=(obligation.code_obligation.obligation_id,),
                        provider_ids=(self._backend_identity()[0],),
                        policy_ids=(obligation.claim.policy_id,),
                    ),
                    violated_property=obligation.claim.predicate,
                )
            else:
                return None
        except (TypeError, ValueError):
            return None
        bindings = result.bindings
        if (
            obligation.claim.tree_id not in bindings.tree_ids
            or obligation.code_obligation.obligation_id not in bindings.obligation_ids
        ):
            return None
        return result

    def _refuted_receipt(
        self,
        obligation: ChangePropagationObligation,
        counterexample: FormalCounterexample,
        *,
        candidate_id: str,
        invalidators: Sequence[str],
    ) -> ProofReceipt:
        code = obligation.code_obligation
        evidence = ProofEvidence(
            kind=EvidenceKind.SOLVER_RESULT,
            authority=EvidenceAuthority.VALIDATION_RUNNER,
            verdict=EvidenceVerdict.REJECTED,
            artifact_id=counterexample.counterexample_id,
            subject_id=code.obligation_id,
            verifier_id="policy-approved-counterexample-checker",
            independent=True,
            metadata={
                "counterexample_verified": True,
                "change_propagation_obligation_id": obligation.obligation_id,
                "candidate_id": candidate_id,
            },
        )
        return ProofReceipt(
            obligation_id=code.obligation_id,
            plan_id=content_identity(
                {
                    "counterexample": counterexample.counterexample_id,
                    "candidate_id": candidate_id,
                }
            ),
            attempt_id=counterexample.counterexample_id,
            repository_id=code.repository_id,
            repository_tree_id=code.repository_tree_id,
            ast_scope_ids=code.ast_scope_ids,
            premise_ids=code.premise_ids,
            translator_id=obligation.claim.translator_id,
            solver_id=self._backend_identity()[0],
            kernel_id="policy-approved-counterexample-checker",
            toolchain_id=obligation.claim.toolchain_id,
            policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget,
            verdict=ProofVerdict.DISPROVED,
            evidence=(evidence,),
            freshness=EvidenceFreshness.CURRENT,
            metadata={
                "candidate_id": candidate_id,
                "invalidators": list(sorted(set(invalidators))),
            },
        )

    def prove_obligation(
        self,
        obligation: ChangePropagationObligation,
        *,
        candidate_id: str,
        premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        reconstruction_inputs: Mapping[str, Any] | None = None,
        invalidators: Sequence[str] = (),
    ) -> CandidateFacetResult:
        """Prove or refute one finite obligation for one candidate identity."""

        if not isinstance(obligation, ChangePropagationObligation):
            raise MissingInputSynthesisError(
                "obligation must be a ChangePropagationObligation"
            )
        candidate_id = _text(candidate_id, "candidate_id")
        invalidator_ids = _ids(invalidators, "invalidators")
        backend_id, _ = self._backend_identity()
        try:
            exact_premises = self._premises_for(obligation, premises)
        except MissingInputSynthesisError as exc:
            key = self._cache_key(
                obligation, (), candidate_id=candidate_id, invalidators=invalidator_ids
            )
            reason = str(exc)
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.INCONCLUSIVE,
                reason=reason,
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=_reason_to_facet_disposition(reason),
                reasons=(reason,),
                key=key,
                unsatisfied_clauses=obligation.claim.premise_ids,
            )
        key = self._cache_key(
            obligation,
            exact_premises,
            candidate_id=candidate_id,
            invalidators=invalidator_ids,
        )
        if self.cache is not None:
            hit = self.cache.lookup(
                key,
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                required_freshness=EvidenceFreshness.CURRENT,
            )
            if hit.status is CacheLookupStatus.HIT and hit.receipt is not None:
                receipt = hit.receipt
                if self._receipt_matches_invalidators(
                    receipt, obligation, invalidator_ids
                ):
                    return self._result(
                        obligation,
                        candidate_id=candidate_id,
                        receipt=receipt,
                        disposition=FacetDisposition.PROVED,
                        reasons=("authoritative_cache_hit",),
                        key=key,
                        from_cache=True,
                    )
                # Stale or invalidator-mismatched cache rows stay non-conclusive.
                receipt = self._non_conclusive_receipt(
                    obligation,
                    verdict=ProofVerdict.INCONCLUSIVE,
                    reason="cache_invalidator_mismatch",
                    backend_id=backend_id,
                    candidate_id=candidate_id,
                    invalidators=invalidator_ids,
                )
                return self._result(
                    obligation,
                    candidate_id=candidate_id,
                    receipt=receipt,
                    disposition=FacetDisposition.UNKNOWN,
                    reasons=("cache_invalidator_mismatch", "stale_cache_entry"),
                    key=key,
                    from_cache=True,
                )
            if hit.status is CacheLookupStatus.REJECTED:
                reasons = tuple(
                    str(item)
                    for item in (getattr(hit, "reasons", ()) or getattr(hit, "reason_codes", ()) or ())
                )
                stale = any("stale" in reason for reason in reasons) or not reasons
                reason = "stale_cache_entry" if stale else "cache_invalidator_mismatch"
                receipt = self._non_conclusive_receipt(
                    obligation,
                    verdict=ProofVerdict.INCONCLUSIVE,
                    reason=reason,
                    backend_id=backend_id,
                    candidate_id=candidate_id,
                    invalidators=invalidator_ids,
                )
                return self._result(
                    obligation,
                    candidate_id=candidate_id,
                    receipt=receipt,
                    disposition=FacetDisposition.UNKNOWN,
                    reasons=(reason, *reasons[:4]),
                    key=key,
                    from_cache=True,
                )

        if not self._backend_supports(ProofProviderOperation.PROVE):
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.UNSUPPORTED,
                reason="missing_backend",
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=FacetDisposition.UNSUPPORTED,
                reasons=("missing_backend",),
                key=key,
            )

        # Optional portfolio plan is diagnostic only.
        if self.multi_prover_router is not None:
            plan_method = getattr(self.multi_prover_router, "route", None) or getattr(
                self.multi_prover_router, "plan", None
            )
            if callable(plan_method):
                try:
                    plan_method(obligation.code_obligation)
                except (TypeError, ValueError, AttributeError):
                    pass

        request = ProviderRequest(
            request_id=content_identity(
                {
                    "obligation": obligation.obligation_id,
                    "candidate_id": candidate_id,
                    "cache_key": key.key_id,
                }
            )[-64:],
            operation=ProofProviderOperation.PROVE,
            payload={
                "obligation": obligation.code_obligation.to_dict(),
                "premises": list(exact_premises),
                "logic_ir_claim": obligation.claim.to_logic_ir(),
                "change_propagation_obligation_id": obligation.obligation_id,
                "candidate_id": candidate_id,
                "value_mapping_claim_id": obligation.value_mapping_claim_id,
                "behavior_refinement_claim_id": obligation.behavior_refinement_claim_id,
                "invalidators": list(invalidator_ids),
            },
            resource_budget=self.resource_budget,
        )
        response = dispatch_provider_request(self.backend, request)
        if not response.ok:
            assert response.error is not None
            reason = _failure_reason(response.error.code)
            if response.error.code is ProviderFailureCode.TIMED_OUT:
                disposition = FacetDisposition.TIMEOUT
                verdict = ProofVerdict.INCONCLUSIVE
            elif response.error.code in {
                ProviderFailureCode.UNSUPPORTED,
                ProviderFailureCode.UNAVAILABLE,
            }:
                disposition = FacetDisposition.UNSUPPORTED
                verdict = ProofVerdict.UNSUPPORTED
            else:
                disposition = FacetDisposition.UNKNOWN
                verdict = ProofVerdict.INCONCLUSIVE
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=verdict,
                reason=reason,
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=disposition,
                reasons=(reason,),
                key=key,
            )

        result = response.result or {}
        status = str(result.get("status", "")).strip().lower()
        raw_counterexample = result.get("counterexample")
        unsatisfied = _ids(
            result.get("unsatisfied_clauses") or result.get("unsat_core") or (),
            "unsatisfied_clauses",
        )
        if status in {"counterexample", "refuted", "disproved", "sat"} and isinstance(
            raw_counterexample, Mapping
        ):
            counterexample = self._verified_counterexample(obligation, raw_counterexample)
            if counterexample is not None:
                receipt = self._refuted_receipt(
                    obligation,
                    counterexample,
                    candidate_id=candidate_id,
                    invalidators=invalidator_ids,
                )
                violated = unsatisfied
                if not violated:
                    prop = str(getattr(counterexample, "violated_property", "") or "").strip()
                    if prop:
                        violated = (prop,)
                return self._result(
                    obligation,
                    candidate_id=candidate_id,
                    receipt=receipt,
                    disposition=FacetDisposition.REFUTED,
                    reasons=("independently_verified_counterexample",),
                    key=key,
                    counterexample=counterexample,
                    unsatisfied_clauses=violated,
                )
            counterexample = self._candidate_counterexample(obligation, raw_counterexample)
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.INCONCLUSIVE,
                reason="unverified_counterexample",
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=FacetDisposition.UNKNOWN,
                reasons=("unverified_counterexample",),
                key=key,
                counterexample=counterexample,
                unsatisfied_clauses=unsatisfied,
            )

        candidate = result.get("proof_candidate")
        if not isinstance(candidate, Mapping) and isinstance(
            result.get("hammer_result"), Mapping
        ):
            candidate = result["hammer_result"].get("proof_candidate")
        if not isinstance(candidate, Mapping):
            if status in {"timeout", "timed_out"}:
                reason = "proof_timed_out"
                disposition = FacetDisposition.TIMEOUT
            elif status in {"unsupported"}:
                reason = "unsupported_semantics"
                disposition = FacetDisposition.UNSUPPORTED
            else:
                reason = (
                    "unknown_backend_result"
                    if status in {"", "unknown", "candidate"}
                    else "backend_non_conclusive"
                )
                disposition = FacetDisposition.UNKNOWN
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.INCONCLUSIVE,
                reason=reason,
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=disposition,
                reasons=(reason,),
                key=key,
                unsatisfied_clauses=unsatisfied,
            )

        proof_candidate_id = str(candidate.get("candidate_id", "")).strip()
        candidate_request_id = str(candidate.get("request_id", "")).strip()
        if not proof_candidate_id or candidate_request_id != request.request_id:
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.INCONCLUSIVE,
                reason="wrong_candidate_or_theorem",
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=FacetDisposition.UNKNOWN,
                reasons=("wrong_candidate_or_theorem",),
                key=key,
            )
        # The solver candidate identity must match the value-candidate under test
        # when the backend echoes it; mismatch is non-conclusive.
        echoed = str(candidate.get("value_candidate_id", proof_candidate_id)).strip()
        if echoed and echoed not in {candidate_id, proof_candidate_id}:
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.INCONCLUSIVE,
                reason="wrong_candidate_or_theorem",
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=FacetDisposition.UNKNOWN,
                reasons=("wrong_candidate_or_theorem",),
                key=key,
            )

        if not self._backend_supports(ProofProviderOperation.RECONSTRUCT):
            receipt = self._non_conclusive_receipt(
                obligation,
                verdict=ProofVerdict.UNSUPPORTED,
                reason="independent_reconstruction_unavailable",
                backend_id=backend_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            return self._result(
                obligation,
                candidate_id=candidate_id,
                receipt=receipt,
                disposition=FacetDisposition.UNSUPPORTED,
                reasons=("independent_reconstruction_unavailable",),
                key=key,
            )

        extras = _canonical_mapping(
            reconstruction_inputs or {}, "reconstruction_inputs"
        )
        reconstruction_request = ProviderRequest(
            request_id=request.request_id,
            operation=ProofProviderOperation.RECONSTRUCT,
            payload={
                **dict(request.payload),
                **extras,
                "proof_candidate": dict(candidate),
                "value_candidate_id": candidate_id,
            },
            resource_budget=self.resource_budget,
        )
        reconstruction = dispatch_provider_request(self.backend, reconstruction_request)
        if reconstruction.ok:
            receipt = self._reconstruction_receipt(
                obligation,
                reconstruction.result or {},
                request_id=request.request_id,
                candidate_id=candidate_id,
                invalidators=invalidator_ids,
            )
            if receipt is not None:
                if self.cache is not None:
                    self.cache.put(key, receipt)
                return self._result(
                    obligation,
                    candidate_id=candidate_id,
                    receipt=receipt,
                    disposition=FacetDisposition.PROVED,
                    reasons=("independent_reconstruction_accepted",),
                    key=key,
                )
        reason = (
            "malformed_or_wrong_theorem_reconstruction"
            if reconstruction.ok
            else _failure_reason(
                reconstruction.error.code if reconstruction.error else None
            )
        )
        disposition = (
            FacetDisposition.TIMEOUT
            if reason == "proof_timed_out"
            else FacetDisposition.UNKNOWN
        )
        receipt = self._non_conclusive_receipt(
            obligation,
            verdict=ProofVerdict.INCONCLUSIVE,
            reason=reason,
            backend_id=backend_id,
            candidate_id=candidate_id,
            invalidators=invalidator_ids,
        )
        return self._result(
            obligation,
            candidate_id=candidate_id,
            receipt=receipt,
            disposition=disposition,
            reasons=(reason,),
            key=key,
        )

    def prove_candidate_mapping(
        self,
        compilation: ChangePropagationObligationCompilation,
        candidate: ValueCandidate,
        *,
        premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        reconstruction_inputs: Mapping[str, Any] | None = None,
        invalidators: Sequence[str] = (),
    ) -> tuple[CandidateFacetResult, ...]:
        """Independently prove every value-mapping facet for one candidate."""

        if not isinstance(compilation, ChangePropagationObligationCompilation):
            raise MissingInputSynthesisError(
                "compilation must be ChangePropagationObligationCompilation"
            )
        if not isinstance(candidate, ValueCandidate):
            raise MissingInputSynthesisError("candidate must be a ValueCandidate")
        # Nomination-only sources may be exercised but never become proved authority.
        facets = [
            item
            for item in compilation.obligations
            if item.kind in _VALUE_MAPPING_FACETS
        ]
        results = tuple(
            self.prove_obligation(
                item,
                candidate_id=candidate.candidate_id,
                premises=premises,
                reconstruction_inputs=reconstruction_inputs,
                invalidators=invalidators,
            )
            for item in sorted(facets, key=lambda item: item.kind.value)
        )
        if candidate.kind in _NOMINATION_ONLY_KINDS:
            # Downgrade any accidental PROVED facets: nominations are non-axioms.
            adjusted: list[CandidateFacetResult] = []
            for item in results:
                if item.disposition is FacetDisposition.PROVED:
                    receipt = self._non_conclusive_receipt(
                        next(
                            obl
                            for obl in facets
                            if obl.obligation_id == item.obligation_id
                        ),
                        verdict=ProofVerdict.UNSUPPORTED,
                        reason="nomination_only_not_authoritative",
                        backend_id=self._backend_identity()[0],
                        candidate_id=candidate.candidate_id,
                        invalidators=invalidators,
                    )
                    adjusted.append(
                        CandidateFacetResult(
                            obligation_id=item.obligation_id,
                            obligation_kind=item.obligation_kind,
                            candidate_id=item.candidate_id,
                            receipt=receipt,
                            disposition=FacetDisposition.UNSUPPORTED,
                            reason_codes=("nomination_only_not_authoritative",),
                            cache_key_id=item.cache_key_id,
                            from_cache=item.from_cache,
                        )
                    )
                else:
                    adjusted.append(item)
            return tuple(adjusted)
        return results

    def prove_behavior(
        self,
        compilation: ChangePropagationObligationCompilation,
        claim: BehaviorRefinementClaim,
        *,
        premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        reconstruction_inputs: Mapping[str, Any] | None = None,
        invalidators: Sequence[str] = (),
    ) -> BehaviorProofSet:
        if not isinstance(compilation, ChangePropagationObligationCompilation):
            raise MissingInputSynthesisError(
                "compilation must be ChangePropagationObligationCompilation"
            )
        if not isinstance(claim, BehaviorRefinementClaim):
            raise MissingInputSynthesisError("claim must be a BehaviorRefinementClaim")
        facets = [
            item
            for item in compilation.obligations
            if item.kind in _BEHAVIOR_FACETS
            and (
                not item.behavior_refinement_claim_id
                or item.behavior_refinement_claim_id == claim.claim_id
            )
        ]
        results = tuple(
            self.prove_obligation(
                item,
                candidate_id=claim.behavior_id,
                premises=premises,
                reconstruction_inputs=reconstruction_inputs,
                invalidators=invalidators,
            )
            for item in sorted(facets, key=lambda item: item.kind.value)
        )
        dispositions = [item.disposition for item in results]
        aggregate = _aggregate_facet_dispositions(dispositions)
        if aggregate is FacetDisposition.PROVED:
            disposition = SynthesisDisposition.UNIQUE_PROVED
        else:
            disposition = _facet_to_synthesis(aggregate)
        reasons = tuple(sorted({code for item in results for code in item.reason_codes}))
        unsatisfied = tuple(
            sorted({clause for item in results for clause in item.unsatisfied_clauses})
        )
        roots = compilation.roots
        return BehaviorProofSet(
            behavior_id=claim.behavior_id,
            consumer_id=compilation.consumer_id,
            disposition=disposition,
            facet_results=results,
            reason_codes=reasons,
            unsatisfied_clauses=unsatisfied,
            refinement_claim_id=claim.claim_id,
            placement_decision_ref=claim.placement_decision_ref,
            repository_id=roots.repository_id,
            tree_id=roots.candidate_tree_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
        )

    def _aggregate_value_mapping(
        self,
        *,
        requirement: MissingInputRequirement | None,
        requirement_id: str,
        consumer_id: str,
        candidates: Sequence[ValueCandidate],
        per_candidate: Mapping[str, tuple[CandidateFacetResult, ...]],
        mapping_claim: ValueMappingClaim | None,
        roots: PropagationAuthorityRoots,
        allow_upstream: bool,
        migration_obligation_id: str,
    ) -> ValueMappingProof:
        proved: list[str] = []
        refuted: list[str] = []
        inconclusive: list[str] = []
        all_results: list[CandidateFacetResult] = []
        reasons: set[str] = set()
        unsatisfied: set[str] = set()
        candidate_facet_status: dict[str, FacetDisposition] = {}

        # Evaluate in content-id order so search/nomination order never leaks.
        ordered = sorted(candidates, key=lambda item: item.candidate_id)
        for candidate in ordered:
            results = per_candidate.get(candidate.candidate_id, ())
            all_results.extend(results)
            for item in results:
                reasons.update(item.reason_codes)
                unsatisfied.update(item.unsatisfied_clauses)
            if not results:
                inconclusive.append(candidate.candidate_id)
                candidate_facet_status[candidate.candidate_id] = FacetDisposition.UNKNOWN
                reasons.add("no_facet_results")
                continue
            status = _aggregate_facet_dispositions(
                [item.disposition for item in results]
            )
            candidate_facet_status[candidate.candidate_id] = status
            if status is FacetDisposition.PROVED:
                proved.append(candidate.candidate_id)
            elif status is FacetDisposition.REFUTED:
                refuted.append(candidate.candidate_id)
            else:
                inconclusive.append(candidate.candidate_id)

        proved_ids = tuple(sorted(set(proved)))
        refuted_ids = tuple(sorted(set(refuted)))
        inconclusive_ids = tuple(sorted(set(inconclusive)))

        if len(proved_ids) == 1 and not inconclusive_ids:
            disposition = SynthesisDisposition.UNIQUE_PROVED
            winner = next(c for c in ordered if c.candidate_id == proved_ids[0])
            expression_ref = winner.expression_ref
            type_ref = winner.type_ref
        elif len(proved_ids) >= 2:
            # Multiple independently reconstructed candidates: ambiguity.  Do not
            # rank, score, or prefer the first nomination.
            disposition = SynthesisDisposition.AMBIGUOUS
            expression_ref = ""
            type_ref = requirement.type_ref if requirement is not None else ""
            reasons.add("multiple_independently_proved_candidates")
        elif proved_ids and inconclusive_ids:
            # Partial reconstruction cannot invent uniqueness or refutation.
            disposition = SynthesisDisposition.UNKNOWN
            expression_ref = ""
            type_ref = requirement.type_ref if requirement is not None else ""
            reasons.add("mixed_proved_and_inconclusive_candidates")
        elif not ordered:
            disposition = SynthesisDisposition.REFUTED
            expression_ref = ""
            type_ref = requirement.type_ref if requirement is not None else ""
            reasons.add("no_candidate")
        else:
            statuses = list(candidate_facet_status.values())
            aggregate = _aggregate_facet_dispositions(statuses)
            if aggregate is FacetDisposition.REFUTED or (
                refuted_ids and not inconclusive_ids and not proved_ids
            ):
                disposition = SynthesisDisposition.REFUTED
                reasons.add("all_candidates_refuted" if ordered else "no_candidate")
            else:
                disposition = _facet_to_synthesis(aggregate)
                if disposition is SynthesisDisposition.UNIQUE_PROVED:
                    # Without a single proved candidate, PROVED aggregate is impossible;
                    # fall back to unknown rather than inventing uniqueness.
                    disposition = SynthesisDisposition.UNKNOWN
            expression_ref = ""
            type_ref = requirement.type_ref if requirement is not None else ""

        upstream: UpstreamThreadRequirement | None = None
        if (
            allow_upstream
            and disposition
            in {
                SynthesisDisposition.REFUTED,
                SynthesisDisposition.UNKNOWN,
                SynthesisDisposition.UNSUPPORTED,
            }
            and requirement is not None
            and requirement.propagation_depth_bound > 0
        ):
            # Upstream threading requires an explicit origin; never invent a rootless
            # requirement from a failed local search.
            upstream = UpstreamThreadRequirement(
                origin_requirement_id=requirement.requirement_id,
                origin_consumer_id=consumer_id,
                origin_obligation_id=migration_obligation_id or requirement.obligation_id,
                parameter_name=requirement.parameter_name,
                type_ref=requirement.type_ref,
                reason_codes=("thread_upstream_with_origin", disposition.value),
            )
            reasons.add("upstream_thread_with_origin")

        if mapping_claim is not None and disposition is SynthesisDisposition.UNIQUE_PROVED:
            expression_ref = expression_ref or mapping_claim.expression_ref
            type_ref = type_ref or mapping_claim.type_ref

        return ValueMappingProof(
            requirement_id=requirement_id,
            consumer_id=consumer_id,
            disposition=disposition,
            facet_results=tuple(
                sorted(
                    all_results,
                    key=lambda item: (item.candidate_id, item.obligation_kind.value),
                )
            ),
            proved_candidate_ids=proved_ids,
            refuted_candidate_ids=refuted_ids,
            inconclusive_candidate_ids=inconclusive_ids,
            reason_codes=tuple(sorted(reasons)),
            unsatisfied_clauses=tuple(sorted(unsatisfied)),
            expression_ref=expression_ref,
            type_ref=type_ref,
            mapping_claim_id=mapping_claim.claim_id if mapping_claim is not None else "",
            upstream_thread=upstream,
            repository_id=roots.repository_id,
            tree_id=roots.candidate_tree_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
        )

    def synthesize(
        self,
        compilation: ChangePropagationObligationCompilation,
        *,
        premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        value_candidates: Sequence[ValueCandidate] = (),
        missing_inputs: Sequence[MissingInputRequirement] = (),
        reconstruction_inputs: Mapping[str, Any] | None = None,
        invalidators: Sequence[str] = (),
        allow_upstream_threading: bool = True,
    ) -> MissingInputSynthesisReceipt:
        """Prove/refute/reconstruct every value-mapping and behavior clause."""

        if not isinstance(compilation, ChangePropagationObligationCompilation):
            raise MissingInputSynthesisError(
                "compilation must be ChangePropagationObligationCompilation"
            )
        roots = compilation.roots
        # Bind invalidators to every authority root that can stale a receipt.
        bound_invalidators = _ids(
            (
                *invalidators,
                roots.candidate_tree_id,
                roots.base_tree_id,
                roots.toolchain_id,
                roots.policy_id,
                roots.translator_id,
                roots.graph_id,
                roots.index_id,
                roots.config_id,
            ),
            "invalidators",
        )

        requirements_by_id: dict[str, MissingInputRequirement] = {}
        for item in missing_inputs:
            if not isinstance(item, MissingInputRequirement):
                raise MissingInputSynthesisError(
                    "missing_inputs must be MissingInputRequirement values"
                )
            requirements_by_id[item.requirement_id] = item

        candidates_by_requirement: dict[str, list[ValueCandidate]] = {}
        for item in value_candidates:
            if not isinstance(item, ValueCandidate):
                raise MissingInputSynthesisError(
                    "value_candidates must be ValueCandidate values"
                )
            if item.disposition is ValueCandidateDisposition.REFUTED:
                # Pre-refuted nominations still appear for audit but are not re-proved.
                candidates_by_requirement.setdefault(item.requirement_id, []).append(item)
                continue
            candidates_by_requirement.setdefault(item.requirement_id, []).append(item)

        # Prefer explicit requirements; fall back to claim requirement ids.
        requirement_ids = sorted(
            set(requirements_by_id)
            | set(candidates_by_requirement)
            | {claim.requirement_id for claim in compilation.value_mapping_claims}
        )

        value_proofs: list[ValueMappingProof] = []
        for requirement_id in requirement_ids:
            requirement = requirements_by_id.get(requirement_id)
            candidates = list(candidates_by_requirement.get(requirement_id, ()))
            # When the compilation already bound a mapping claim, ensure that
            # candidate is present for evaluation without inventing new ones.
            mapping_claim = next(
                (
                    claim
                    for claim in compilation.value_mapping_claims
                    if claim.requirement_id == requirement_id
                ),
                None,
            )
            if mapping_claim is not None and not any(
                item.candidate_id == mapping_claim.candidate_id for item in candidates
            ):
                # Claim-bound candidate identity is evaluated even if the caller
                # omitted the ValueCandidate record; create a synthetic shell only
                # for routing, never as semantic authority.
                pass

            per_candidate: dict[str, tuple[CandidateFacetResult, ...]] = {}
            # Sort candidates by id before proving so wall-clock order cannot
            # influence which independent proof is observed first in aggregates.
            for candidate in sorted(candidates, key=lambda item: item.candidate_id):
                if candidate.disposition is ValueCandidateDisposition.REFUTED:
                    # Materialize explicit refutation facets without calling the backend.
                    facets = [
                        item
                        for item in compilation.obligations
                        if item.kind in _VALUE_MAPPING_FACETS
                    ]
                    pre_refuted: list[CandidateFacetResult] = []
                    for facet in sorted(facets, key=lambda item: item.kind.value):
                        key = self._cache_key(
                            facet,
                            (),
                            candidate_id=candidate.candidate_id,
                            invalidators=bound_invalidators,
                        )
                        # Without an independently verified counterexample, pre-refuted
                        # dispositions stay non-conclusive rather than authoritative REFUTED.
                        pre_refuted.append(
                            CandidateFacetResult(
                                obligation_id=facet.obligation_id,
                                obligation_kind=facet.kind,
                                candidate_id=candidate.candidate_id,
                                receipt=self._non_conclusive_receipt(
                                    facet,
                                    verdict=ProofVerdict.INCONCLUSIVE,
                                    reason="pre_refuted_without_verified_counterexample",
                                    backend_id=self._backend_identity()[0],
                                    candidate_id=candidate.candidate_id,
                                    invalidators=bound_invalidators,
                                ),
                                disposition=FacetDisposition.UNKNOWN,
                                reason_codes=(
                                    "pre_refuted_without_verified_counterexample",
                                    *candidate.rejection_reasons,
                                ),
                                cache_key_id=key.key_id,
                                unsatisfied_clauses=tuple(candidate.rejection_reasons),
                            )
                        )
                    per_candidate[candidate.candidate_id] = tuple(pre_refuted)
                    continue
                per_candidate[candidate.candidate_id] = self.prove_candidate_mapping(
                    compilation,
                    candidate,
                    premises=premises,
                    reconstruction_inputs=reconstruction_inputs,
                    invalidators=bound_invalidators,
                )

            value_proofs.append(
                self._aggregate_value_mapping(
                    requirement=requirement,
                    requirement_id=requirement_id,
                    consumer_id=compilation.consumer_id,
                    candidates=candidates,
                    per_candidate=per_candidate,
                    mapping_claim=mapping_claim,
                    roots=roots,
                    allow_upstream=allow_upstream_threading,
                    migration_obligation_id=compilation.migration_obligation_id,
                )
            )

        behavior_sets = tuple(
            self.prove_behavior(
                compilation,
                claim,
                premises=premises,
                reconstruction_inputs=reconstruction_inputs,
                invalidators=bound_invalidators,
            )
            for claim in sorted(
                compilation.behavior_refinement_claims, key=lambda item: item.behavior_id
            )
        )

        backend_id, backend_version = self._backend_identity()
        reasons = tuple(
            sorted(
                {
                    *(code for proof in value_proofs for code in proof.reason_codes),
                    *(code for proof in behavior_sets for code in proof.reason_codes),
                }
            )
        )
        return MissingInputSynthesisReceipt(
            roots=roots,
            delta_id=compilation.delta_id,
            consumer_id=compilation.consumer_id,
            migration_obligation_id=compilation.migration_obligation_id,
            value_mapping_proofs=tuple(value_proofs),
            behavior_proof_sets=behavior_sets,
            backend_id=backend_id,
            backend_version=backend_version,
            reason_codes=reasons,
            invalidators=bound_invalidators,
        )


def reconstruct_missing_input_proof(
    obligation: ChangePropagationObligation,
    *,
    candidate_id: str,
    backend: Any,
    premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    reconstruction_inputs: Mapping[str, Any] | None = None,
    cache: FormalVerificationCache | None = None,
    invalidators: Sequence[str] = (),
) -> CandidateFacetResult:
    """One-obligation convenience entry retaining all fail-closed checks."""

    return MissingInputSynthesizer(backend, cache=cache).prove_obligation(
        obligation,
        candidate_id=candidate_id,
        premises=premises,
        reconstruction_inputs=reconstruction_inputs,
        invalidators=invalidators,
    )


# ProofReconstructor-compatible alias for interface documentation.
ProofReconstructor = MissingInputSynthesizer


__all__ = [
    "BEHAVIOR_PROOF_SET_SCHEMA",
    "BehaviorProofSet",
    "CANDIDATE_FACET_RESULT_SCHEMA",
    "CandidateFacetResult",
    "FacetDisposition",
    "IPFS_DATASETS_LOGIC_PROVIDER_ID",
    "MISSING_INPUT_SYNTHESIS_INTERFACE",
    "MISSING_INPUT_SYNTHESIS_RECEIPT_SCHEMA",
    "MissingInputSynthesisError",
    "MissingInputSynthesisReceipt",
    "MissingInputSynthesizer",
    "ProofReconstructor",
    "SynthesisDisposition",
    "UPSTREAM_THREAD_SCHEMA",
    "UpstreamThreadRequirement",
    "VALUE_MAPPING_PROOF_SCHEMA",
    "ValueMappingProof",
    "reconstruct_missing_input_proof",
]
