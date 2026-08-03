"""Proof-gated, deterministic ordering for contract-repair candidates.

Candidate retrieval is a recall operation and implementation-site admission is
only a statement about a possible site.  This module is the deliberately
narrow join which refuses to rank a candidate until its target, authority,
independent expectation, complete supported slice, and reconstructed proofs
all bind the same current authority roots.  Its output is a ranking receipt;
it is *not* a target decision and cannot grant a write path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..planning.implementation_site_admissibility import (
    PlacementDecision,
    PlacementDisposition,
)
from ..proof.contract_repair_prover import CandidateProofBundle
from ..proof.formal_verification_contracts import content_identity
from .contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    MAX_CANDIDATE_COUNT,
    RepairCandidate,
    RepairStrategy,
    candidate_set_identity,
)


CONTRACT_REPAIR_RERANKER_INTERFACE: Final[str] = "ContractRepairReranker@1"
CONTRACT_REPAIR_RERANK_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-rerank-policy@1"
)
CONTRACT_REPAIR_RERANK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-rerank-receipt@1"
)
SCORE_SCALE: Final[int] = 1_000_000


class ContractRepairRerankerError(ValueError):
    """A ranking input attempted to weaken a proof or authority boundary."""


class CandidateEligibilityDisposition(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class RerankDisposition(str, Enum):
    RANKED = "ranked"
    AMBIGUOUS = "ambiguous"
    ABSTAINED = "abstained"


class RankingSignal(str, Enum):
    """The normative lexicographic ordering; earlier signals dominate later."""

    PROOF_COVERAGE = "proof_coverage"
    LINEAGE = "lineage"
    GRAPH_OWNERSHIP = "graph_ownership"
    AUTHORITATIVE_SPEC_TEST = "authoritative_spec_test"
    AST = "ast"
    LEXICAL = "lexical"
    VECTOR = "vector"


RANKING_ORDER: Final[tuple[RankingSignal, ...]] = tuple(RankingSignal)
_INDEPENDENT_EXPECTATION_KINDS: Final[frozenset[str]] = frozenset(
    {
        "conformance_test",
        "idl",
        "manifest",
        "normative_spec",
        "public_signature",
        "reviewed_schema",
        "reviewed_spec",
        "reviewed_stub",
        "reviewed_test",
        "schema",
        "specification",
    }
)
_CANDIDATE_DEFINED_MARKERS: Final[frozenset[str]] = frozenset(
    {"candidate", "embedding", "llm", "model", "retrieval", "vector"}
)


def _ids(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairRerankerError(f"{name} must be a sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ContractRepairRerankerError(f"{name} must contain non-empty identifiers")
        result.add(value.strip())
    normalized = tuple(sorted(result))
    if required and not normalized:
        raise ContractRepairRerankerError(f"{name} must not be empty")
    return normalized


def _refs(values: Sequence[EvidenceReference], name: str, *, required: bool = False) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairRerankerError(f"{name} must be evidence references")
    if not all(isinstance(value, EvidenceReference) for value in values):
        raise ContractRepairRerankerError(f"{name} must contain EvidenceReference values")
    normalized = tuple(sorted(set(values), key=lambda value: value.content_id))
    if required and not normalized:
        raise ContractRepairRerankerError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class RankingEvidence:
    """One bounded, provenance-bearing soft-ranking signal.

    A missing signal is represented by no ``RankingEvidence`` row and scores
    zero.  It can therefore never increase another signal's contribution.
    """

    signal: RankingSignal
    value: int
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal", RankingSignal(self.signal))
        if isinstance(self.value, bool) or not isinstance(self.value, int) or not 0 <= self.value <= SCORE_SCALE:
            raise ContractRepairRerankerError(
                f"{self.signal.value} score must be an integer from 0 through {SCORE_SCALE}"
            )
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "ranking evidence", required=True))


@dataclass(frozen=True)
class RerankPolicy:
    """Fixed scoring and ambiguity settings, bound into every receipt."""

    policy_id: str
    minimum_margin: int = 1
    weights: tuple[tuple[RankingSignal, int], ...] = tuple(
        (signal, 1) for signal in RANKING_ORDER
    )
    tie_breaker: str = "candidate_content_id_ascending"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ContractRepairRerankerError("policy_id is required")
        object.__setattr__(self, "policy_id", self.policy_id.strip())
        if isinstance(self.minimum_margin, bool) or not isinstance(self.minimum_margin, int) or not 1 <= self.minimum_margin <= SCORE_SCALE:
            raise ContractRepairRerankerError("minimum_margin must be a positive bounded integer")
        if self.tie_breaker != "candidate_content_id_ascending":
            raise ContractRepairRerankerError("unsupported rerank tie breaker")
        rows = self.weights.items() if isinstance(self.weights, Mapping) else self.weights
        normalized: list[tuple[RankingSignal, int]] = []
        for row in rows:
            try:
                signal, weight = row
            except (TypeError, ValueError) as exc:
                raise ContractRepairRerankerError("weights must contain signal and weight pairs") from exc
            signal = RankingSignal(signal)
            if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= SCORE_SCALE:
                raise ContractRepairRerankerError("weights must be positive bounded integers")
            normalized.append((signal, weight))
        normalized.sort(key=lambda row: RANKING_ORDER.index(row[0]))
        if tuple(signal for signal, _ in normalized) != RANKING_ORDER:
            raise ContractRepairRerankerError("weights must name each ranking signal exactly once")
        object.__setattr__(self, "weights", tuple(normalized))

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CONTRACT_REPAIR_RERANK_POLICY_SCHEMA,
            "interface": CONTRACT_REPAIR_RERANKER_INTERFACE,
            "policy_id": self.policy_id,
            "minimum_margin": self.minimum_margin,
            "weights": [{"signal": signal.value, "weight": weight} for signal, weight in self.weights],
            "tie_breaker": self.tie_breaker,
        }


@dataclass(frozen=True)
class CandidateEligibility:
    """All compact, independently produced facts required before ranking.

    ``target_valid`` and ``write_authorized`` are assertions made by their
    named evidence producers.  They do not themselves grant path authority;
    that remains the downstream ``RepairTargetDecision`` boundary.
    """

    candidate: RepairCandidate
    proof_bundle: CandidateProofBundle
    placement_decision: PlacementDecision
    expectation_roots: AuthorityRoots
    expectation_refs: tuple[EvidenceReference, ...]
    complete_supported_slice: bool
    target_valid: bool
    target_validity_refs: tuple[EvidenceReference, ...]
    write_authorized: bool
    write_authority_refs: tuple[EvidenceReference, ...]
    mandatory_obligation_ids: tuple[str, ...]
    ranking_evidence: tuple[RankingEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RepairCandidate):
            raise ContractRepairRerankerError("candidate must be RepairCandidate")
        if not isinstance(self.proof_bundle, CandidateProofBundle):
            raise ContractRepairRerankerError("proof_bundle must be CandidateProofBundle")
        if not isinstance(self.placement_decision, PlacementDecision):
            raise ContractRepairRerankerError("placement_decision must be PlacementDecision")
        if not isinstance(self.expectation_roots, AuthorityRoots):
            raise ContractRepairRerankerError("expectation_roots must be AuthorityRoots")
        object.__setattr__(self, "expectation_refs", _refs(self.expectation_refs, "expectation_refs", required=True))
        object.__setattr__(self, "target_validity_refs", _refs(self.target_validity_refs, "target_validity_refs", required=True))
        object.__setattr__(self, "write_authority_refs", _refs(self.write_authority_refs, "write_authority_refs", required=True))
        object.__setattr__(self, "mandatory_obligation_ids", _ids(self.mandatory_obligation_ids, "mandatory_obligation_ids", required=True))
        for name in ("complete_supported_slice", "target_valid", "write_authorized"):
            if not isinstance(getattr(self, name), bool):
                raise ContractRepairRerankerError(f"{name} must be boolean")
        if isinstance(self.ranking_evidence, (str, bytes, bytearray)) or not isinstance(self.ranking_evidence, Sequence):
            raise ContractRepairRerankerError("ranking_evidence must be a sequence")
        if not all(isinstance(item, RankingEvidence) for item in self.ranking_evidence):
            raise ContractRepairRerankerError("ranking_evidence must contain RankingEvidence values")
        rows = tuple(sorted(self.ranking_evidence, key=lambda item: RANKING_ORDER.index(item.signal)))
        if len({item.signal for item in rows}) != len(rows):
            raise ContractRepairRerankerError("ranking_evidence cannot repeat a signal")
        object.__setattr__(self, "ranking_evidence", rows)


@dataclass(frozen=True)
class CandidateRank:
    """Eligibility outcome plus the full lexicographic score vector."""

    candidate_id: str
    disposition: CandidateEligibilityDisposition
    score_vector: tuple[int, ...]
    reason_codes: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ContractRepairRerankerError("candidate_id is required")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "disposition", CandidateEligibilityDisposition(self.disposition))
        if len(self.score_vector) != len(RANKING_ORDER) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.score_vector
        ):
            raise ContractRepairRerankerError("score_vector must contain one non-negative integer per ranking signal")
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(self, "proof_receipt_ids", _ids(self.proof_receipt_ids, "proof_receipt_ids"))
        if self.disposition is CandidateEligibilityDisposition.ELIGIBLE and self.reason_codes:
            raise ContractRepairRerankerError("eligible ranks cannot carry rejection reasons")
        if self.disposition is CandidateEligibilityDisposition.INELIGIBLE and not self.reason_codes:
            raise ContractRepairRerankerError("ineligible ranks require rejection reasons")

    @property
    def eligible(self) -> bool:
        return self.disposition is CandidateEligibilityDisposition.ELIGIBLE


@dataclass(frozen=True)
class RerankReceipt:
    """Replayable result of filtering and ordering a complete candidate set."""

    roots: AuthorityRoots
    candidate_set_id: str
    policy_receipt_id: str
    ranks: tuple[CandidateRank, ...]
    disposition: RerankDisposition
    selected_candidate_id: str = ""
    reason_codes: tuple[str, ...] = ()
    schema: str = CONTRACT_REPAIR_RERANK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairRerankerError("roots must be AuthorityRoots")
        if not isinstance(self.candidate_set_id, str) or not self.candidate_set_id.strip():
            raise ContractRepairRerankerError("candidate_set_id is required")
        if not isinstance(self.policy_receipt_id, str) or not self.policy_receipt_id.strip():
            raise ContractRepairRerankerError("policy_receipt_id is required")
        if self.schema != CONTRACT_REPAIR_RERANK_RECEIPT_SCHEMA:
            raise ContractRepairRerankerError("unsupported rerank receipt schema")
        if not self.ranks or not all(isinstance(item, CandidateRank) for item in self.ranks):
            raise ContractRepairRerankerError("ranks must be a non-empty CandidateRank sequence")
        ranks = tuple(sorted(self.ranks, key=lambda item: item.candidate_id))
        if len({item.candidate_id for item in ranks}) != len(ranks):
            raise ContractRepairRerankerError("ranks cannot repeat a candidate")
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "disposition", RerankDisposition(self.disposition))
        if not isinstance(self.selected_candidate_id, str):
            raise ContractRepairRerankerError("selected_candidate_id must be a string")
        object.__setattr__(self, "selected_candidate_id", self.selected_candidate_id.strip())
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        if self.disposition is RerankDisposition.RANKED:
            selected = next((item for item in ranks if item.candidate_id == self.selected_candidate_id), None)
            if selected is None or not selected.eligible:
                raise ContractRepairRerankerError("ranked receipt requires one eligible selected candidate")
        elif self.selected_candidate_id:
            raise ContractRepairRerankerError("ambiguous and abstained receipts cannot select a target")

    @property
    def write_paths(self) -> tuple[str, ...]:
        """Reranking is never write authority."""

        return ()

    @property
    def permitted_write_paths(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "interface": CONTRACT_REPAIR_RERANKER_INTERFACE,
            "roots": self.roots.to_dict(),
            "candidate_set_id": self.candidate_set_id,
            "policy_receipt_id": self.policy_receipt_id,
            "ranks": [
                {
                    "candidate_id": item.candidate_id,
                    "disposition": item.disposition.value,
                    "score_vector": list(item.score_vector),
                    "reason_codes": list(item.reason_codes),
                    "proof_receipt_ids": list(item.proof_receipt_ids),
                }
                for item in self.ranks
            ],
            "disposition": self.disposition.value,
            "selected_candidate_id": self.selected_candidate_id,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())


class ContractRepairReranker:
    """Hard-filter complete candidate sets and lexicographically rerank survivors."""

    def __init__(self, policy: RerankPolicy | None = None) -> None:
        self.policy = policy

    def rank(
        self,
        candidates: Sequence[CandidateEligibility],
        *,
        roots: AuthorityRoots,
        policy: RerankPolicy | None = None,
    ) -> RerankReceipt:
        """Return a receipt, abstaining rather than repairing malformed input.

        The only deterministic tie breaker is used to order the receipt rows;
        it never converts a substantive tie or insufficient margin into a
        selected target.
        """

        if not isinstance(roots, AuthorityRoots):
            raise ContractRepairRerankerError("roots must be AuthorityRoots")
        active_policy = policy or self.policy or RerankPolicy(roots.policy_id)
        if not isinstance(active_policy, RerankPolicy):
            raise ContractRepairRerankerError("policy must be RerankPolicy")
        if active_policy.policy_id != roots.policy_id:
            raise ContractRepairRerankerError("rerank policy must bind the exact authority policy root")
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
            raise ContractRepairRerankerError("candidates must be a sequence")
        rows = tuple(candidates)
        if not rows or len(rows) > MAX_CANDIDATE_COUNT or not all(isinstance(item, CandidateEligibility) for item in rows):
            raise ContractRepairRerankerError("candidates must be a bounded non-empty CandidateEligibility sequence")
        candidate_records = tuple(item.candidate for item in rows)
        try:
            candidate_set_id = candidate_set_identity(candidate_records)
        except ValueError as exc:
            raise ContractRepairRerankerError("candidate set is invalid") from exc

        ranks = tuple(self._evaluate(item, roots, candidate_set_id, active_policy) for item in rows)
        eligible = tuple(item for item in ranks if item.eligible)
        if not eligible:
            return RerankReceipt(
                roots, candidate_set_id, active_policy.receipt_id, ranks,
                RerankDisposition.ABSTAINED, reason_codes=("no_eligible_candidate",),
            )
        ordered = tuple(sorted(eligible, key=lambda item: (tuple(-value for value in item.score_vector), item.candidate_id)))
        if len(ordered) > 1:
            margin = self._margin(ordered[0].score_vector, ordered[1].score_vector)
            if margin is None:
                return RerankReceipt(
                    roots, candidate_set_id, active_policy.receipt_id, ranks,
                    RerankDisposition.AMBIGUOUS, reason_codes=("rank_tie",),
                )
            if margin < active_policy.minimum_margin:
                return RerankReceipt(
                    roots, candidate_set_id, active_policy.receipt_id, ranks,
                    RerankDisposition.AMBIGUOUS, reason_codes=("insufficient_rank_margin",),
                )
        return RerankReceipt(
            roots, candidate_set_id, active_policy.receipt_id, ranks,
            RerankDisposition.RANKED, selected_candidate_id=ordered[0].candidate_id,
        )

    rerank = rank
    evaluate = rank
    assess = rank

    @staticmethod
    def _margin(first: tuple[int, ...], second: tuple[int, ...]) -> int | None:
        for left, right in zip(first, second):
            if left != right:
                return left - right
        return None

    def _evaluate(
        self,
        item: CandidateEligibility,
        roots: AuthorityRoots,
        candidate_set_id: str,
        policy: RerankPolicy,
    ) -> CandidateRank:
        candidate = item.candidate
        reasons: set[str] = set()
        if candidate.roots != roots or item.expectation_roots != roots:
            reasons.add("authority_roots_mismatch")
        if candidate.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}:
            reasons.add("candidate_strategy_not_repairable")
        if candidate.rejection_reasons:
            reasons.add("candidate_previously_rejected")
        if candidate.candidate_write_paths or candidate.permitted_read_paths:
            reasons.add("candidate_attempted_to_grant_path_authority")
        if not item.target_valid or not item.target_validity_refs:
            reasons.add("target_or_site_not_valid")
        if not item.write_authorized or not item.write_authority_refs:
            reasons.add("write_authority_not_exact")
        placement = item.placement_decision
        if (
            placement.disposition is not PlacementDisposition.ADMITTED
            or placement.candidate_set_id != candidate_set_id
            or placement.selected_candidate_id != candidate.content_id
            or placement.target_path != candidate.target_span.path
            or not placement.proof_receipt_ids
        ):
            reasons.add("placement_not_exactly_admitted")
        if not item.complete_supported_slice:
            reasons.add("incomplete_supported_slice")
        if not self._independent_expectation(item.expectation_refs, candidate):
            reasons.add("independent_expectation_missing")
        reasons.update(self._proof_reasons(item, roots))
        if reasons:
            return CandidateRank(
                candidate.content_id, CandidateEligibilityDisposition.INELIGIBLE,
                (0,) * len(RANKING_ORDER), tuple(reasons),
            )
        by_signal = {row.signal: row for row in item.ranking_evidence}
        weights = dict(policy.weights)
        score = tuple(
            by_signal[signal].value * weights[signal] if signal in by_signal else 0
            for signal in RANKING_ORDER
        )
        proof_ids = tuple(sorted(result.receipt.receipt_id for result in item.proof_bundle.results))
        return CandidateRank(candidate.content_id, CandidateEligibilityDisposition.ELIGIBLE, score, proof_receipt_ids=proof_ids)

    @staticmethod
    def _independent_expectation(refs: tuple[EvidenceReference, ...], candidate: RepairCandidate) -> bool:
        candidate_ref_ids = {ref.content_id for ref in candidate.evidence_refs}
        for ref in refs:
            kind = ref.kind.casefold().replace("-", "_")
            producer = ref.producer_id.casefold().replace("-", "_")
            if ref.content_id in candidate_ref_ids:
                continue
            if kind in _INDEPENDENT_EXPECTATION_KINDS and not any(marker in producer for marker in _CANDIDATE_DEFINED_MARKERS):
                return True
        return False

    @staticmethod
    def _proof_reasons(item: CandidateEligibility, roots: AuthorityRoots) -> set[str]:
        candidate = item.candidate
        bundle = item.proof_bundle
        reasons: set[str] = set()
        if bundle.candidate_id != candidate.content_id:
            reasons.add("proof_candidate_binding_mismatch")
        if bundle.repository_id != roots.repository_id or bundle.tree_id != roots.tree_id:
            reasons.add("proof_tree_binding_mismatch")
        result_by_id = {result.obligation_id: result for result in bundle.results}
        mandatory = set(item.mandatory_obligation_ids)
        if not mandatory.issubset(result_by_id):
            reasons.add("mandatory_proof_missing")
        if bundle.counterexample_refs or any(result.counterexample is not None for result in bundle.results):
            reasons.add("counterexample_present")
        for result in bundle.results:
            # A bundle is a single candidate-specific proof assertion.  An
            # otherwise authoritative extra receipt from another root must not
            # be smuggled into it merely because it is not score-relevant.
            receipt = result.receipt
            if (
                receipt.repository_id != roots.repository_id
                or receipt.repository_tree_id != roots.tree_id
                or receipt.translator_id != roots.translator_id
                or receipt.toolchain_id != roots.toolchain_id
                or receipt.policy_id != roots.policy_id
            ):
                reasons.add("proof_receipt_binding_mismatch")
        for obligation_id in mandatory:
            result = result_by_id.get(obligation_id)
            if result is None:
                continue
            if not result.authoritative:
                reasons.add("mandatory_proof_not_reconstructed")
                continue
            receipt = result.receipt
            if (
                receipt.repository_id != roots.repository_id
                or receipt.repository_tree_id != roots.tree_id
                or receipt.translator_id != roots.translator_id
                or receipt.toolchain_id != roots.toolchain_id
                or receipt.policy_id != roots.policy_id
            ):
                reasons.add("proof_receipt_binding_mismatch")
        if not bundle.candidate_authoritative:
            reasons.add("proof_bundle_not_authoritative")
        return reasons


__all__ = [
    "CONTRACT_REPAIR_RERANKER_INTERFACE",
    "CONTRACT_REPAIR_RERANK_POLICY_SCHEMA",
    "CONTRACT_REPAIR_RERANK_RECEIPT_SCHEMA",
    "SCORE_SCALE",
    "CandidateEligibility",
    "CandidateEligibilityDisposition",
    "CandidateRank",
    "ContractRepairReranker",
    "ContractRepairRerankerError",
    "RANKING_ORDER",
    "RankingEvidence",
    "RankingSignal",
    "RerankDisposition",
    "RerankPolicy",
    "RerankReceipt",
]
