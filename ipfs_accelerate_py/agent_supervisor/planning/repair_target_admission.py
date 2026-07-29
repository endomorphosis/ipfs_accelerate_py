"""Issue one expiring, proof-gated repair target decision or abstain.

The reranker decides whether a candidate is a unique winner; it deliberately
does not grant mutation authority.  This module is the final narrow boundary:
it replays the complete rerank receipt against the complete candidate set and
derives the decision's paths only from an exact repository authority for the
selected target span.  In particular, a candidate's proposed paths are never
used as authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
    candidate_set_identity,
)
from ..analysis.contract_repair_reranker import (
    CandidateEligibilityDisposition,
    CandidateRank,
    RerankDisposition,
    RerankReceipt,
)
from ..proof.formal_verification_contracts import content_identity

REPAIR_TARGET_ADMISSION_INTERFACE: Final[str] = "RepairTargetAdmission@1"
REPAIR_TARGET_ADMISSION_AUDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-target-admission-audit@1"
)


class RepairTargetAdmissionError(ValueError):
    """A target-admission input weakens the final authority boundary."""


class AdmissionInvalidator(str, Enum):
    """Closed causes that make an otherwise issued decision unusable."""

    ROOT_CHANGED = "root_changed"
    TARGET_MISSING = "target_missing"
    READ_ONLY_PATH = "read_only_path"
    PROOF_DOWNGRADE = "proof_downgrade"
    CANDIDATE_SET_MUTATION = "candidate_set_mutation"
    RANK_TIE = "rank_tie"
    LOW_MARGIN = "low_margin"
    EXPIRED = "expired"


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(char.isspace() for char in value.strip())
    ):
        raise RepairTargetAdmissionError(
            f"{name} must be a non-empty compact identifier"
        )
    return value.strip()


def _refs(
    values: Sequence[EvidenceReference], name: str, *, required: bool = True
) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairTargetAdmissionError(
            f"{name} must be a sequence of EvidenceReference values"
        )
    if not all(isinstance(item, EvidenceReference) for item in values):
        raise RepairTargetAdmissionError(
            f"{name} must contain EvidenceReference values"
        )
    result = tuple(sorted(set(values), key=lambda item: item.content_id))
    if required and not result:
        raise RepairTargetAdmissionError(f"{name} must not be empty")
    return result


def _ids(
    values: Sequence[str], name: str, *, required: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairTargetAdmissionError(f"{name} must be a sequence of identifiers")
    result = tuple(sorted({_identifier(value, name) for value in values}))
    if required and not result:
        raise RepairTargetAdmissionError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class DecisionExpiry:
    """A compact validity window, expressed as UTC epoch seconds."""

    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        for name in ("issued_at", "expires_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RepairTargetAdmissionError(
                    f"{name} must be a non-negative integer"
                )
        if self.expires_at <= self.issued_at:
            raise RepairTargetAdmissionError("expiry must be after issuance")

    def to_dict(self) -> dict[str, int]:
        return {"issued_at": self.issued_at, "expires_at": self.expires_at}

    @property
    def content_id(self) -> str:
        return content_identity({"schema": "repair-target-expiry@1", **self.to_dict()})

    def valid_at(self, now: int) -> bool:
        return (
            isinstance(now, int)
            and not isinstance(now, bool)
            and self.issued_at <= now < self.expires_at
        )


@dataclass(frozen=True)
class TargetRepositoryAuthority:
    """Repository-owned, exact authority for one selected candidate target.

    The span lists are intentionally the source of decision read/write paths.
    They must describe only the candidate's target path; an admission cannot
    use this object to broaden a repair into neighbouring files.
    """

    roots: AuthorityRoots
    candidate_set_id: str
    candidate_id: str
    target_span: SourceSpan
    permitted_read_spans: tuple[SourceSpan, ...]
    permitted_write_spans: tuple[SourceSpan, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    target_exists: bool = True
    insertion_anchor_proved: bool = False
    read_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(
            self.target_span, SourceSpan
        ):
            raise RepairTargetAdmissionError(
                "roots and target_span must be typed contracts"
            )
        object.__setattr__(
            self,
            "candidate_set_id",
            _identifier(self.candidate_set_id, "candidate_set_id"),
        )
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        for name in ("permitted_read_spans", "permitted_write_spans"):
            value = getattr(self, name)
            if (
                isinstance(value, (str, bytes, bytearray))
                or not isinstance(value, Sequence)
                or not value
            ):
                raise RepairTargetAdmissionError(
                    f"{name} must be a non-empty SourceSpan sequence"
                )
            if not all(isinstance(item, SourceSpan) for item in value):
                raise RepairTargetAdmissionError(
                    f"{name} must contain SourceSpan values"
                )
            spans = tuple(sorted(set(value)))
            if any(item.path != self.target_span.path for item in spans):
                raise RepairTargetAdmissionError(
                    f"{name} may only cover the selected target path"
                )
            object.__setattr__(self, name, spans)
        object.__setattr__(
            self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs")
        )
        for name in ("target_exists", "insertion_anchor_proved", "read_only"):
            if not isinstance(getattr(self, name), bool):
                raise RepairTargetAdmissionError(f"{name} must be boolean")
        if (
            self.target_span not in self.permitted_read_spans
            or self.target_span not in self.permitted_write_spans
        ):
            raise RepairTargetAdmissionError(
                "repository authority must include the exact selected target span"
            )

    @property
    def permitted_read_paths(self) -> tuple[str, ...]:
        return tuple(sorted({span.path for span in self.permitted_read_spans}))

    @property
    def permitted_write_paths(self) -> tuple[str, ...]:
        return tuple(sorted({span.path for span in self.permitted_write_spans}))


@dataclass(frozen=True)
class TargetAdmissionAudit:
    """Content-addressed replay evidence for the complete ranking outcome."""

    roots: AuthorityRoots
    candidate_set_id: str
    rerank_receipt_id: str
    ranks: tuple[CandidateRank, ...]
    expiry: DecisionExpiry
    decision_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(
            self.expiry, DecisionExpiry
        ):
            raise RepairTargetAdmissionError("roots and expiry must be typed contracts")
        object.__setattr__(
            self,
            "candidate_set_id",
            _identifier(self.candidate_set_id, "candidate_set_id"),
        )
        object.__setattr__(
            self,
            "rerank_receipt_id",
            _identifier(self.rerank_receipt_id, "rerank_receipt_id"),
        )
        if (
            not isinstance(self.ranks, Sequence)
            or not self.ranks
            or not all(isinstance(item, CandidateRank) for item in self.ranks)
        ):
            raise RepairTargetAdmissionError(
                "ranks must be a non-empty CandidateRank sequence"
            )
        ranks = tuple(
            sorted(
                self.ranks,
                key=lambda row: (
                    tuple(-item for item in row.score_vector),
                    row.candidate_id,
                ),
            )
        )
        if len({item.candidate_id for item in ranks}) != len(ranks):
            raise RepairTargetAdmissionError("ranks cannot repeat candidates")
        object.__setattr__(self, "ranks", ranks)
        if self.decision_id:
            object.__setattr__(
                self, "decision_id", _identifier(self.decision_id, "decision_id")
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPAIR_TARGET_ADMISSION_AUDIT_SCHEMA,
            "interface": REPAIR_TARGET_ADMISSION_INTERFACE,
            "roots": self.roots.to_dict(),
            "candidate_set_id": self.candidate_set_id,
            "rerank_receipt_id": self.rerank_receipt_id,
            "ranks": [
                {
                    "candidate_id": row.candidate_id,
                    "disposition": row.disposition.value,
                    "score_vector": list(row.score_vector),
                    "reason_codes": list(row.reason_codes),
                    "proof_receipt_ids": list(row.proof_receipt_ids),
                }
                for row in self.ranks
            ],
            "expiry": self.expiry.to_dict(),
            "decision_id": self.decision_id,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class AdmissionResult:
    """Decision plus compact replay evidence and the exact span allowlists."""

    decision: RepairTargetDecision
    audit: TargetAdmissionAudit
    expiry: DecisionExpiry
    permitted_read_spans: tuple[SourceSpan, ...] = ()
    permitted_write_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RepairTargetDecision) or not isinstance(
            self.audit, TargetAdmissionAudit
        ):
            raise RepairTargetAdmissionError(
                "decision and audit must be typed contracts"
            )
        if self.decision.content_id != self.audit.decision_id:
            raise RepairTargetAdmissionError(
                "audit must bind the exact decision identity"
            )
        for name in ("permitted_read_spans", "permitted_write_spans"):
            values = getattr(self, name)
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes, bytearray))
                or not all(isinstance(item, SourceSpan) for item in values)
            ):
                raise RepairTargetAdmissionError(
                    f"{name} must contain SourceSpan values"
                )
            object.__setattr__(self, name, tuple(sorted(set(values))))
        if (
            self.decision.disposition is not DecisionDisposition.ADMITTED
            and self.permitted_write_spans
        ):
            raise RepairTargetAdmissionError("abstention cannot carry write spans")


class RepairTargetAdmission:
    """Replay a deterministic rerank and issue one exact decision or abstain."""

    def decide(
        self,
        candidates: Sequence[RepairCandidate],
        rerank_receipt: RerankReceipt,
        authorities: Sequence[TargetRepositoryAuthority],
        *,
        expiry: DecisionExpiry,
    ) -> RepairTargetDecision:
        return self.admit(
            candidates, rerank_receipt, authorities, expiry=expiry
        ).decision

    def admit(
        self,
        candidates: Sequence[RepairCandidate],
        rerank_receipt: RerankReceipt,
        authorities: Sequence[TargetRepositoryAuthority],
        *,
        expiry: DecisionExpiry,
    ) -> AdmissionResult:
        rows = self._candidates(candidates)
        if not isinstance(rerank_receipt, RerankReceipt) or not isinstance(
            expiry, DecisionExpiry
        ):
            raise RepairTargetAdmissionError(
                "rerank_receipt and expiry must be typed contracts"
            )
        candidate_set_id = candidate_set_identity(rows)
        problems = self._receipt_problems(rows, candidate_set_id, rerank_receipt)
        authority_map, authority_problems = self._authorities(
            authorities, rows, candidate_set_id
        )
        problems.update(authority_problems)

        selected = next(
            (
                item
                for item in rows
                if item.content_id == rerank_receipt.selected_candidate_id
            ),
            None,
        )
        decision: RepairTargetDecision
        spans: tuple[SourceSpan, ...] = ()
        if rerank_receipt.disposition is RerankDisposition.AMBIGUOUS:
            problems.update(self._ambiguity_reasons(rerank_receipt))
            decision = self._abstention(
                rows,
                candidate_set_id,
                RepairStrategy.AMBIGUOUS,
                problems,
                rerank_receipt,
                expiry,
            )
        elif (
            rerank_receipt.disposition is not RerankDisposition.RANKED
            or selected is None
            or problems
        ):
            decision = self._abstention(
                rows,
                candidate_set_id,
                RepairStrategy.REJECT,
                problems or {"no_ranked_candidate"},
                rerank_receipt,
                expiry,
            )
        else:
            authority = authority_map.get(selected.content_id)
            target_problems = self._target_problems(
                selected, authority, candidate_set_id
            )
            if target_problems:
                decision = self._abstention(
                    rows,
                    candidate_set_id,
                    RepairStrategy.REJECT,
                    target_problems,
                    rerank_receipt,
                    expiry,
                )
            else:
                assert authority is not None
                proof_refs = self._proof_refs(selected, rerank_receipt)
                evidence = self._evidence_refs(
                    selected, authority, rerank_receipt, expiry
                )
                decision = RepairTargetDecision(
                    roots=selected.roots,
                    candidates=rows,
                    candidate_set_id=candidate_set_id,
                    disposition=DecisionDisposition.ADMITTED,
                    strategy=selected.strategy,
                    selected_candidate_id=selected.content_id,
                    permitted_read_paths=authority.permitted_read_paths,
                    permitted_write_paths=authority.permitted_write_paths,
                    evidence_refs=evidence,
                    proof_refs=proof_refs,
                    invalidation_refs=self._invalidation_refs(
                        candidate_set_id, rerank_receipt, authority, expiry
                    ),
                )
                spans = authority.permitted_read_spans
                write_spans = authority.permitted_write_spans
                return self._result(
                    decision, rerank_receipt, expiry, spans, write_spans
                )
        return self._result(decision, rerank_receipt, expiry, (), ())

    assess = decide
    evaluate = decide
    admit_target = decide

    @staticmethod
    def _candidates(
        candidates: Sequence[RepairCandidate],
    ) -> tuple[RepairCandidate, ...]:
        if (
            isinstance(candidates, (str, bytes, bytearray))
            or not isinstance(candidates, Sequence)
            or not candidates
        ):
            raise RepairTargetAdmissionError(
                "candidates must be a non-empty RepairCandidate sequence"
            )
        rows = tuple(sorted(candidates, key=lambda item: item.content_id))
        if not all(isinstance(item, RepairCandidate) for item in rows):
            raise RepairTargetAdmissionError(
                "candidates must contain RepairCandidate values"
            )
        if len({item.content_id for item in rows}) != len(rows) or any(
            item.roots != rows[0].roots for item in rows
        ):
            raise RepairTargetAdmissionError(
                "candidates must be unique and bind exact common roots"
            )
        return rows

    @staticmethod
    def _receipt_problems(
        rows: tuple[RepairCandidate, ...], candidate_set_id: str, receipt: RerankReceipt
    ) -> set[str]:
        reasons: set[str] = set()
        if receipt.roots != rows[0].roots:
            reasons.add(AdmissionInvalidator.ROOT_CHANGED.value)
        if receipt.candidate_set_id != candidate_set_id:
            reasons.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION.value)
        rank_ids = {row.candidate_id for row in receipt.ranks}
        if rank_ids != {row.content_id for row in rows} or len(receipt.ranks) != len(
            rows
        ):
            reasons.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION.value)
        if receipt.disposition is RerankDisposition.RANKED:
            selected = next(
                (
                    rank
                    for rank in receipt.ranks
                    if rank.candidate_id == receipt.selected_candidate_id
                ),
                None,
            )
            if (
                selected is None
                or selected.disposition is not CandidateEligibilityDisposition.ELIGIBLE
            ):
                reasons.add("ranked_candidate_not_eligible")
        return reasons

    @staticmethod
    def _authorities(
        authorities: Sequence[TargetRepositoryAuthority],
        rows: tuple[RepairCandidate, ...],
        candidate_set_id: str,
    ) -> tuple[dict[str, TargetRepositoryAuthority], set[str]]:
        if isinstance(authorities, (str, bytes, bytearray)) or not isinstance(
            authorities, Sequence
        ):
            raise RepairTargetAdmissionError(
                "authorities must be a TargetRepositoryAuthority sequence"
            )
        result: dict[str, TargetRepositoryAuthority] = {}
        reasons: set[str] = set()
        candidates = {item.content_id: item for item in rows}
        for authority in authorities:
            if not isinstance(authority, TargetRepositoryAuthority):
                raise RepairTargetAdmissionError(
                    "authorities must contain TargetRepositoryAuthority values"
                )
            candidate = candidates.get(authority.candidate_id)
            if authority.candidate_id in result:
                reasons.add("duplicate_repository_authority")
                continue
            result[authority.candidate_id] = authority
            if candidate is None or authority.candidate_set_id != candidate_set_id:
                reasons.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION.value)
            elif authority.roots != candidate.roots:
                reasons.add(AdmissionInvalidator.ROOT_CHANGED.value)
        return result, reasons

    @staticmethod
    def _target_problems(
        candidate: RepairCandidate,
        authority: TargetRepositoryAuthority | None,
        candidate_set_id: str,
    ) -> set[str]:
        reasons: set[str] = set()
        if candidate.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}:
            reasons.add("selected_strategy_not_repairable")
        if candidate.rejection_reasons:
            reasons.add("selected_candidate_rejected")
        if authority is None:
            reasons.add("repository_authority_missing")
            return reasons
        if (
            authority.candidate_set_id != candidate_set_id
            or authority.candidate_id != candidate.content_id
        ):
            reasons.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION.value)
        if authority.roots != candidate.roots:
            reasons.add(AdmissionInvalidator.ROOT_CHANGED.value)
        if authority.target_span != candidate.target_span:
            reasons.add("selected_target_span_mismatch")
        # ``RepairTargetDecision@1`` retains a legacy subset check against
        # candidate_write_paths.  It cannot be allowed to choose the path:
        # require exact equality with repository authority before satisfying
        # that schema-level compatibility condition.
        if candidate.candidate_write_paths != authority.permitted_write_paths:
            reasons.add("candidate_write_path_not_exactly_repository_derived")
        if not authority.target_exists:
            reasons.add(AdmissionInvalidator.TARGET_MISSING.value)
        if authority.read_only:
            reasons.add(AdmissionInvalidator.READ_ONLY_PATH.value)
        return reasons

    @staticmethod
    def _ambiguity_reasons(receipt: RerankReceipt) -> set[str]:
        reasons = set(receipt.reason_codes)
        if "rank_tie" in reasons:
            reasons.add(AdmissionInvalidator.RANK_TIE.value)
        if "insufficient_rank_margin" in reasons:
            reasons.add(AdmissionInvalidator.LOW_MARGIN.value)
        return reasons or {"rerank_ambiguous"}

    @staticmethod
    def _proof_refs(
        candidate: RepairCandidate, receipt: RerankReceipt
    ) -> tuple[EvidenceReference, ...]:
        rank = next(
            row for row in receipt.ranks if row.candidate_id == candidate.content_id
        )
        refs = [*candidate.proof_refs]
        refs.extend(
            EvidenceReference(
                "proof_receipt", value, producer_id=REPAIR_TARGET_ADMISSION_INTERFACE
            )
            for value in rank.proof_receipt_ids
        )
        if not refs:
            # A ranked receipt has already hard-gated proof reconstruction, but
            # retain the receipt identity as the compact proof provenance edge.
            refs.append(
                EvidenceReference(
                    "rerank_proof_receipt",
                    receipt.receipt_id,
                    producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
                )
            )
        return tuple(sorted(set(refs), key=lambda item: item.content_id))

    @staticmethod
    def _evidence_refs(
        candidate: RepairCandidate,
        authority: TargetRepositoryAuthority,
        receipt: RerankReceipt,
        expiry: DecisionExpiry,
    ) -> tuple[EvidenceReference, ...]:
        refs = [*candidate.evidence_refs, *authority.evidence_refs]
        refs.append(
            EvidenceReference(
                "rerank_receipt",
                receipt.receipt_id,
                producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
            )
        )
        refs.append(
            EvidenceReference(
                "decision_expiry",
                expiry.content_id,
                producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
            )
        )
        return tuple(sorted(set(refs), key=lambda item: item.content_id))

    @staticmethod
    def _invalidation_refs(
        candidate_set_id: str,
        receipt: RerankReceipt,
        authority: TargetRepositoryAuthority | None,
        expiry: DecisionExpiry,
    ) -> tuple[str, ...]:
        values = [candidate_set_id, receipt.receipt_id, expiry.content_id]
        if authority is not None:
            values.extend(
                (
                    authority.candidate_id,
                    authority.target_span.artifact_id,
                    *[ref.content_id for ref in authority.evidence_refs],
                )
            )
        return tuple(sorted(set(values)))

    def _abstention(
        self,
        rows: tuple[RepairCandidate, ...],
        candidate_set_id: str,
        strategy: RepairStrategy,
        reasons: set[str],
        receipt: RerankReceipt,
        expiry: DecisionExpiry,
    ) -> RepairTargetDecision:
        disposition = (
            DecisionDisposition.ABSTAINED
            if strategy is RepairStrategy.AMBIGUOUS
            else DecisionDisposition.REJECTED
        )
        evidence = [
            EvidenceReference(
                "rerank_receipt",
                receipt.receipt_id,
                producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
            ),
            EvidenceReference(
                "decision_expiry",
                expiry.content_id,
                producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
            ),
        ]
        evidence.extend(
            EvidenceReference(
                "admission_rejection",
                reason,
                producer_id=REPAIR_TARGET_ADMISSION_INTERFACE,
            )
            for reason in sorted(reasons)
        )
        return RepairTargetDecision(
            roots=rows[0].roots,
            candidates=rows,
            candidate_set_id=candidate_set_id,
            disposition=disposition,
            strategy=strategy,
            evidence_refs=tuple(evidence),
            invalidation_refs=self._invalidation_refs(
                candidate_set_id, receipt, None, expiry
            ),
        )

    @staticmethod
    def _result(
        decision: RepairTargetDecision,
        receipt: RerankReceipt,
        expiry: DecisionExpiry,
        reads: tuple[SourceSpan, ...],
        writes: tuple[SourceSpan, ...],
    ) -> AdmissionResult:
        audit = TargetAdmissionAudit(
            decision.roots,
            decision.candidate_set_id,
            receipt.receipt_id,
            receipt.ranks,
            expiry,
            decision.content_id,
        )
        return AdmissionResult(decision, audit, expiry, reads, writes)


class RepairTargetDecisionValidator:
    """Validate a decision immediately before it grants a repair packet scope."""

    def validate(
        self,
        result: AdmissionResult,
        *,
        roots: AuthorityRoots,
        candidates: Sequence[RepairCandidate],
        rerank_receipt: RerankReceipt,
        authorities: Sequence[TargetRepositoryAuthority],
        now: int,
    ) -> tuple[AdmissionInvalidator, ...]:
        if not isinstance(result, AdmissionResult) or not isinstance(
            roots, AuthorityRoots
        ):
            raise RepairTargetAdmissionError("result and roots must be typed contracts")
        invalid: set[AdmissionInvalidator] = set()
        decision = result.decision
        if roots != decision.roots or roots != result.audit.roots:
            invalid.add(AdmissionInvalidator.ROOT_CHANGED)
        if not result.expiry.valid_at(now):
            invalid.add(AdmissionInvalidator.EXPIRED)
        try:
            rows = RepairTargetAdmission._candidates(candidates)
            candidate_set_id = candidate_set_identity(rows)
        except (RepairTargetAdmissionError, ValueError):
            invalid.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION)
            return tuple(sorted(invalid, key=lambda item: item.value))
        if (
            candidate_set_id != decision.candidate_set_id
            or tuple(item.content_id for item in rows) != decision.candidate_ids
        ):
            invalid.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION)
        if rerank_receipt.roots != roots:
            invalid.add(AdmissionInvalidator.ROOT_CHANGED)
        elif rerank_receipt.receipt_id != result.audit.rerank_receipt_id:
            invalid.add(AdmissionInvalidator.PROOF_DOWNGRADE)
        if {
            (
                row.candidate_id,
                row.disposition,
                row.score_vector,
                row.reason_codes,
                row.proof_receipt_ids,
            )
            for row in rerank_receipt.ranks
        } != {
            (
                row.candidate_id,
                row.disposition,
                row.score_vector,
                row.reason_codes,
                row.proof_receipt_ids,
            )
            for row in result.audit.ranks
        }:
            invalid.add(AdmissionInvalidator.PROOF_DOWNGRADE)
        if rerank_receipt.disposition is RerankDisposition.AMBIGUOUS:
            invalid.add(
                AdmissionInvalidator.RANK_TIE
                if "rank_tie" in rerank_receipt.reason_codes
                else AdmissionInvalidator.LOW_MARGIN
            )
        if decision.disposition is DecisionDisposition.ADMITTED:
            selected = next(
                (
                    item
                    for item in rows
                    if item.content_id == decision.selected_candidate_id
                ),
                None,
            )
            matching_authorities = tuple(
                item
                for item in authorities
                if isinstance(item, TargetRepositoryAuthority)
                and item.candidate_id == decision.selected_candidate_id
            )
            authority = (
                matching_authorities[0] if len(matching_authorities) == 1 else None
            )
            if (
                selected is None
                or authority is None
                or authority.target_span != selected.target_span
                or not authority.target_exists
            ):
                invalid.add(AdmissionInvalidator.TARGET_MISSING)
            elif authority.roots != roots:
                invalid.add(AdmissionInvalidator.ROOT_CHANGED)
            elif authority.candidate_set_id != candidate_set_id:
                invalid.add(AdmissionInvalidator.CANDIDATE_SET_MUTATION)
            elif (
                authority.read_only
                or authority.permitted_read_paths != decision.permitted_read_paths
                or authority.permitted_write_paths != decision.permitted_write_paths
                or selected.candidate_write_paths != authority.permitted_write_paths
            ):
                invalid.add(AdmissionInvalidator.READ_ONLY_PATH)
        return tuple(sorted(invalid, key=lambda item: item.value))

    def is_valid(self, *args: object, **kwargs: object) -> bool:
        return not self.validate(*args, **kwargs)  # type: ignore[arg-type]

    def require_valid(self, *args: object, **kwargs: object) -> AdmissionResult:
        result = args[0] if args else kwargs.get("result")
        invalid = self.validate(*args, **kwargs)  # type: ignore[arg-type]
        if invalid:
            raise RepairTargetAdmissionError(
                "repair target decision is invalid: "
                + ", ".join(item.value for item in invalid)
            )
        if not isinstance(result, AdmissionResult):
            raise RepairTargetAdmissionError("result must be AdmissionResult")
        return result


__all__ = [
    "REPAIR_TARGET_ADMISSION_AUDIT_SCHEMA",
    "REPAIR_TARGET_ADMISSION_INTERFACE",
    "AdmissionInvalidator",
    "AdmissionResult",
    "DecisionExpiry",
    "RepairTargetAdmission",
    "RepairTargetAdmissionError",
    "RepairTargetDecisionValidator",
    "TargetAdmissionAudit",
    "TargetRepositoryAuthority",
]
