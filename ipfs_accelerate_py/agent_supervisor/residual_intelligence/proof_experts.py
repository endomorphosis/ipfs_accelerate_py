"""Candidate rankings for proofs and tactics.

The actual prover remains authoritative.  Residual experts may rank premises,
lemmas, tactics, branches, and counterexample classes, but they never emit a
proof-acceptance field or public proof-witness body.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    canonical_id,
    reject_candidate_authority,
    required_text,
    strict_fields,
    text_tuple,
)
from .local_experts import IndependentValidationReceipt

PROOF_EXPERT_ADAPTER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-proof-expert-adapter@1"
)
PREMISE_RANKING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-premise-ranking@1"
)
TACTIC_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-tactic-candidate@1"
)
PROOF_CANDIDATE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-proof-candidate-receipt@1"
)
REASON_STALE_OBLIGATION: Final = "stale_obligation"
REASON_PROVER_UNAVAILABLE: Final = "prover_environment_unavailable"
REASON_PROOF_OMISSION: Final = "proof_omission_rejected"
REASON_SUGGESTION_NOT_PROOF: Final = "suggestion_is_not_proof"
REASON_WITNESS_NOT_PUBLIC: Final = "proof_witness_not_public"
MAX_RANKED: Final = 32

ProverCheck = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _ranked_ids(values: Any, name: str) -> tuple[str, ...]:
    items = text_tuple(values, name, allow_empty=True, max_items=MAX_RANKED)
    if len(set(items)) != len(items):
        raise ResidualIntelligenceError(f"{name} contains duplicate identities")
    return items


@dataclass(frozen=True)
class PremiseRanking:
    obligation_id: str
    source_cid: str
    environment_id: str
    premise_ids: tuple[str, ...]
    lemma_ids: tuple[str, ...] = ()
    branch_ids: tuple[str, ...] = ()
    counterexample_class: str = ""
    candidate_only: bool = True
    schema: str = PREMISE_RANKING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREMISE_RANKING_SCHEMA:
            raise ResidualIntelligenceError("unsupported premise ranking schema")
        object.__setattr__(self, "obligation_id", required_text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "source_cid", required_text(self.source_cid, "source_cid"))
        object.__setattr__(
            self, "environment_id", required_text(self.environment_id, "environment_id")
        )
        object.__setattr__(self, "premise_ids", _ranked_ids(self.premise_ids, "premise_ids"))
        object.__setattr__(self, "lemma_ids", _ranked_ids(self.lemma_ids, "lemma_ids"))
        object.__setattr__(self, "branch_ids", _ranked_ids(self.branch_ids, "branch_ids"))
        object.__setattr__(
            self,
            "counterexample_class",
            ""
            if self.counterexample_class in (None, "")
            else required_text(self.counterexample_class, "counterexample_class"),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("premise rankings must remain candidate_only")
        reject_candidate_authority(self.to_dict(include_id=False))

    @property
    def ranking_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "obligation_id": self.obligation_id,
            "source_cid": self.source_cid,
            "environment_id": self.environment_id,
            "premise_ids": self.premise_ids,
            "lemma_ids": self.lemma_ids,
            "branch_ids": self.branch_ids,
            "counterexample_class": self.counterexample_class,
            "candidate_only": True,
        }
        if include_id:
            payload["ranking_id"] = self.ranking_id
        return payload


@dataclass(frozen=True)
class TacticCandidate:
    obligation_id: str
    tactic_id: str
    failed: bool = False
    lineage_id: str = ""
    candidate_only: bool = True
    schema: str = TACTIC_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TACTIC_CANDIDATE_SCHEMA:
            raise ResidualIntelligenceError("unsupported tactic candidate schema")
        object.__setattr__(self, "obligation_id", required_text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "tactic_id", required_text(self.tactic_id, "tactic_id"))
        object.__setattr__(self, "failed", _require_bool(self.failed, "failed"))
        object.__setattr__(
            self,
            "lineage_id",
            "" if self.lineage_id in (None, "") else required_text(self.lineage_id, "lineage_id"),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("tactic candidates must remain candidate_only")
        reject_candidate_authority(self.to_dict(include_id=False))

    @property
    def candidate_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "obligation_id": self.obligation_id,
            "tactic_id": self.tactic_id,
            "failed": self.failed,
            "lineage_id": self.lineage_id,
            "candidate_only": True,
        }
        if include_id:
            payload["candidate_id"] = self.candidate_id
        return payload


@dataclass(frozen=True)
class ProofCandidateReceipt:
    obligation_id: str
    source_cid: str
    environment_id: str
    ranking: PremiseRanking
    tactics: tuple[TacticCandidate, ...]
    prover_checked: bool
    prover_accepted: bool
    disposition: ExpertDisposition
    reason_codes: tuple[str, ...]
    privacy_class: PrivacyClass = PrivacyClass.PROOF_WITNESS
    candidate_only: bool = True
    schema: str = PROOF_CANDIDATE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROOF_CANDIDATE_RECEIPT_SCHEMA:
            raise ResidualIntelligenceError("unsupported proof candidate receipt schema")
        object.__setattr__(self, "obligation_id", required_text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "source_cid", required_text(self.source_cid, "source_cid"))
        object.__setattr__(
            self, "environment_id", required_text(self.environment_id, "environment_id")
        )
        if not isinstance(self.ranking, PremiseRanking):
            raise ResidualIntelligenceError("receipt requires PremiseRanking")
        object.__setattr__(self, "tactics", tuple(self.tactics))
        object.__setattr__(self, "prover_checked", _require_bool(self.prover_checked, "prover_checked"))
        object.__setattr__(
            self, "prover_accepted", _require_bool(self.prover_accepted, "prover_accepted")
        )
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        object.__setattr__(
            self, "reason_codes", text_tuple(self.reason_codes, "reason_codes", max_items=32)
        )
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        if self.privacy_class is not PrivacyClass.PROOF_WITNESS:
            raise ResidualIntelligenceError(REASON_WITNESS_NOT_PUBLIC)
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("proof receipts must remain candidate_only")
        if "proof_accepted" in self.reason_codes or "proof" == self.disposition.value.lower():
            raise ResidualIntelligenceError(REASON_SUGGESTION_NOT_PROOF)
        reject_candidate_authority(self.to_dict(include_id=False))

    @property
    def receipt_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "obligation_id": self.obligation_id,
            "source_cid": self.source_cid,
            "environment_id": self.environment_id,
            "ranking": self.ranking.to_dict(),
            "tactics": tuple(item.to_dict() for item in self.tactics),
            "prover_checked": self.prover_checked,
            "prover_accepted": self.prover_accepted,
            "disposition": self.disposition.value,
            "reason_codes": self.reason_codes,
            "privacy_class": self.privacy_class.value,
            "candidate_only": True,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class ProofExpertAdapter:
    """Nominate proof/tactic candidates; the prover decides."""

    current_obligation_id: str
    current_source_cid: str
    current_environment_id: str
    prover: ProverCheck | None = None
    family: ResidualTaskFamily = ResidualTaskFamily.TACTIC_SUGGESTION
    risk: RiskClass = RiskClass.R5
    schema: str = PROOF_EXPERT_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROOF_EXPERT_ADAPTER_SCHEMA:
            raise ResidualIntelligenceError("unsupported proof expert adapter schema")
        object.__setattr__(
            self,
            "current_obligation_id",
            required_text(self.current_obligation_id, "current_obligation_id"),
        )
        object.__setattr__(
            self, "current_source_cid", required_text(self.current_source_cid, "current_source_cid")
        )
        object.__setattr__(
            self,
            "current_environment_id",
            required_text(self.current_environment_id, "current_environment_id"),
        )
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "risk", RiskClass(self.risk))
        if self.risk is not RiskClass.R5:
            raise ResidualIntelligenceError("proof expert adapter is R5")

    def nominate(
        self,
        ranking: PremiseRanking,
        tactics: Sequence[TacticCandidate],
        *,
        validation: IndependentValidationReceipt | None = None,
    ) -> ProofCandidateReceipt:
        reasons: list[str] = [REASON_SUGGESTION_NOT_PROOF]
        if ranking.obligation_id != self.current_obligation_id:
            return self._blocked(ranking, tactics, (REASON_STALE_OBLIGATION,))
        if ranking.source_cid != self.current_source_cid:
            return self._blocked(ranking, tactics, (REASON_STALE_OBLIGATION,))
        if ranking.environment_id != self.current_environment_id:
            return self._blocked(ranking, tactics, (REASON_STALE_OBLIGATION,))
        if any(item.obligation_id != ranking.obligation_id for item in tactics):
            return self._blocked(ranking, tactics, (REASON_STALE_OBLIGATION,))
        if self.prover is None:
            return self._blocked(ranking, tactics, (REASON_PROVER_UNAVAILABLE,))
        checked = self.prover(
            {
                "obligation_id": ranking.obligation_id,
                "source_cid": ranking.source_cid,
                "environment_id": ranking.environment_id,
                "assumptions": ranking.premise_ids,
            }
        )
        if not isinstance(checked, Mapping) or checked.get("checked") is not True:
            return self._blocked(ranking, tactics, (REASON_PROOF_OMISSION,))
        accepted = checked.get("accepted") is True
        disposition = ExpertDisposition.VALIDATION_REQUIRED
        if validation is not None and validation.accepted is not True:
            disposition = ExpertDisposition.REJECT_INPUT
        elif validation is not None and accepted:
            disposition = ExpertDisposition.ACCEPT
        failed = tuple(
            TacticCandidate(
                obligation_id=item.obligation_id,
                tactic_id=item.tactic_id,
                failed=item.failed or not accepted,
                lineage_id=item.lineage_id or canonical_id(item.to_dict(include_id=False)),
            )
            for item in tactics
        )
        return ProofCandidateReceipt(
            obligation_id=ranking.obligation_id,
            source_cid=ranking.source_cid,
            environment_id=ranking.environment_id,
            ranking=ranking,
            tactics=failed,
            prover_checked=True,
            prover_accepted=accepted,
            disposition=disposition,
            reason_codes=tuple(reasons),
        )

    def _blocked(
        self,
        ranking: PremiseRanking,
        tactics: Sequence[TacticCandidate],
        reasons: Sequence[str],
    ) -> ProofCandidateReceipt:
        return ProofCandidateReceipt(
            obligation_id=ranking.obligation_id,
            source_cid=ranking.source_cid,
            environment_id=ranking.environment_id,
            ranking=ranking,
            tactics=tuple(tactics),
            prover_checked=False,
            prover_accepted=False,
            disposition=ExpertDisposition.REJECT_INPUT
            if REASON_STALE_OBLIGATION in reasons
            else ExpertDisposition.CAPABILITY_UNAVAILABLE,
            reason_codes=tuple(reasons) + (REASON_SUGGESTION_NOT_PROOF,),
        )
