"""Fail-closed admission of existing and new implementation sites.

Candidate retrieval intentionally has no mutation authority, and proof
reconstruction intentionally says nothing about which repository location may
be edited.  This module is the narrow join between those two boundaries.  It
does not inspect source text or invent architecture facts: the caller supplies
compact, authority-bound site facts and the already reconstructed obligations.

An admitted :class:`PlacementDecision` identifies a site for the next ranking
gate, but deliberately grants *no* write path.  ``RepairTargetDecision`` is
the sole contract which can grant path authority later in the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    candidate_set_identity,
)
from ..proof.contract_repair_obligations import (
    ObligationKind,
    PlacementObligation,
    ProofObligation,
)
from ..proof.contract_repair_prover import CandidateProofBundle
from ..proof.formal_verification_contracts import ProofReceipt


IMPLEMENTATION_SITE_ADMISSIBILITY_INTERFACE: Final[str] = (
    "ImplementationSiteAdmissibility@1"
)
PLACEMENT_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/placement-decision@1"
)


class ImplementationSiteAdmissibilityError(ValueError):
    """A placement input is malformed or binds incompatible authority roots."""


class PlacementDisposition(str, Enum):
    """The only outcomes of implementation-site admission."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    AMBIGUOUS = "ambiguous"


_PLACEMENT_KINDS: Final[frozenset[ObligationKind]] = frozenset(
    {
        ObligationKind.PLACEMENT_OWNERSHIP,
        ObligationKind.PLACEMENT_NO_OMITTED_COMPATIBLE_IMPLEMENTATION,
        ObligationKind.PLACEMENT_DEPENDENCY_DAG,
        ObligationKind.PLACEMENT_VISIBILITY_REGISTRATION,
        ObligationKind.PLACEMENT_EXACT_STUB_CONTRACT,
    }
)
_SUPPORT_KINDS: Final[frozenset[ObligationKind]] = frozenset(
    {
        ObligationKind.EFFECT_COMPATIBILITY,
        ObligationKind.CAPABILITY_COMPATIBILITY,
        ObligationKind.MEMORY_COMPATIBILITY,
    }
)
_REQUIRED_KINDS: Final[frozenset[ObligationKind]] = _PLACEMENT_KINDS | _SUPPORT_KINDS
_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "archive",
        "archives",
        "build",
        "dist",
        "generated",
        "node_modules",
        "third_party",
        "vendor",
        "vendors",
    }
)
_RETRIEVAL_REJECTION_CODES: Final[frozenset[str]] = frozenset(
    {
        "forbidden_layer",
        "generated_vendor_archive_target",
        "read_only_target",
        "stale_or_cross_tree",
    }
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImplementationSiteAdmissibilityError(f"{field_name} is required")
    return value.strip()


def _path(value: object, field_name: str) -> str:
    raw = _identifier(value, field_name).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise ImplementationSiteAdmissibilityError(
            f"{field_name} must be a normalized repository-relative path"
        )
    return raw


def _refs(values: Sequence[EvidenceReference], field_name: str) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ImplementationSiteAdmissibilityError(f"{field_name} must be a sequence")
    if not values or not all(isinstance(item, EvidenceReference) for item in values):
        raise ImplementationSiteAdmissibilityError(
            f"{field_name} must contain at least one EvidenceReference"
        )
    return tuple(sorted(set(values), key=lambda item: item.content_id))


@dataclass(frozen=True)
class RepositoryAuthority:
    """Exact, reviewed facts about one prospective implementation site.

    The boolean facts are not proof substitutes.  They make the site and its
    policy constraints explicit; the corresponding reconstructed obligations
    are still required by :class:`ImplementationSiteAdmissibility`.
    """

    roots: AuthorityRoots
    candidate_set_id: str
    target_path: str
    target_interface_id: str
    owner_id: str
    declaration_or_architecture_anchor_id: str
    sender_requirement_id: str
    generated_stub_contract_id: str
    evidence_refs: tuple[EvidenceReference, ...]
    ownership_exact: bool = False
    owner_unambiguous: bool = False
    write_authorized: bool = False
    external_read_only: bool = False
    generated: bool = False
    vendor: bool = False
    archive: bool = False
    forbidden_layer: bool = False
    dependency_cycle: bool = False
    visibility_route_satisfiable: bool = False
    export_route_satisfiable: bool = False
    registration_route_satisfiable: bool = False
    required_effects_supported: bool = False
    required_capabilities_supported: bool = False
    memory_policy_supported: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ImplementationSiteAdmissibilityError("roots must be AuthorityRoots")
        for name in (
            "candidate_set_id",
            "target_interface_id",
            "owner_id",
            "declaration_or_architecture_anchor_id",
            "sender_requirement_id",
            "generated_stub_contract_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs"))
        for name in (
            "write_authorized",
            "ownership_exact",
            "owner_unambiguous",
            "external_read_only",
            "generated",
            "vendor",
            "archive",
            "forbidden_layer",
            "dependency_cycle",
            "visibility_route_satisfiable",
            "export_route_satisfiable",
            "registration_route_satisfiable",
            "required_effects_supported",
            "required_capabilities_supported",
            "memory_policy_supported",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ImplementationSiteAdmissibilityError(f"{name} must be boolean")


@dataclass(frozen=True)
class PlacementProposal:
    """One candidate's complete placement input, without write authority."""

    candidate: RepairCandidate
    placement_obligation: PlacementObligation
    supporting_obligations: tuple[ProofObligation, ...]
    proof_bundle: CandidateProofBundle
    repository_authority: RepositoryAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RepairCandidate):
            raise ImplementationSiteAdmissibilityError("candidate must be RepairCandidate")
        if not isinstance(self.placement_obligation, PlacementObligation):
            raise ImplementationSiteAdmissibilityError("placement_obligation must be PlacementObligation")
        if not isinstance(self.placement_obligation.obligations, Sequence) or not all(
            isinstance(item, ProofObligation) for item in self.placement_obligation.obligations
        ):
            raise ImplementationSiteAdmissibilityError(
                "placement_obligation must contain ProofObligation values"
            )
        if not isinstance(self.proof_bundle, CandidateProofBundle):
            raise ImplementationSiteAdmissibilityError("proof_bundle must be CandidateProofBundle")
        if not isinstance(self.repository_authority, RepositoryAuthority):
            raise ImplementationSiteAdmissibilityError("repository_authority must be RepositoryAuthority")
        if not isinstance(self.supporting_obligations, Sequence) or isinstance(
            self.supporting_obligations, (str, bytes, bytearray)
        ) or not all(isinstance(item, ProofObligation) for item in self.supporting_obligations):
            raise ImplementationSiteAdmissibilityError("supporting_obligations must contain ProofObligation values")
        object.__setattr__(self, "supporting_obligations", tuple(self.supporting_obligations))

    @property
    def obligations(self) -> tuple[ProofObligation, ...]:
        """All required claims, with duplicate kinds retained for rejection."""

        return tuple(self.placement_obligation.obligations) + self.supporting_obligations


@dataclass(frozen=True)
class PlacementDecision:
    """A deterministic site decision; it never grants mutation authority."""

    disposition: PlacementDisposition
    candidate_set_id: str
    selected_candidate_id: str = ""
    target_path: str = ""
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    schema: str = PLACEMENT_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "disposition", PlacementDisposition(self.disposition))
        object.__setattr__(self, "candidate_set_id", _identifier(self.candidate_set_id, "candidate_set_id"))
        if self.schema != PLACEMENT_DECISION_SCHEMA:
            raise ImplementationSiteAdmissibilityError("unsupported placement decision schema")
        for name in ("selected_candidate_id", "target_path"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ImplementationSiteAdmissibilityError(f"{name} must be a string")
            object.__setattr__(self, name, value.strip())
        if self.target_path:
            object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        reasons = tuple(sorted({_identifier(value, "reason_code") for value in self.reason_codes}))
        object.__setattr__(self, "reason_codes", reasons)
        if self.evidence_refs:
            object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs"))
        else:
            object.__setattr__(self, "evidence_refs", ())
        receipts = tuple(sorted({_identifier(value, "proof_receipt_id") for value in self.proof_receipt_ids}))
        object.__setattr__(self, "proof_receipt_ids", receipts)
        if self.disposition is PlacementDisposition.ADMITTED:
            if not self.selected_candidate_id or not self.target_path or not receipts:
                raise ImplementationSiteAdmissibilityError("admission requires candidate, target path, and proof receipts")
        elif self.selected_candidate_id or self.target_path or receipts:
            raise ImplementationSiteAdmissibilityError("abstention cannot select a target or carry proof authority")

    @property
    def admitted(self) -> bool:
        return self.disposition is PlacementDisposition.ADMITTED

    @property
    def write_paths(self) -> tuple[str, ...]:
        """No path authority exists at the placement stage."""

        return ()

    @property
    def permitted_write_paths(self) -> tuple[str, ...]:
        """Compatibility spelling for policy consumers."""

        return ()


class ImplementationSiteAdmissibility:
    """Admit exactly one fully proved, architecture-valid implementation site."""

    def decide(
        self,
        proposals: Sequence[PlacementProposal],
        *,
        candidates: Sequence[RepairCandidate],
    ) -> PlacementDecision:
        """Evaluate the complete nominated set and abstain on any uncertainty."""

        try:
            candidate_rows = self._candidates(candidates)
            candidate_set_id = candidate_set_identity(candidate_rows)
            proposal_rows = self._proposals(proposals)
        except (ImplementationSiteAdmissibilityError, ValueError):
            return PlacementDecision(
                PlacementDisposition.ABSTAINED,
                "invalid:candidate-set",
                reason_codes=("invalid_admission_input",),
            )

        seen = set()
        admissible: list[PlacementProposal] = []
        rejection_codes: set[str] = set()
        duplicate_proposal = False
        for proposal in proposal_rows:
            candidate_id = proposal.candidate.content_id
            if candidate_id in seen:
                rejection_codes.add("duplicate_placement_proposal")
                duplicate_proposal = True
                continue
            seen.add(candidate_id)
            reasons = self._rejection_reasons(proposal, candidate_rows, candidate_set_id)
            if reasons:
                rejection_codes.update(reasons)
            else:
                admissible.append(proposal)

        if duplicate_proposal:
            return PlacementDecision(
                PlacementDisposition.ABSTAINED,
                candidate_set_id,
                reason_codes=("duplicate_placement_proposal",),
            )
        if len(admissible) == 1:
            return self._admitted_decision(admissible[0], candidate_set_id)
        if len(admissible) > 1:
            return PlacementDecision(
                PlacementDisposition.AMBIGUOUS,
                candidate_set_id,
                reason_codes=("multiple_equal_admissible_sites",),
            )
        return PlacementDecision(
            PlacementDisposition.ABSTAINED,
            candidate_set_id,
            reason_codes=tuple(rejection_codes or {"no_admissible_implementation_site"}),
        )

    assess = decide
    evaluate = decide
    admit = decide

    @staticmethod
    def _candidates(candidates: Sequence[RepairCandidate]) -> tuple[RepairCandidate, ...]:
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
            raise ImplementationSiteAdmissibilityError("candidates must be a sequence")
        result = tuple(candidates)
        if not result or not all(isinstance(item, RepairCandidate) for item in result):
            raise ImplementationSiteAdmissibilityError("candidates must contain RepairCandidate values")
        if any(item.roots != result[0].roots for item in result):
            raise ImplementationSiteAdmissibilityError(
                "candidates must bind one exact authority-root set"
            )
        return result

    @staticmethod
    def _proposals(proposals: Sequence[PlacementProposal]) -> tuple[PlacementProposal, ...]:
        if isinstance(proposals, (str, bytes, bytearray)) or not isinstance(proposals, Sequence):
            raise ImplementationSiteAdmissibilityError("proposals must be a sequence")
        result = tuple(proposals)
        if not all(isinstance(item, PlacementProposal) for item in result):
            raise ImplementationSiteAdmissibilityError("proposals must contain PlacementProposal values")
        return result

    def _rejection_reasons(
        self,
        proposal: PlacementProposal,
        candidates: tuple[RepairCandidate, ...],
        candidate_set_id: str,
    ) -> tuple[str, ...]:
        candidate = proposal.candidate
        authority = proposal.repository_authority
        reasons: set[str] = set()
        if candidate not in candidates:
            reasons.add("candidate_not_in_complete_candidate_set")
        if candidate.strategy not in {
            RepairStrategy.IMPLEMENT_EXISTING_DECLARATION,
            RepairStrategy.NEW_IMPLEMENTATION,
        }:
            reasons.add("candidate_is_not_an_implementation_site")
        if candidate.candidate_write_paths:
            reasons.add("candidate_attempted_to_grant_write_authority")
        if candidate.rejection_reasons:
            reasons.add("candidate_has_retrieval_rejection")
            reasons.update(set(candidate.rejection_reasons) & _RETRIEVAL_REJECTION_CODES)
        if candidate.roots != authority.roots:
            reasons.add("authority_roots_mismatch")
        if authority.candidate_set_id != candidate_set_id:
            reasons.add("candidate_set_binding_mismatch")
        if candidate.target_span.path != authority.target_path:
            reasons.add("target_path_mismatch")
        if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in PurePosixPath(authority.target_path).parts):
            reasons.add("generated_vendor_archive_target")
        if not authority.write_authorized:
            reasons.add("write_authority_not_exact")
        if not authority.ownership_exact:
            reasons.add("target_ownership_not_exact")
        if not authority.owner_unambiguous:
            reasons.add("ambiguous_owner")
        if authority.external_read_only:
            reasons.add("external_read_only_target")
        if authority.generated or authority.vendor or authority.archive:
            reasons.add("generated_vendor_archive_target")
        if authority.forbidden_layer:
            reasons.add("forbidden_dependency_layer")
        if authority.dependency_cycle:
            reasons.add("dependency_cycle")
        if not authority.visibility_route_satisfiable:
            reasons.add("visibility_route_unsatisfied")
        if not authority.export_route_satisfiable:
            reasons.add("export_route_unsatisfied")
        if not authority.registration_route_satisfiable:
            reasons.add("registration_route_unsatisfied")
        if not authority.required_effects_supported:
            reasons.add("required_effects_unsupported")
        if not authority.required_capabilities_supported:
            reasons.add("required_capabilities_unsupported")
        if not authority.memory_policy_supported:
            reasons.add("memory_policy_unsupported")
        if authority.sender_requirement_id != authority.generated_stub_contract_id:
            reasons.add("stub_contract_mismatch")
        reasons.update(self._proof_rejection_reasons(proposal))
        return tuple(sorted(reasons))

    @staticmethod
    def _proof_rejection_reasons(proposal: PlacementProposal) -> set[str]:
        candidate = proposal.candidate
        roots = candidate.roots
        bundle = proposal.proof_bundle
        obligations = proposal.obligations
        reasons: set[str] = set()
        if bundle.candidate_id != candidate.content_id:
            reasons.add("proof_candidate_binding_mismatch")
        if bundle.repository_id != roots.repository_id or bundle.tree_id != roots.tree_id:
            reasons.add("proof_tree_binding_mismatch")
        if not bundle.candidate_authoritative:
            reasons.add("placement_proof_not_authoritative")
        by_kind: dict[ObligationKind, ProofObligation] = {}
        for obligation in obligations:
            if obligation.candidate_id != candidate.content_id:
                reasons.add("obligation_candidate_binding_mismatch")
                continue
            claim = obligation.claim
            if (
                claim.repository_id != roots.repository_id
                or claim.tree_id != roots.tree_id
                or claim.translator_id != roots.translator_id
                or claim.toolchain_id != roots.toolchain_id
                or claim.policy_id != roots.policy_id
            ):
                reasons.add("obligation_authority_binding_mismatch")
            if obligation.kind in by_kind:
                reasons.add("duplicate_required_obligation")
            by_kind[obligation.kind] = obligation
        missing = _REQUIRED_KINDS.difference(by_kind)
        if missing:
            reasons.add("missing_required_placement_obligation")
        result_by_id = {result.obligation_id: result for result in bundle.results}
        for kind in _REQUIRED_KINDS.intersection(by_kind):
            obligation = by_kind[kind]
            result = result_by_id.get(obligation.obligation_id)
            if result is None:
                reasons.add("missing_required_proof_receipt")
                continue
            if not result.authoritative or not isinstance(result.receipt, ProofReceipt):
                reasons.add("required_proof_not_reconstructed")
                continue
            receipt = result.receipt
            if (
                receipt.obligation_id != obligation.code_obligation.obligation_id
                or receipt.repository_id != roots.repository_id
                or receipt.repository_tree_id != roots.tree_id
                or receipt.translator_id != roots.translator_id
                or receipt.toolchain_id != roots.toolchain_id
                or receipt.policy_id != roots.policy_id
            ):
                reasons.add("proof_receipt_binding_mismatch")
        return reasons

    @staticmethod
    def _admitted_decision(proposal: PlacementProposal, candidate_set_id: str) -> PlacementDecision:
        required_ids = {
            obligation.obligation_id
            for obligation in proposal.obligations
            if obligation.kind in _REQUIRED_KINDS
        }
        receipts = tuple(
            sorted(
                result.receipt.receipt_id
                for result in proposal.proof_bundle.results
                if result.obligation_id in required_ids
            )
        )
        evidence = tuple(
            sorted(
                {
                    *proposal.candidate.evidence_refs,
                    *proposal.repository_authority.evidence_refs,
                },
                key=lambda item: item.content_id,
            )
        )
        return PlacementDecision(
            PlacementDisposition.ADMITTED,
            candidate_set_id,
            selected_candidate_id=proposal.candidate.content_id,
            target_path=proposal.repository_authority.target_path,
            evidence_refs=evidence,
            proof_receipt_ids=receipts,
        )


__all__ = [
    "IMPLEMENTATION_SITE_ADMISSIBILITY_INTERFACE",
    "PLACEMENT_DECISION_SCHEMA",
    "ImplementationSiteAdmissibility",
    "ImplementationSiteAdmissibilityError",
    "PlacementDecision",
    "PlacementDisposition",
    "PlacementProposal",
    "RepositoryAuthority",
]
