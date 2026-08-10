"""DCR-074: deterministic merge and publication provenance.

This module deliberately *does not* invoke Git or mutate a checkout.  It
turns independently observed commit and pin facts into a typed publication
proposal.  A caller must still perform any eventual operator-approved merge.
Keeping this boundary evidence-only means a stale head can never be mistaken
for an implicit successful merge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final
import json

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)
from .contracts import (
    AuthorityStage,
    PublicationReceipt,
    RepairEvidenceEnvelope,
)
from .transaction import ADMITTED_OWNER_ROOTS


REPAIR_PUBLICATION_INTERFACE: Final[str] = "RepairPublication@1"
MERGE_PROVENANCE_INTERFACE: Final[str] = "MergeProvenance@1"
SUBMODULE_PIN_TRANSITION_INTERFACE: Final[str] = "SubmodulePinTransition@1"
DCR_MERGE_PROVENANCE_EVIDENCE: Final[str] = "dcr/merge-provenance@1"
DCR_MERGE_PROVENANCE_VERSION: Final[int] = 1

MERGE_PROVENANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-merge-provenance@1"
)
SUBMODULE_PIN_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-submodule-pin-transition@1"
)
REPAIR_PUBLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-repair-publication@1"
)
PUBLICATION_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-publication-catalog@1"
)
DEFAULT_PUBLICATION_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/publication.json"
)
MAX_PUBLICATION_ITEMS: Final[int] = 1_024
MAX_TEXT_BYTES: Final[int] = 4_096


class RepairPublicationError(ContractValidationError):
    """Publication evidence is incomplete, stale, or contradicts its roots."""


class PublicationDisposition(str, Enum):
    """Closed outcomes; only ``PUBLISHED`` advances the authority envelope."""

    PUBLISHED = "published"
    STALE = "stale"
    REPLAN = "replan"
    REJECTED = "rejected"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RepairPublicationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise RepairPublicationError(f"{name} is required")
    if any(char.isspace() for char in result):
        raise RepairPublicationError(f"{name} must be a compact identifier")
    if len(result.encode("utf-8")) > MAX_TEXT_BYTES:
        raise RepairPublicationError(f"{name} exceeds its byte bound")
    return result


def _root(value: Any, name: str = "owner_root") -> str:
    result = _identifier(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != result:
        raise RepairPublicationError(f"{name} must be a safe repository root")
    if result not in ADMITTED_OWNER_ROOTS:
        raise RepairPublicationError(f"{name} is not an admitted owner root")
    return result


def _ids(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairPublicationError(f"{name} must be an identifier sequence")
    result = tuple(_identifier(value, name) for value in values)
    if len(result) > MAX_PUBLICATION_ITEMS:
        raise RepairPublicationError(f"{name} exceeds its item bound")
    if len(set(result)) != len(result):
        raise RepairPublicationError(f"{name} must not contain duplicates")
    if required and not result:
        raise RepairPublicationError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class SubmodulePinTransition(CanonicalContract):
    """Observed predecessor/successor pin pair for one owner repository."""

    SCHEMA: ClassVar[str] = SUBMODULE_PIN_TRANSITION_SCHEMA
    INTERFACE: ClassVar[str] = SUBMODULE_PIN_TRANSITION_INTERFACE

    owner_root: str
    predecessor_pin: str
    successor_pin: str
    provider_commit_id: str
    pin_commit_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_root", _root(self.owner_root))
        for field_name in (
            "predecessor_pin",
            "successor_pin",
            "provider_commit_id",
            "pin_commit_id",
        ):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        if self.predecessor_pin == self.successor_pin:
            raise RepairPublicationError("submodule pin transition must change the pin")
        if self.provider_commit_id == self.pin_commit_id:
            raise RepairPublicationError("provider and pin commits must be distinct")

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "owner_root": self.owner_root,
            "predecessor_pin": self.predecessor_pin,
            "successor_pin": self.successor_pin,
            "provider_commit_id": self.provider_commit_id,
            "pin_commit_id": self.pin_commit_id,
        }


@dataclass(frozen=True)
class MergeProvenance(CanonicalContract):
    """A complete, evidence-only proposal for a single target-head merge."""

    SCHEMA: ClassVar[str] = MERGE_PROVENANCE_SCHEMA
    INTERFACE: ClassVar[str] = MERGE_PROVENANCE_INTERFACE

    repair_id: str
    target_ref: str
    expected_target_head: str
    observed_target_head: str
    predecessor_evidence_cid: str
    admission_receipt_cid: str
    mutation_receipt_cid: str
    post_edit_validation_receipt_cid: str
    reproof_receipt_cid: str
    validation_evidence_cid: str
    provider_commit_ids: tuple[str, ...]
    consumer_commit_ids: tuple[str, ...]
    pin_transitions: tuple[SubmodulePinTransition, ...] = ()
    disposition: PublicationDisposition = PublicationDisposition.REJECTED
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "repair_id", "target_ref", "expected_target_head", "observed_target_head",
            "predecessor_evidence_cid", "admission_receipt_cid", "mutation_receipt_cid",
            "post_edit_validation_receipt_cid", "reproof_receipt_cid", "validation_evidence_cid",
        ):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "provider_commit_ids", _ids(self.provider_commit_ids, "provider_commit_ids", required=True))
        object.__setattr__(self, "consumer_commit_ids", _ids(self.consumer_commit_ids, "consumer_commit_ids", required=True))
        if set(self.provider_commit_ids).intersection(self.consumer_commit_ids):
            raise RepairPublicationError("provider commits must precede distinct consumer commits")
        transitions = tuple(self.pin_transitions)
        if len(transitions) > MAX_PUBLICATION_ITEMS or not all(isinstance(item, SubmodulePinTransition) for item in transitions):
            raise RepairPublicationError("pin_transitions must contain SubmodulePinTransition")
        if len({item.owner_root for item in transitions}) != len(transitions):
            raise RepairPublicationError("pin_transitions must have unique owner roots")
        for item in transitions:
            if item.provider_commit_id not in self.provider_commit_ids:
                raise RepairPublicationError("pin transition provider commit is not proposed")
            if item.pin_commit_id not in self.consumer_commit_ids:
                raise RepairPublicationError("pin transition consumer commit is not proposed")
        object.__setattr__(self, "pin_transitions", transitions)
        if isinstance(self.disposition, str):
            try:
                object.__setattr__(self, "disposition", PublicationDisposition(self.disposition))
            except ValueError as exc:
                raise RepairPublicationError("invalid publication disposition") from exc
        if not isinstance(self.disposition, PublicationDisposition):
            raise RepairPublicationError("disposition must be PublicationDisposition")
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        if self.disposition is PublicationDisposition.PUBLISHED:
            if self.expected_target_head != self.observed_target_head:
                raise RepairPublicationError("published provenance requires an unchanged target head")
            if self.reason_codes:
                raise RepairPublicationError("published provenance cannot contain failure reasons")
        elif not self.reason_codes:
            raise RepairPublicationError("non-published provenance requires a reason code")

    @property
    def target_is_current(self) -> bool:
        return self.expected_target_head == self.observed_target_head

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "repair_id": self.repair_id,
            "target_ref": self.target_ref,
            "expected_target_head": self.expected_target_head,
            "observed_target_head": self.observed_target_head,
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "mutation_receipt_cid": self.mutation_receipt_cid,
            "post_edit_validation_receipt_cid": self.post_edit_validation_receipt_cid,
            "reproof_receipt_cid": self.reproof_receipt_cid,
            "validation_evidence_cid": self.validation_evidence_cid,
            "provider_commit_ids": list(self.provider_commit_ids),
            "consumer_commit_ids": list(self.consumer_commit_ids),
            "pin_transitions": [item.to_dict() for item in self.pin_transitions],
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RepairPublication(CanonicalContract):
    """The publication decision and, on success, its next authority envelope."""

    SCHEMA: ClassVar[str] = REPAIR_PUBLICATION_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_PUBLICATION_INTERFACE

    provenance: MergeProvenance
    publication_receipt: PublicationReceipt | None = None
    published_envelope: RepairEvidenceEnvelope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, MergeProvenance):
            raise RepairPublicationError("provenance must be MergeProvenance")
        if self.provenance.disposition is PublicationDisposition.PUBLISHED:
            if not isinstance(self.publication_receipt, PublicationReceipt):
                raise RepairPublicationError("published result requires a PublicationReceipt")
            if not isinstance(self.published_envelope, RepairEvidenceEnvelope):
                raise RepairPublicationError("published result requires an evidence envelope")
            if self.published_envelope.authority_stage is not AuthorityStage.PUBLISHED:
                raise RepairPublicationError("published envelope must have published authority stage")
            if self.published_envelope.publication_receipt != self.publication_receipt:
                raise RepairPublicationError("publication receipt must be bound into the envelope")
        elif self.publication_receipt is not None or self.published_envelope is not None:
            raise RepairPublicationError("non-published result must not mint publication evidence")

    @property
    def published(self) -> bool:
        return self.provenance.disposition is PublicationDisposition.PUBLISHED

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "provenance": self.provenance.to_dict(),
            "publication_receipt": self.publication_receipt.to_dict() if self.publication_receipt else None,
            "published_envelope": self.published_envelope.to_dict() if self.published_envelope else None,
        }


class RepairPublisher:
    """Build a fail-closed publication proposal from a re-proved envelope.

    The publisher accepts observations as arguments rather than looking at a
    repository.  In particular it cannot race a target ref by executing Git
    itself; a head mismatch deterministically returns ``REPLAN``.
    """

    INTERFACE: ClassVar[str] = REPAIR_PUBLICATION_INTERFACE

    def publish(
        self,
        envelope: RepairEvidenceEnvelope,
        *,
        target_ref: str,
        expected_target_head: str,
        observed_target_head: str,
        validation_evidence_cid: str,
        provider_commit_ids: Sequence[str],
        consumer_commit_ids: Sequence[str],
        pin_transitions: Sequence[SubmodulePinTransition] = (),
    ) -> RepairPublication:
        if not isinstance(envelope, RepairEvidenceEnvelope):
            raise RepairPublicationError("envelope must be RepairEvidenceEnvelope")
        if envelope.authority_stage is not AuthorityStage.REPROVED:
            raise RepairPublicationError("only a re-proved envelope may be published")
        if not envelope.reproof_receipt or not envelope.reproof_receipt.proved:
            raise RepairPublicationError("publication requires a successful typed reproof")
        if not envelope.post_edit_validation_receipt or not envelope.post_edit_validation_receipt.passed:
            raise RepairPublicationError("publication requires successful typed validation")
        if not envelope.admission_receipt:
            raise RepairPublicationError("publication requires typed admission evidence")

        expected = _identifier(expected_target_head, "expected_target_head")
        observed = _identifier(observed_target_head, "observed_target_head")
        validation_cid = _identifier(validation_evidence_cid, "validation_evidence_cid")
        if validation_cid != envelope.post_edit_validation_cid:
            raise RepairPublicationError(
                "publication validation evidence must match the current typed validation receipt"
            )
        common = dict(
            repair_id=envelope.repair_id,
            target_ref=target_ref,
            expected_target_head=expected,
            observed_target_head=observed,
            predecessor_evidence_cid=envelope.content_id,
            admission_receipt_cid=envelope.admission_cid,
            mutation_receipt_cid=envelope.mutation_receipt_cid,
            post_edit_validation_receipt_cid=envelope.post_edit_validation_cid,
            reproof_receipt_cid=envelope.reproof_cid,
            validation_evidence_cid=validation_cid,
            provider_commit_ids=tuple(provider_commit_ids),
            consumer_commit_ids=tuple(consumer_commit_ids),
            pin_transitions=tuple(pin_transitions),
        )
        if expected != observed:
            return RepairPublication(MergeProvenance(
                **common,
                disposition=PublicationDisposition.REPLAN,
                reason_codes=("target_head_changed",),
            ))

        provenance = MergeProvenance(**common, disposition=PublicationDisposition.PUBLISHED)
        receipt = PublicationReceipt(
            repair_id=envelope.repair_id,
            authority_roots=envelope.authority_roots,
            predecessor_evidence_cid=envelope.content_id,
            admission_receipt_cid=envelope.admission_cid,
            post_edit_validation_receipt_cid=envelope.post_edit_validation_cid,
            reproof_receipt_cid=envelope.reproof_cid,
            mutation_receipt_cid=envelope.mutation_receipt_cid,
            published=True,
        )
        published = RepairEvidenceEnvelope(
            repair_id=envelope.repair_id,
            disposition=envelope.disposition,
            authority_stage=AuthorityStage.PUBLISHED,
            authority_roots=envelope.authority_roots,
            observation_cid=envelope.observation_cid,
            previous_authority_stage=AuthorityStage.REPROVED,
            previous_envelope_cid=envelope.content_id,
            derivation_cid=envelope.derivation_cid,
            admission_cid=envelope.admission_cid,
            mutation_receipt_cid=envelope.mutation_receipt_cid,
            post_edit_validation_cid=envelope.post_edit_validation_cid,
            reproof_cid=envelope.reproof_cid,
            publication_cid=receipt.content_id,
            admission_receipt=envelope.admission_receipt,
            post_edit_validation_receipt=envelope.post_edit_validation_receipt,
            reproof_receipt=envelope.reproof_receipt,
            publication_receipt=receipt,
            implementation_disposition=envelope.implementation_disposition,
            producer_id=envelope.producer_id,
        )
        return RepairPublication(provenance, receipt, published)


def materialize_publication(
    publication: RepairPublication,
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write an explicit publication event; never infer success from a write."""

    if not isinstance(publication, RepairPublication):
        raise RepairPublicationError("publication must be RepairPublication")
    payload = {
        "schema": PUBLICATION_CATALOG_SCHEMA,
        "interface": REPAIR_PUBLICATION_INTERFACE,
        "evidence_id": DCR_MERGE_PROVENANCE_EVIDENCE,
        "version": DCR_MERGE_PROVENANCE_VERSION,
        "publication": publication.to_dict(),
        "publication_cid": publication.content_id,
        "published": publication.published,
        "runtime_model_calls": 0,
        "runtime_provider_calls": 0,
        "performs_git_operations": False,
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = Path(destination) if destination is not None else base.joinpath(*PurePosixPath(DEFAULT_PUBLICATION_PATH).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_MERGE_PROVENANCE_EVIDENCE",
    "DCR_MERGE_PROVENANCE_VERSION",
    "DEFAULT_PUBLICATION_PATH",
    "MERGE_PROVENANCE_INTERFACE",
    "PublicationDisposition",
    "REPAIR_PUBLICATION_INTERFACE",
    "SUBMODULE_PIN_TRANSITION_INTERFACE",
    "MergeProvenance",
    "RepairPublication",
    "RepairPublicationError",
    "RepairPublisher",
    "SubmodulePinTransition",
    "materialize_publication",
]
