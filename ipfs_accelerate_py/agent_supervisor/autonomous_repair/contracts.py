"""Contracts for domain-agnostic autonomous repair."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)

AUTONOMOUS_REPAIR_INTERFACE: Final = "AutonomousRepairEngine@1"
AUTONOMOUS_REPAIR_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-repair-report@1"
)
REPAIR_WORK_ITEM_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/autonomous-repair-work-item@1"
REPAIR_PLAN_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/autonomous-repair-plan@1"

# DCR-002 is intentionally a small, leaf evidence boundary.  It references
# existing RPR authority by CID rather than importing or duplicating RPR
# packet/plan records.
DETERMINISTIC_REPAIR_CONTRACT_VERSION: Final[int] = 1
DETERMINISTIC_REPAIR_INTERFACE: Final[str] = "DeterministicRepairContracts@1"
REPAIR_EVIDENCE_ENVELOPE_INTERFACE: Final[str] = "RepairEvidenceEnvelope@1"
REPAIR_AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/authority-roots@1"
)
REPAIR_EVIDENCE_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/evidence-envelope@1"
)
REPAIR_ADMISSION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/admission-receipt@1"
)
POST_EDIT_VALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/post-edit-validation-receipt@1"
)
REPROOF_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/reproof-receipt@1"
)
PUBLICATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/publication-receipt@1"
)
MAX_DETERMINISTIC_REPAIR_TEXT_BYTES: Final[int] = 4_096
MAX_DETERMINISTIC_REPAIR_RECORD_BYTES: Final[int] = 262_144


class RepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed disposition vocabulary for one work item."""

    SINGLE_PATH_READY = "single_path_ready"
    """Exactly one MCP surface anchor; deterministic transform may proceed."""

    MULTI_PATH_COLLAPSE = "multi_path_collapse"
    """Multiple anchors; prefer mediation/collapse rules, no silent rewrite."""

    MISSING_SURFACE = "missing_surface"
    """No register_tool surface found; needs registration or IDL alias."""

    IDL_GAP = "idl_gap"
    """GUI/ORB/IDL name not aligned with package MCP tools."""

    ANALYSIS_ONLY = "analysis_only"
    """IR/doctor analysis ready; code edit not admitted yet."""

    BLOCKED = "blocked"
    """Doctor abstention or IR failure; residual may need RPR."""


class AuthorityStage(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed internal DCR-002 authority lifecycle stages.

    ``OBSERVED`` and ``DERIVED`` are informational evidence only.  They have
    no write or completion authority.  A repair may write only after the
    separate RPR admission state, and may complete only after post-edit
    validation, reproof, and publication have each been recorded.
    """

    OBSERVED = "observed"
    DERIVED = "derived"
    ADMITTED = "admitted"
    MUTATED = "mutated"
    POST_EDIT_VALIDATED = "post_edit_validated"
    REPROVED = "reproved"
    PUBLISHED = "published"

    @property
    def authorizes_mutation(self) -> bool:
        """Only an admitted RPR-bound state can authorize an edit."""

        return self is AuthorityStage.ADMITTED


class DeterministicRepairDisposition(str, Enum):  # noqa: UP042 - Python 3.8
    """Closed public outcomes for deterministic repair evidence.

    This vocabulary intentionally is not a lifecycle.  In particular,
    ``repaired_pending_validation`` cannot claim completion, and ``completed``
    remains subject to the separate authority-stage chain.
    """

    PROVED_VALID = "proved_valid"
    REFUTED_REPAIRABLE = "refuted_repairable"
    REPAIRED_PENDING_VALIDATION = "repaired_pending_validation"
    ABSTAIN_REVIEW = "abstain_review"
    DEFER_CAPABILITY = "defer_capability"
    REJECTED = "rejected"
    COMPLETED = "completed"


class DeterministicRepairContractError(ContractValidationError):
    """Malformed DCR-002 evidence or an unsafe lifecycle transition."""


class DeterministicRepairAuthorityError(DeterministicRepairContractError):
    """Repair authority, roots, or lifecycle progression was not exact."""


class ForgedRepairEvidenceIdentityError(DeterministicRepairContractError):
    """A stored DCR-002 content identity disagreed with its canonical body."""


_FORWARD_REPAIR_TRANSITIONS: Final[Mapping[AuthorityStage, frozenset[AuthorityStage]]] = {
    AuthorityStage.OBSERVED: frozenset({AuthorityStage.DERIVED}),
    AuthorityStage.DERIVED: frozenset({AuthorityStage.ADMITTED}),
    AuthorityStage.ADMITTED: frozenset({AuthorityStage.MUTATED}),
    AuthorityStage.MUTATED: frozenset({AuthorityStage.POST_EDIT_VALIDATED}),
    AuthorityStage.POST_EDIT_VALIDATED: frozenset({AuthorityStage.REPROVED}),
    AuthorityStage.REPROVED: frozenset({AuthorityStage.PUBLISHED}),
    AuthorityStage.PUBLISHED: frozenset(),
}


def _dcr_text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise DeterministicRepairContractError(f"{field_name} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise DeterministicRepairContractError(f"{field_name} is required")
    if len(normalized.encode("utf-8")) > MAX_DETERMINISTIC_REPAIR_TEXT_BYTES:
        raise DeterministicRepairContractError(f"{field_name} exceeds its byte bound")
    return normalized


def _dcr_identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    result = _dcr_text(value, field_name, required=required)
    if result and any(char.isspace() for char in result):
        raise DeterministicRepairContractError(f"{field_name} must be an opaque compact identifier")
    return result


def _dcr_enum(value: Any, field_name: str) -> DeterministicRepairDisposition:
    # Explicitly do not coerce arbitrary Enum instances through ``str``:
    # accepting a synthetic enum with a matching representation weakens the
    # closed vocabulary at this authority boundary.
    if isinstance(value, DeterministicRepairDisposition):
        return value
    if isinstance(value, Enum) or type(value) is not str:
        raise DeterministicRepairContractError(
            f"{field_name} must be one of: "
            + ", ".join(item.value for item in DeterministicRepairDisposition)
        )
    try:
        return DeterministicRepairDisposition(value)
    except ValueError as exc:
        raise DeterministicRepairContractError(
            f"{field_name} must be one of: "
            + ", ".join(item.value for item in DeterministicRepairDisposition)
        ) from exc


def _authority_stage(value: Any, field_name: str) -> AuthorityStage:
    if isinstance(value, AuthorityStage):
        return value
    if isinstance(value, Enum) or type(value) is not str:
        raise DeterministicRepairContractError(
            f"{field_name} must be one of: " + ", ".join(item.value for item in AuthorityStage)
        )
    try:
        return AuthorityStage(value)
    except ValueError as exc:
        raise DeterministicRepairContractError(
            f"{field_name} must be one of: " + ", ".join(item.value for item in AuthorityStage)
        ) from exc


def parse_deterministic_repair_disposition(
    value: Any,
) -> DeterministicRepairDisposition:
    """Parse only a member of the DCR-002 closed lifecycle vocabulary."""

    return _dcr_enum(value, "disposition")


def closed_deterministic_repair_dispositions() -> frozenset[DeterministicRepairDisposition]:
    """Return the complete closed DCR-002 lifecycle vocabulary."""

    return frozenset(DeterministicRepairDisposition)


def parse_authority_stage(value: Any) -> AuthorityStage:
    """Parse only a member of the closed internal authority-stage vocabulary."""

    return _authority_stage(value, "authority_stage")


def assert_deterministic_repair_transition(
    previous: AuthorityStage | str | None,
    current: AuthorityStage | str,
) -> AuthorityStage:
    """Validate one lifecycle edge without inferring missing authority.

    The initial record must be ``OBSERVED``.  Failure dispositions may be
    entered from any non-terminal state, but no state can leave a terminal
    disposition.  In particular there is no observation/derivation shortcut
    to mutation or completion.
    """

    next_state = _authority_stage(current, "authority_stage")
    if previous is None:
        if next_state is not AuthorityStage.OBSERVED:
            raise DeterministicRepairAuthorityError(
                "the initial repair evidence disposition must be observed"
            )
        return next_state
    previous_state = _authority_stage(previous, "previous_authority_stage")
    if next_state not in _FORWARD_REPAIR_TRANSITIONS[previous_state]:
        raise DeterministicRepairAuthorityError(
            f"illegal deterministic repair transition: {previous_state.value} -> {next_state.value}"
        )
    return next_state


assert_authority_stage_transition = assert_deterministic_repair_transition


@dataclass(frozen=True)
class RepairAuthorityRoots(CanonicalContract):
    """Exact observation and RPR authority roots for one deterministic repair."""

    SCHEMA: ClassVar[str] = REPAIR_AUTHORITY_ROOTS_SCHEMA

    repository_id: str
    repository_forest_cid: str
    git_tree_id: str
    policy_root: str
    rpr_plan_cid: str
    rpr_packet_cid: str

    def __post_init__(self) -> None:
        for field_name in (
            "repository_id",
            "repository_forest_cid",
            "git_tree_id",
            "policy_root",
            "rpr_plan_cid",
            "rpr_packet_cid",
        ):
            object.__setattr__(
                self, field_name, _dcr_identifier(getattr(self, field_name), field_name)
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "repository_forest_cid": self.repository_forest_cid,
            "git_tree_id": self.git_tree_id,
            "policy_root": self.policy_root,
            "rpr_plan_cid": self.rpr_plan_cid,
            "rpr_packet_cid": self.rpr_packet_cid,
        }

    def matches(self, other: RepairAuthorityRoots) -> bool:
        return isinstance(other, RepairAuthorityRoots) and self == other

    def require_current(self, current: RepairAuthorityRoots) -> None:
        if not self.matches(current):
            raise DeterministicRepairAuthorityError(
                "repair authority roots are missing, stale, or do not match exactly"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairAuthorityRoots:
        fields = (
            "repository_id",
            "repository_forest_cid",
            "git_tree_id",
            "policy_root",
            "rpr_plan_cid",
            "rpr_packet_cid",
        )
        values = _dcr_decode(payload, cls.SCHEMA, fields, "repair authority roots")
        result = cls(**values)
        _dcr_verify_identity(payload, result)
        return result


def _dcr_decode(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, MappingABC) or payload.get("schema") != schema:
        raise DeterministicRepairContractError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, DETERMINISTIC_REPAIR_CONTRACT_VERSION):
        raise DeterministicRepairContractError(f"{name} has an unsupported contract version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise DeterministicRepairContractError(
            f"{name} contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    missing = [field_name for field_name in fields if field_name not in payload]
    if missing:
        raise DeterministicRepairContractError(
            f"{name} omits required fields: " + ", ".join(missing)
        )
    return {field_name: payload[field_name] for field_name in fields}


def _dcr_verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    for key in ("content_id", "cid"):
        if key not in payload or payload[key] in (None, ""):
            continue
        if not isinstance(payload[key], str) or payload[key] != record.content_id:
            raise ForgedRepairEvidenceIdentityError(
                "stored content identity does not match the canonical repair evidence"
            )


def repair_evidence_digest(value: Any) -> str:
    """Return the deterministic SHA-256 digest for DCR-002 canonical bytes."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def repair_evidence_cid(value: Any) -> str:
    """Return the deterministic CIDv1 identity for DCR-002 canonical data."""

    return content_identity(value)


def _receipt_roots(value: Any, field_name: str) -> RepairAuthorityRoots:
    if not isinstance(value, RepairAuthorityRoots):
        raise DeterministicRepairContractError(f"{field_name} must be RepairAuthorityRoots")
    return value


def _receipt_values(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    name: str,
) -> dict[str, Any]:
    values = _dcr_decode(payload, schema, fields, name)
    roots = values["authority_roots"]
    if not isinstance(roots, MappingABC):
        raise DeterministicRepairContractError(f"{name} authority_roots must be an object")
    values["authority_roots"] = RepairAuthorityRoots.from_dict(roots)
    return values


@dataclass(frozen=True)
class RepairAdmissionReceipt(CanonicalContract):
    """Canonical RPR admission bound to one exact predecessor and root set."""

    SCHEMA: ClassVar[str] = REPAIR_ADMISSION_RECEIPT_SCHEMA

    repair_id: str
    authority_roots: RepairAuthorityRoots
    predecessor_evidence_cid: str
    derivation_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_id", _dcr_identifier(self.repair_id, "repair_id"))
        object.__setattr__(
            self,
            "authority_roots",
            _receipt_roots(self.authority_roots, "authority_roots"),
        )
        for field_name in ("predecessor_evidence_cid", "derivation_cid"):
            object.__setattr__(
                self,
                field_name,
                _dcr_identifier(getattr(self, field_name), field_name),
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repair_id": self.repair_id,
            "authority_roots": self.authority_roots.to_dict(),
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "derivation_cid": self.derivation_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairAdmissionReceipt:
        fields = (
            "repair_id",
            "authority_roots",
            "predecessor_evidence_cid",
            "derivation_cid",
        )
        result = cls(**_receipt_values(payload, cls.SCHEMA, fields, "admission receipt"))
        _dcr_verify_identity(payload, result)
        return result


@dataclass(frozen=True)
class PostEditValidationReceipt(CanonicalContract):
    """Canonical successful post-edit validation bound to an admitted mutation."""

    SCHEMA: ClassVar[str] = POST_EDIT_VALIDATION_RECEIPT_SCHEMA

    repair_id: str
    authority_roots: RepairAuthorityRoots
    predecessor_evidence_cid: str
    admission_receipt_cid: str
    mutation_receipt_cid: str
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_id", _dcr_identifier(self.repair_id, "repair_id"))
        object.__setattr__(
            self,
            "authority_roots",
            _receipt_roots(self.authority_roots, "authority_roots"),
        )
        for field_name in (
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "mutation_receipt_cid",
        ):
            object.__setattr__(
                self,
                field_name,
                _dcr_identifier(getattr(self, field_name), field_name),
            )
        if not isinstance(self.passed, bool):
            raise DeterministicRepairContractError("passed must be boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repair_id": self.repair_id,
            "authority_roots": self.authority_roots.to_dict(),
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "mutation_receipt_cid": self.mutation_receipt_cid,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PostEditValidationReceipt:
        fields = (
            "repair_id",
            "authority_roots",
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "mutation_receipt_cid",
            "passed",
        )
        result = cls(**_receipt_values(payload, cls.SCHEMA, fields, "post-edit validation receipt"))
        _dcr_verify_identity(payload, result)
        return result


@dataclass(frozen=True)
class ReproofReceipt(CanonicalContract):
    """Canonical successful reproof linked to a specific validated mutation."""

    SCHEMA: ClassVar[str] = REPROOF_RECEIPT_SCHEMA

    repair_id: str
    authority_roots: RepairAuthorityRoots
    predecessor_evidence_cid: str
    admission_receipt_cid: str
    post_edit_validation_receipt_cid: str
    mutation_receipt_cid: str
    proved: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_id", _dcr_identifier(self.repair_id, "repair_id"))
        object.__setattr__(
            self,
            "authority_roots",
            _receipt_roots(self.authority_roots, "authority_roots"),
        )
        for field_name in (
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "post_edit_validation_receipt_cid",
            "mutation_receipt_cid",
        ):
            object.__setattr__(
                self,
                field_name,
                _dcr_identifier(getattr(self, field_name), field_name),
            )
        if not isinstance(self.proved, bool):
            raise DeterministicRepairContractError("proved must be boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repair_id": self.repair_id,
            "authority_roots": self.authority_roots.to_dict(),
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "post_edit_validation_receipt_cid": self.post_edit_validation_receipt_cid,
            "mutation_receipt_cid": self.mutation_receipt_cid,
            "proved": self.proved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReproofReceipt:
        fields = (
            "repair_id",
            "authority_roots",
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "post_edit_validation_receipt_cid",
            "mutation_receipt_cid",
            "proved",
        )
        result = cls(**_receipt_values(payload, cls.SCHEMA, fields, "reproof receipt"))
        _dcr_verify_identity(payload, result)
        return result


@dataclass(frozen=True)
class PublicationReceipt(CanonicalContract):
    """Canonical publication linked to the exact admitted, validated reproof."""

    SCHEMA: ClassVar[str] = PUBLICATION_RECEIPT_SCHEMA

    repair_id: str
    authority_roots: RepairAuthorityRoots
    predecessor_evidence_cid: str
    admission_receipt_cid: str
    post_edit_validation_receipt_cid: str
    reproof_receipt_cid: str
    mutation_receipt_cid: str
    published: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_id", _dcr_identifier(self.repair_id, "repair_id"))
        object.__setattr__(
            self,
            "authority_roots",
            _receipt_roots(self.authority_roots, "authority_roots"),
        )
        for field_name in (
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "post_edit_validation_receipt_cid",
            "reproof_receipt_cid",
            "mutation_receipt_cid",
        ):
            object.__setattr__(
                self,
                field_name,
                _dcr_identifier(getattr(self, field_name), field_name),
            )
        if not isinstance(self.published, bool):
            raise DeterministicRepairContractError("published must be boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repair_id": self.repair_id,
            "authority_roots": self.authority_roots.to_dict(),
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "post_edit_validation_receipt_cid": self.post_edit_validation_receipt_cid,
            "reproof_receipt_cid": self.reproof_receipt_cid,
            "mutation_receipt_cid": self.mutation_receipt_cid,
            "published": self.published,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PublicationReceipt:
        fields = (
            "repair_id",
            "authority_roots",
            "predecessor_evidence_cid",
            "admission_receipt_cid",
            "post_edit_validation_receipt_cid",
            "reproof_receipt_cid",
            "mutation_receipt_cid",
            "published",
        )
        result = cls(**_receipt_values(payload, cls.SCHEMA, fields, "publication receipt"))
        _dcr_verify_identity(payload, result)
        return result


@dataclass(frozen=True)
class RepairEvidenceEnvelope(CanonicalContract):
    """Canonical, append-only evidence for one DCR-002 lifecycle state.

    The envelope contains only opaque IDs.  The RPR packet/plan remains the
    source of edit authority; DCR-002 merely binds that authority to exact
    roots and makes all completion prerequisites explicit.
    """

    SCHEMA: ClassVar[str] = REPAIR_EVIDENCE_ENVELOPE_SCHEMA

    repair_id: str
    disposition: DeterministicRepairDisposition
    authority_stage: AuthorityStage
    authority_roots: RepairAuthorityRoots
    observation_cid: str
    previous_authority_stage: AuthorityStage | None = None
    previous_envelope_cid: str = ""
    derivation_cid: str = ""
    admission_cid: str = ""
    mutation_receipt_cid: str = ""
    post_edit_validation_cid: str = ""
    reproof_cid: str = ""
    publication_cid: str = ""
    admission_receipt: RepairAdmissionReceipt | None = None
    post_edit_validation_receipt: PostEditValidationReceipt | None = None
    reproof_receipt: ReproofReceipt | None = None
    publication_receipt: PublicationReceipt | None = None
    implementation_disposition: str = "closed_deterministic"
    producer_id: str = "deterministic-repair-contracts@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_id", _dcr_identifier(self.repair_id, "repair_id"))
        disposition = _dcr_enum(self.disposition, "disposition")
        stage = _authority_stage(self.authority_stage, "authority_stage")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "authority_stage", stage)
        if not isinstance(self.authority_roots, RepairAuthorityRoots):
            raise DeterministicRepairContractError("authority_roots must be RepairAuthorityRoots")
        object.__setattr__(
            self, "observation_cid", _dcr_identifier(self.observation_cid, "observation_cid")
        )
        previous = (
            None
            if self.previous_authority_stage is None
            else _authority_stage(self.previous_authority_stage, "previous_authority_stage")
        )
        object.__setattr__(self, "previous_authority_stage", previous)
        object.__setattr__(
            self,
            "previous_envelope_cid",
            _dcr_identifier(self.previous_envelope_cid, "previous_envelope_cid", required=False),
        )
        for field_name in (
            "derivation_cid",
            "admission_cid",
            "mutation_receipt_cid",
            "post_edit_validation_cid",
            "reproof_cid",
            "publication_cid",
        ):
            object.__setattr__(
                self,
                field_name,
                _dcr_identifier(getattr(self, field_name), field_name, required=False),
            )
        for field_name, receipt_type in (
            ("admission_receipt", RepairAdmissionReceipt),
            ("post_edit_validation_receipt", PostEditValidationReceipt),
            ("reproof_receipt", ReproofReceipt),
            ("publication_receipt", PublicationReceipt),
        ):
            receipt = getattr(self, field_name)
            if receipt is not None and not isinstance(receipt, receipt_type):
                raise DeterministicRepairContractError(
                    f"{field_name} must be {receipt_type.__name__}"
                )
        object.__setattr__(
            self,
            "implementation_disposition",
            _dcr_identifier(self.implementation_disposition, "implementation_disposition"),
        )
        object.__setattr__(self, "producer_id", _dcr_identifier(self.producer_id, "producer_id"))

        assert_deterministic_repair_transition(previous, stage)
        if previous is None:
            if self.previous_envelope_cid:
                raise DeterministicRepairAuthorityError(
                    "initial observed evidence must not name a previous envelope"
                )
        elif not self.previous_envelope_cid:
            raise DeterministicRepairAuthorityError(
                "non-initial repair evidence requires previous_envelope_cid"
            )
        self._require_evidence_for_state()
        if (
            self.disposition is DeterministicRepairDisposition.COMPLETED
            and self.authority_stage is not AuthorityStage.PUBLISHED
        ):
            raise DeterministicRepairAuthorityError(
                "completion requires a published authority stage"
            )

    def _require_evidence_for_state(self) -> None:
        state = self.authority_stage
        if (
            state
            in {
                AuthorityStage.DERIVED,
                AuthorityStage.ADMITTED,
                AuthorityStage.MUTATED,
                AuthorityStage.POST_EDIT_VALIDATED,
                AuthorityStage.REPROVED,
                AuthorityStage.PUBLISHED,
            }
            and not self.derivation_cid
        ):
            raise DeterministicRepairAuthorityError(
                "derived and later repair evidence requires derivation_cid"
            )
        if state in {
            AuthorityStage.ADMITTED,
            AuthorityStage.MUTATED,
            AuthorityStage.POST_EDIT_VALIDATED,
            AuthorityStage.REPROVED,
            AuthorityStage.PUBLISHED,
        }:
            if not self.admission_cid:
                raise DeterministicRepairAuthorityError(
                    "admitted and later repair evidence requires admission_cid"
                )
            if self.implementation_disposition != "closed_deterministic":
                raise DeterministicRepairAuthorityError(
                    "mutation authority requires implementation disposition closed_deterministic"
                )
        if (
            state
            in {
                AuthorityStage.MUTATED,
                AuthorityStage.POST_EDIT_VALIDATED,
                AuthorityStage.REPROVED,
                AuthorityStage.PUBLISHED,
            }
            and not self.mutation_receipt_cid
        ):
            raise DeterministicRepairAuthorityError(
                "mutated and later repair evidence requires mutation_receipt_cid"
            )
        if (
            state
            in {
                AuthorityStage.POST_EDIT_VALIDATED,
                AuthorityStage.REPROVED,
                AuthorityStage.PUBLISHED,
            }
            and not self.post_edit_validation_cid
        ):
            raise DeterministicRepairAuthorityError(
                "post-edit validation is required before reproof, publication, or completion"
            )
        if (
            state
            in {
                AuthorityStage.REPROVED,
                AuthorityStage.PUBLISHED,
            }
            and not self.reproof_cid
        ):
            raise DeterministicRepairAuthorityError(
                "reproof is required before publication or completion"
            )
        if (
            state
            in {
                AuthorityStage.PUBLISHED,
            }
            and not self.publication_cid
        ):
            raise DeterministicRepairAuthorityError("publication is required before completion")

    def require_typed_authority(
        self,
        *,
        require_mutation_authority: bool = False,
        require_completion: bool = False,
    ) -> None:
        """Verify typed receipts and exact links before granting authority."""

        state = self.authority_stage
        needs_admission = require_mutation_authority or state in {
            AuthorityStage.ADMITTED,
            AuthorityStage.MUTATED,
            AuthorityStage.POST_EDIT_VALIDATED,
            AuthorityStage.REPROVED,
            AuthorityStage.PUBLISHED,
        }
        admission = self.admission_receipt
        if needs_admission:
            if not isinstance(admission, RepairAdmissionReceipt):
                raise DeterministicRepairAuthorityError(
                    "typed admission receipt is required for mutation authority"
                )
            self._require_receipt_common(admission, "admission receipt")
            if (
                (
                    state is AuthorityStage.ADMITTED
                    and admission.predecessor_evidence_cid != self.previous_envelope_cid
                )
                or admission.derivation_cid != self.derivation_cid
                or self.admission_cid != admission.content_id
            ):
                raise DeterministicRepairAuthorityError(
                    "admission receipt is not linked to the exact predecessor evidence"
                )
        if (
            state
            in {
                AuthorityStage.POST_EDIT_VALIDATED,
                AuthorityStage.REPROVED,
                AuthorityStage.PUBLISHED,
            }
            or require_completion
        ):
            validation = self.post_edit_validation_receipt
            if not isinstance(validation, PostEditValidationReceipt) or not validation.passed:
                raise DeterministicRepairAuthorityError(
                    "successful typed post-edit validation receipt is required"
                )
            self._require_receipt_common(validation, "post-edit validation receipt")
            if (
                (
                    state is AuthorityStage.POST_EDIT_VALIDATED
                    and validation.predecessor_evidence_cid != self.previous_envelope_cid
                )
                or validation.admission_receipt_cid != self.admission_cid
                or validation.mutation_receipt_cid != self.mutation_receipt_cid
                or self.post_edit_validation_cid != validation.content_id
            ):
                raise DeterministicRepairAuthorityError(
                    "post-edit validation receipt is not linked to the exact mutation"
                )
        if state in {AuthorityStage.REPROVED, AuthorityStage.PUBLISHED} or require_completion:
            reproof = self.reproof_receipt
            if not isinstance(reproof, ReproofReceipt) or not reproof.proved:
                raise DeterministicRepairAuthorityError(
                    "successful typed reproof receipt is required"
                )
            self._require_receipt_common(reproof, "reproof receipt")
            if (
                (
                    state is AuthorityStage.REPROVED
                    and reproof.predecessor_evidence_cid != self.previous_envelope_cid
                )
                or reproof.admission_receipt_cid != self.admission_cid
                or reproof.post_edit_validation_receipt_cid != self.post_edit_validation_cid
                or reproof.mutation_receipt_cid != self.mutation_receipt_cid
                or self.reproof_cid != reproof.content_id
            ):
                raise DeterministicRepairAuthorityError(
                    "reproof receipt is not linked to the exact validated mutation"
                )
        if state is AuthorityStage.PUBLISHED or require_completion:
            publication = self.publication_receipt
            if not isinstance(publication, PublicationReceipt) or not publication.published:
                raise DeterministicRepairAuthorityError(
                    "typed publication receipt is required for completion"
                )
            self._require_receipt_common(publication, "publication receipt")
            if (
                publication.predecessor_evidence_cid != self.previous_envelope_cid
                or publication.admission_receipt_cid != self.admission_cid
                or publication.post_edit_validation_receipt_cid != self.post_edit_validation_cid
                or publication.reproof_receipt_cid != self.reproof_cid
                or publication.mutation_receipt_cid != self.mutation_receipt_cid
                or self.publication_cid != publication.content_id
            ):
                raise DeterministicRepairAuthorityError(
                    "publication receipt is not linked to the exact reproof"
                )

    def _require_receipt_common(
        self,
        receipt: CanonicalContract,
        receipt_name: str,
    ) -> None:
        receipt_repair_id = getattr(receipt, "repair_id", "")
        receipt_roots = getattr(receipt, "authority_roots", None)
        if receipt_repair_id != self.repair_id:
            raise DeterministicRepairAuthorityError(
                f"{receipt_name} repair_id does not match evidence"
            )
        if not isinstance(receipt_roots, RepairAuthorityRoots):
            raise DeterministicRepairAuthorityError(f"{receipt_name} roots are malformed")
        receipt_roots.require_current(self.authority_roots)

    @property
    def authorizes_mutation(self) -> bool:
        if not self.authority_stage.authorizes_mutation:
            return False
        try:
            self.require_typed_authority(require_mutation_authority=True)
        except DeterministicRepairAuthorityError:
            return False
        return True

    @property
    def completion_authoritative(self) -> bool:
        """Return false without the prior envelope needed to prove continuity.

        A published envelope carries typed receipts, but not a recursively
        embedded predecessor chain.  Call
        :func:`verify_repair_evidence_envelope` with its exact ``REPROVED``
        predecessor before treating completion as authoritative.
        """

        # This property cannot inspect the exact predecessor required to bind
        # the publication to the append-only evidence chain.  Fail closed
        # rather than overstating the authority of a standalone envelope.
        return False

    def _require_preserved_evidence(self, previous: RepairEvidenceEnvelope) -> None:
        """Reject successors that rewrite evidence already fixed by ``previous``."""

        for field_name in (
            "observation_cid",
            "derivation_cid",
            "admission_cid",
            "mutation_receipt_cid",
            "post_edit_validation_cid",
            "reproof_cid",
            "publication_cid",
        ):
            prior_value = getattr(previous, field_name)
            if prior_value and getattr(self, field_name) != prior_value:
                raise DeterministicRepairAuthorityError(
                    f"{field_name} must remain bound to the exact prior evidence"
                )
        for field_name in (
            "admission_receipt",
            "post_edit_validation_receipt",
            "reproof_receipt",
            "publication_receipt",
        ):
            prior_receipt = getattr(previous, field_name)
            if prior_receipt is not None and getattr(self, field_name) != prior_receipt:
                raise DeterministicRepairAuthorityError(
                    f"{field_name} must remain bound to the exact prior receipt"
                )
        if self.implementation_disposition != previous.implementation_disposition:
            raise DeterministicRepairAuthorityError(
                "implementation_disposition must remain bound to prior policy authority"
            )
        if self.producer_id != previous.producer_id:
            raise DeterministicRepairAuthorityError(
                "producer_id must remain bound to the prior repair evidence"
            )

    @property
    def completion_structurally_complete(self) -> bool:
        """Return whether local typed completion prerequisites are present.

        This deliberately does not assert authority: exact predecessor
        continuity remains mandatory and is checked by the verifier.
        """

        if not (
            self.disposition is DeterministicRepairDisposition.COMPLETED
            and self.authority_stage is AuthorityStage.PUBLISHED
        ):
            return False
        try:
            self.require_typed_authority(require_completion=True)
        except DeterministicRepairAuthorityError:
            return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "repair_id": self.repair_id,
            "disposition": self.disposition.value,
            "authority_stage": self.authority_stage.value,
            "authority_roots": self.authority_roots.to_dict(),
            "observation_cid": self.observation_cid,
            "previous_authority_stage": (
                self.previous_authority_stage.value if self.previous_authority_stage else None
            ),
            "previous_envelope_cid": self.previous_envelope_cid,
            "derivation_cid": self.derivation_cid,
            "admission_cid": self.admission_cid,
            "mutation_receipt_cid": self.mutation_receipt_cid,
            "post_edit_validation_cid": self.post_edit_validation_cid,
            "reproof_cid": self.reproof_cid,
            "publication_cid": self.publication_cid,
            "admission_receipt": (
                self.admission_receipt.to_dict() if self.admission_receipt is not None else None
            ),
            "post_edit_validation_receipt": (
                self.post_edit_validation_receipt.to_dict()
                if self.post_edit_validation_receipt is not None
                else None
            ),
            "reproof_receipt": (
                self.reproof_receipt.to_dict() if self.reproof_receipt is not None else None
            ),
            "publication_receipt": (
                self.publication_receipt.to_dict() if self.publication_receipt is not None else None
            ),
            "implementation_disposition": self.implementation_disposition,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairEvidenceEnvelope:
        fields = (
            "repair_id",
            "disposition",
            "authority_stage",
            "authority_roots",
            "observation_cid",
            "previous_authority_stage",
            "previous_envelope_cid",
            "derivation_cid",
            "admission_cid",
            "mutation_receipt_cid",
            "post_edit_validation_cid",
            "reproof_cid",
            "publication_cid",
            "admission_receipt",
            "post_edit_validation_receipt",
            "reproof_receipt",
            "publication_receipt",
            "implementation_disposition",
            "producer_id",
        )
        values = _dcr_decode(payload, cls.SCHEMA, fields, "repair evidence envelope")
        roots = values["authority_roots"]
        if not isinstance(roots, MappingABC):
            raise DeterministicRepairContractError("authority_roots must be an object")
        values["authority_roots"] = RepairAuthorityRoots.from_dict(roots)
        for field_name, receipt_type in (
            ("admission_receipt", RepairAdmissionReceipt),
            ("post_edit_validation_receipt", PostEditValidationReceipt),
            ("reproof_receipt", ReproofReceipt),
            ("publication_receipt", PublicationReceipt),
        ):
            receipt = values[field_name]
            if receipt is None:
                continue
            if not isinstance(receipt, MappingABC):
                raise DeterministicRepairContractError(f"{field_name} must be an object or null")
            values[field_name] = receipt_type.from_dict(receipt)
        result = cls(**values)
        _dcr_verify_identity(payload, result)
        return result

    def advances(self, previous: RepairEvidenceEnvelope) -> bool:
        """Return true only when this envelope is the exact next evidence link."""

        try:
            self.require_advances(previous)
        except DeterministicRepairContractError:
            return False
        return True

    def require_advances(self, previous: RepairEvidenceEnvelope) -> None:
        if not isinstance(previous, RepairEvidenceEnvelope):
            raise DeterministicRepairAuthorityError(
                "repair evidence transition requires a canonical previous envelope"
            )
        if self.repair_id != previous.repair_id:
            raise DeterministicRepairAuthorityError("repair evidence repair_id does not match")
        previous.authority_roots.require_current(self.authority_roots)
        if self.previous_envelope_cid != previous.content_id:
            raise DeterministicRepairAuthorityError(
                "previous_envelope_cid does not match the exact prior evidence"
            )
        if self.previous_authority_stage is not previous.authority_stage:
            raise DeterministicRepairAuthorityError(
                "previous_authority_stage does not match the exact prior evidence"
            )
        assert_deterministic_repair_transition(previous.authority_stage, self.authority_stage)
        self._require_preserved_evidence(previous)


def verify_repair_evidence_envelope(
    payload: Mapping[str, Any] | RepairEvidenceEnvelope,
    *,
    expected_authority_roots: RepairAuthorityRoots | None = None,
    previous: RepairEvidenceEnvelope | None = None,
    require_mutation_authority: bool = False,
    require_completion: bool = False,
) -> RepairEvidenceEnvelope:
    """Decode and bind DCR-002 evidence against roots and its exact predecessor."""

    envelope = (
        payload
        if isinstance(payload, RepairEvidenceEnvelope)
        else RepairEvidenceEnvelope.from_dict(payload)
    )
    if expected_authority_roots is not None:
        envelope.authority_roots.require_current(expected_authority_roots)
    if previous is not None:
        envelope.require_advances(previous)
    if require_mutation_authority:
        if envelope.authority_stage is not AuthorityStage.ADMITTED:
            raise DeterministicRepairAuthorityError(
                "mutation requires an admitted deterministic repair evidence envelope"
            )
        envelope.require_typed_authority(require_mutation_authority=True)
    if require_completion:
        if (
            envelope.disposition is not DeterministicRepairDisposition.COMPLETED
            or envelope.authority_stage is not AuthorityStage.PUBLISHED
        ):
            raise DeterministicRepairAuthorityError(
                "completion requires a completed published evidence envelope"
            )
        if previous is None:
            raise DeterministicRepairAuthorityError(
                "completion requires the exact prior reproved evidence envelope"
            )
        if previous.authority_stage is not AuthorityStage.REPROVED:
            raise DeterministicRepairAuthorityError(
                "completion requires an exact prior reproved evidence envelope"
            )
        envelope.require_typed_authority(require_completion=True)
    return envelope


@dataclass
class AutonomousRepairPolicy:
    """Policy for one autonomous repair run (LLM remains forbidden)."""

    domain: str = "agent_supervisor"
    consumer: str = "autonomous_repair"
    apply_ir_logic: bool = True
    apply_doctor: bool = True
    require_zero_model_calls: bool = True
    prefer_mcp_server: bool = True
    prefer_mediation: bool = True  # package_mcp_interop / tools/call
    max_items: int = 32
    ir_families: tuple[str, ...] = (
        "intent_ir",
        "legal_ir",
        "security_ir",
        "ui_ir",
        "ast",
        "knowledge_graph",
        "vector_index",
    )
    allow_code_edit_materialize: bool = False
    """When True, single-path items may emit materialize-ready edit plans.
    Default False: analysis + plan only (fail-closed for production trees).
    """

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AutonomousRepairPolicy:
        raw = dict(raw or {})
        fams = raw.get("irFamilies") or raw.get("ir_families")
        return cls(
            domain=str(raw.get("domain") or "agent_supervisor"),
            consumer=str(raw.get("consumer") or "autonomous_repair"),
            apply_ir_logic=bool(raw.get("applyIrLogic", raw.get("apply_ir_logic", True))),
            apply_doctor=bool(raw.get("applyDoctor", raw.get("apply_doctor", True))),
            require_zero_model_calls=bool(
                raw.get(
                    "requireZeroModelCalls",
                    raw.get("require_zero_model_calls", True),
                )
            ),
            prefer_mcp_server=bool(raw.get("preferMcpServer", raw.get("prefer_mcp_server", True))),
            prefer_mediation=bool(raw.get("preferMediation", raw.get("prefer_mediation", True))),
            max_items=int(raw.get("maxItems") or raw.get("max_items") or 32),
            ir_families=tuple(fams or cls.ir_families),
            allow_code_edit_materialize=bool(
                raw.get(
                    "allowCodeEditMaterialize",
                    raw.get("allow_code_edit_materialize", False),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairWorkItem:
    """One domain-agnostic repair work item (not SCA-specific)."""

    work_id: str
    operation: str
    kind: str = "work_item"
    contract_id: str = ""
    package: str = ""
    path: str = ""
    symbol: str = ""
    write_paths: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    domain: str = "agent_supervisor"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RepairWorkItem:
        raw = dict(raw or {})
        op = str(raw.get("operation") or raw.get("op") or raw.get("tool") or "")
        contract_id = str(
            raw.get("contract_id") or raw.get("operation_id") or raw.get("surface_id") or ""
        )
        if not op and contract_id and ":" in contract_id:
            op = contract_id.split(":", 1)[-1]
        package = str(raw.get("package") or "")
        if not package and ":" in contract_id:
            package = contract_id.split(":", 1)[0]
        write_paths = raw.get("write_paths") or raw.get("writePaths") or ()
        if isinstance(write_paths, str):
            write_paths = (write_paths,)
        reasons = raw.get("reason_codes") or raw.get("reasonCodes") or ()
        if isinstance(reasons, str):
            reasons = (reasons,)
        path = str(raw.get("path") or (write_paths[0] if write_paths else "") or "")
        return cls(
            work_id=str(
                raw.get("work_id")
                or raw.get("task_id")
                or raw.get("finding_id")
                or raw.get("id")
                or f"work:{op or 'unknown'}"
            ),
            operation=op or "unknown",
            kind=str(
                raw.get("kind")
                or raw.get("reason_code")
                or (reasons[0] if reasons else "work_item")
            ),
            contract_id=contract_id or f"surface:{op or 'unknown'}",
            package=package,
            path=path,
            symbol=str(raw.get("symbol") or op or ""),
            write_paths=tuple(str(p) for p in write_paths),
            reason_codes=tuple(str(r) for r in reasons),
            domain=str(raw.get("domain") or "agent_supervisor"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = REPAIR_WORK_ITEM_SCHEMA
        return d


@dataclass
class AutonomousRepairReport:
    """Batch report for one autonomous repair engine run."""

    policy: dict[str, Any]
    rows: list[dict[str, Any]]
    passed: bool
    model_call_count: int = 0
    llm_used: bool = False
    recorded_at: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUTONOMOUS_REPAIR_REPORT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_INTERFACE,
            "recorded_at": self.recorded_at,
            "passed": self.passed,
            "llm_used": self.llm_used,
            "model_call_count": self.model_call_count,
            "completion_authoritative": False,
            "policy": self.policy,
            "summary": self.summary,
            "rows": self.rows,
            "notes": self.notes,
        }


__all__ = [
    "AUTONOMOUS_REPAIR_INTERFACE",
    "AUTONOMOUS_REPAIR_REPORT_SCHEMA",
    "AuthorityStage",
    "DETERMINISTIC_REPAIR_CONTRACT_VERSION",
    "DETERMINISTIC_REPAIR_INTERFACE",
    "DeterministicRepairAuthorityError",
    "DeterministicRepairContractError",
    "DeterministicRepairDisposition",
    "ForgedRepairEvidenceIdentityError",
    "REPAIR_PLAN_SCHEMA",
    "REPAIR_AUTHORITY_ROOTS_SCHEMA",
    "REPAIR_EVIDENCE_ENVELOPE_INTERFACE",
    "REPAIR_EVIDENCE_ENVELOPE_SCHEMA",
    "REPAIR_WORK_ITEM_SCHEMA",
    "AutonomousRepairPolicy",
    "AutonomousRepairReport",
    "RepairAuthorityRoots",
    "RepairDisposition",
    "RepairEvidenceEnvelope",
    "RepairWorkItem",
    "assert_deterministic_repair_transition",
    "assert_authority_stage_transition",
    "closed_deterministic_repair_dispositions",
    "parse_deterministic_repair_disposition",
    "parse_authority_stage",
    "repair_evidence_cid",
    "repair_evidence_digest",
    "verify_repair_evidence_envelope",
]
