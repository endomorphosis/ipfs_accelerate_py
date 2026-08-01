"""Prove exact placement for new support classes, methods, and data structures.

When change propagation needs a new class, method, data structure, provider,
factory, or schema, :class:`RequiredBehaviorContract` states *what* is required.
This module proves *where* it may live (or reuses an existing admissible
implementation) under architecture ownership, layering, visibility,
registration/export/DI wiring, and dependency acyclicity.

Authority rules (fail-closed):

* Vector, knowledge-graph, and LLM nominations never authorize a target path.
* Paths on an admitted decision come only from the selected candidate's exact
  placement paths; candidates never inherit write authority from retrieval.
* Prefer an existing admissible implementation over creating a duplicate.
* Unique eligible candidate plus rank margin are required; ties, missing
  owner, cross-root writes, cycles, unsupported lifecycle/native semantics, or
  unproved behavior yield abstention (or review-only when authority is thin).
* Reuses :class:`ImplementationSiteAdmissibility` site decisions as evidence
  when supplied; does not invent architecture or graph facts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Final

from ..analysis.change_propagation_contracts import (
    BehaviorKind,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)
from ..proof.formal_verification_contracts import (
    ProofReceipt,
    ProofVerdict,
    content_identity,
)
from .implementation_site_admissibility import (
    PlacementDecision,
    PlacementDisposition,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

SUPPORT_BEHAVIOR_PLACEMENT_INTERFACE: Final[str] = "SupportBehaviorPlacement@1"
SUPPORT_PLACEMENT_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/support-placement-candidate@1"
)
SUPPORT_PLACEMENT_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/support-placement-decision@1"
)
PLACEMENT_ANCHOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/support-placement-anchor@1"
)
PRODUCER_ID: Final[str] = "support-behavior-placement@1"

MAX_CANDIDATES: Final[int] = 256
MAX_ANCHORS: Final[int] = 256
MAX_PATHS: Final[int] = 32
MAX_PROOF_RECEIPTS: Final[int] = 64
MAX_EVIDENCE_REFS: Final[int] = 128
DEFAULT_MINIMUM_MARGIN: Final[int] = 1
SCORE_SCALE: Final[int] = 1_000

_SUPPORTED_RUNTIMES: Final[frozenset[str]] = frozenset(
    {
        "python",
        "python3",
        "py",
        "cpython",
    }
)
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
        ".git",
    }
)
_FORBIDDEN_NOMINATION_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "vector",
        "embedding",
        "kg",
        "knowledge_graph",
        "llm",
        "model",
        "retrieval",
        "similarity",
        "bm25",
        "lexical",
    }
)


# ---------------------------------------------------------------------------
# Errors / enumerations
# ---------------------------------------------------------------------------


class SupportBehaviorPlacementError(ValueError):
    """Malformed or unsafe support-placement input."""


class SupportBehaviorPlacementAuthorityError(SupportBehaviorPlacementError):
    """Root, identity, or authority promotion failure."""


class PlacementAnchorKind(str, Enum):
    """Closed sources from which placement candidates may be enumerated."""

    DECLARATION = "declaration"
    INTERFACE = "interface"
    ARCHITECTURE_OWNERSHIP = "architecture_ownership"
    FACTORY = "factory"
    PROVIDER = "provider"
    SCHEMA = "schema"
    EXISTING_PLACEMENT_ANCHOR = "existing_placement_anchor"
    EXISTING_ADMISSIBLE_IMPLEMENTATION = "existing_admissible_implementation"


class SupportPlacementDisposition(str, Enum):
    """Outcomes of support-behavior placement admission."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    AMBIGUOUS = "ambiguous"
    REVIEW_ONLY = "review_only"


class SupportPlacementAction(str, Enum):
    """What the admitted placement authorizes next."""

    REUSE_EXISTING = "reuse_existing"
    PLACE_NEW = "place_new"
    NONE = "none"


# Anchor kinds that are structural (not automatic reuse).
_STRUCTURAL_ANCHOR_KINDS: Final[frozenset[PlacementAnchorKind]] = frozenset(
    {
        PlacementAnchorKind.DECLARATION,
        PlacementAnchorKind.INTERFACE,
        PlacementAnchorKind.ARCHITECTURE_OWNERSHIP,
        PlacementAnchorKind.FACTORY,
        PlacementAnchorKind.PROVIDER,
        PlacementAnchorKind.SCHEMA,
        PlacementAnchorKind.EXISTING_PLACEMENT_ANCHOR,
    }
)

# Higher is better for soft ranking among eligible candidates.
_ANCHOR_PRIORITY: Final[Mapping[PlacementAnchorKind, int]] = {
    PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION: 100,
    PlacementAnchorKind.ARCHITECTURE_OWNERSHIP: 90,
    PlacementAnchorKind.DECLARATION: 80,
    PlacementAnchorKind.INTERFACE: 70,
    PlacementAnchorKind.FACTORY: 60,
    PlacementAnchorKind.PROVIDER: 55,
    PlacementAnchorKind.SCHEMA: 50,
    PlacementAnchorKind.EXISTING_PLACEMENT_ANCHOR: 40,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupportBehaviorPlacementError(f"{field_name} is required")
    text = value.strip()
    if any(char.isspace() for char in text):
        raise SupportBehaviorPlacementError(
            f"{field_name} must be a compact identifier"
        )
    if len(text.encode("utf-8")) > 512:
        raise SupportBehaviorPlacementError(f"{field_name} exceeds size bound")
    return text


def _text(value: object, field_name: str, *, required: bool = True) -> str:
    if value is None or value == "":
        if required:
            raise SupportBehaviorPlacementError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise SupportBehaviorPlacementError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise SupportBehaviorPlacementError(f"{field_name} is required")
    if len(text.encode("utf-8")) > 1024:
        raise SupportBehaviorPlacementError(f"{field_name} exceeds size bound")
    return text


def _path(value: object, field_name: str) -> str:
    raw = _identifier(value, field_name).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise SupportBehaviorPlacementError(
            f"{field_name} must be a normalized repository-relative path"
        )
    return raw


def _paths(values: Sequence[str], field_name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SupportBehaviorPlacementError(f"{field_name} must be a sequence of paths")
    rows = tuple(_path(item, field_name) for item in values)
    if required and not rows:
        raise SupportBehaviorPlacementError(f"{field_name} must not be empty")
    if len(rows) > MAX_PATHS:
        raise SupportBehaviorPlacementError(f"{field_name} exceeds path bound")
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in rows:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _ids(values: Sequence[str], field_name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SupportBehaviorPlacementError(
            f"{field_name} must be a sequence of identifiers"
        )
    result = tuple(sorted({_identifier(item, field_name) for item in values}))
    if required and not result:
        raise SupportBehaviorPlacementError(f"{field_name} must not be empty")
    return result


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SupportBehaviorPlacementError(f"{field_name} must be boolean")
    return value


def _enum(value: object, enum_cls: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise SupportBehaviorPlacementError(
                f"{field_name} must be a valid {enum_cls.__name__}"
            ) from exc
    raise SupportBehaviorPlacementError(
        f"{field_name} must be a valid {enum_cls.__name__}"
    )


def _roots(value: object) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    raise SupportBehaviorPlacementError("roots must be PropagationAuthorityRoots")


def _path_forbidden(path: str) -> bool:
    return any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in PurePosixPath(path).parts)


def _nomination_forbidden(source: str) -> bool:
    token = source.strip().casefold().replace("-", "_")
    if not token:
        return False
    if token in _FORBIDDEN_NOMINATION_SOURCES:
        return True
    return any(marker in token for marker in _FORBIDDEN_NOMINATION_SOURCES)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementAnchor:
    """One architecture- or declaration-bound placement anchor fact.

    Anchors are enumerated into candidates; they do not themselves grant
    placement or mutation authority.
    """

    roots: PropagationAuthorityRoots
    anchor_id: str
    kind: PlacementAnchorKind
    target_path: str
    owner_id: str
    module_owner_id: str
    language_runtime: str = "python"
    evidence_refs: tuple[str, ...] = ()
    interface_id: str = ""
    declaration_id: str = ""
    registration_route_id: str = ""
    export_route_id: str = ""
    di_wiring_route_id: str = ""
    nomination_source: str = "architecture"

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "anchor_id", _identifier(self.anchor_id, "anchor_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, PlacementAnchorKind, "kind")
        )
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        object.__setattr__(self, "owner_id", _identifier(self.owner_id, "owner_id"))
        object.__setattr__(
            self, "module_owner_id", _identifier(self.module_owner_id, "module_owner_id")
        )
        object.__setattr__(
            self,
            "language_runtime",
            _identifier(self.language_runtime, "language_runtime").casefold(),
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        for name in (
            "interface_id",
            "declaration_id",
            "registration_route_id",
            "export_route_id",
            "di_wiring_route_id",
            "nomination_source",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.kind is PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION:
            raise SupportBehaviorPlacementError(
                "use ExistingImplementationFact for reuse candidates, not PlacementAnchor"
            )

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": PLACEMENT_ANCHOR_SCHEMA,
                "anchor_id": self.anchor_id,
                "kind": self.kind.value,
                "target_path": self.target_path,
                "owner_id": self.owner_id,
                "module_owner_id": self.module_owner_id,
                "roots": self.roots.content_id,
            }
        )


@dataclass(frozen=True)
class ExistingImplementationFact:
    """An existing implementation that may be reused instead of duplicating."""

    roots: PropagationAuthorityRoots
    implementation_id: str
    target_path: str
    owner_id: str
    module_owner_id: str
    subject_symbol_id: str
    language_runtime: str = "python"
    admissible: bool = False
    behavior_contract_fit: bool = False
    site_placement_admitted: bool = False
    proof_receipt_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    placement_paths: tuple[str, ...] = ()
    nomination_source: str = "architecture"

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self,
            "implementation_id",
            _identifier(self.implementation_id, "implementation_id"),
        )
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        object.__setattr__(self, "owner_id", _identifier(self.owner_id, "owner_id"))
        object.__setattr__(
            self, "module_owner_id", _identifier(self.module_owner_id, "module_owner_id")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "language_runtime",
            _identifier(self.language_runtime, "language_runtime").casefold(),
        )
        for name in ("admissible", "behavior_contract_fit", "site_placement_admitted"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "proof_receipt_ids",
            _ids(self.proof_receipt_ids, "proof_receipt_ids"),
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        if self.placement_paths:
            object.__setattr__(
                self,
                "placement_paths",
                _paths(self.placement_paths, "placement_paths", required=True),
            )
        else:
            object.__setattr__(self, "placement_paths", (self.target_path,))
        object.__setattr__(
            self,
            "nomination_source",
            _text(self.nomination_source, "nomination_source", required=False)
            or "architecture",
        )


@dataclass(frozen=True)
class SupportPlacementCandidate:
    """One bounded placement candidate for a required support behavior.

    Boolean facts make policy constraints explicit; reconstructed proof receipt
    identities are still required for admission.  ``placement_paths`` are the
    only paths that may appear on a later admitted decision for this candidate.
    """

    roots: PropagationAuthorityRoots
    candidate_id: str
    behavior_id: str
    subject_symbol_id: str
    anchor_kind: PlacementAnchorKind
    anchor_id: str
    target_path: str
    placement_paths: tuple[str, ...]
    owner_id: str
    module_owner_id: str
    language_runtime: str = "python"
    proof_receipt_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    interface_id: str = ""
    declaration_id: str = ""
    nomination_source: str = "architecture"
    site_placement_decision_ref: str = ""
    # Proved architectural / policy facts
    ownership_exact: bool = False
    owner_unambiguous: bool = False
    visibility_route_satisfiable: bool = False
    dependency_direction_legal: bool = False
    dependency_acyclic: bool = False
    registration_export_di_wiring_satisfiable: bool = False
    capability_supported: bool = False
    effect_supported: bool = False
    resource_supported: bool = False
    memory_supported: bool = False
    mutation_authority_exact: bool = False
    behavior_contract_fit: bool = False
    behavior_proved: bool = False
    lifecycle_supported: bool = False
    # Hard exclusion flags (True means reject)
    generated: bool = False
    vendor: bool = False
    read_only: bool = False
    cross_root_write: bool = False
    native_semantics_unsupported: bool = False
    lifecycle_native_unsupported: bool = False
    site_placement_admitted: bool = False
    is_reuse: bool = False
    score_vector: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "behavior_id", _identifier(self.behavior_id, "behavior_id")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "anchor_kind",
            _enum(self.anchor_kind, PlacementAnchorKind, "anchor_kind"),
        )
        object.__setattr__(self, "anchor_id", _identifier(self.anchor_id, "anchor_id"))
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        object.__setattr__(
            self,
            "placement_paths",
            _paths(self.placement_paths, "placement_paths", required=True),
        )
        if self.target_path not in self.placement_paths:
            raise SupportBehaviorPlacementError(
                "placement_paths must include the candidate target_path"
            )
        object.__setattr__(self, "owner_id", _identifier(self.owner_id, "owner_id"))
        object.__setattr__(
            self, "module_owner_id", _identifier(self.module_owner_id, "module_owner_id")
        )
        object.__setattr__(
            self,
            "language_runtime",
            _identifier(self.language_runtime, "language_runtime").casefold(),
        )
        object.__setattr__(
            self,
            "proof_receipt_ids",
            _ids(self.proof_receipt_ids, "proof_receipt_ids"),
        )
        if len(self.proof_receipt_ids) > MAX_PROOF_RECEIPTS:
            raise SupportBehaviorPlacementError("proof_receipt_ids exceeds bound")
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        if len(self.evidence_refs) > MAX_EVIDENCE_REFS:
            raise SupportBehaviorPlacementError("evidence_refs exceeds bound")
        for name in (
            "interface_id",
            "declaration_id",
            "nomination_source",
            "site_placement_decision_ref",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if not self.nomination_source:
            object.__setattr__(self, "nomination_source", "architecture")
        for name in (
            "ownership_exact",
            "owner_unambiguous",
            "visibility_route_satisfiable",
            "dependency_direction_legal",
            "dependency_acyclic",
            "registration_export_di_wiring_satisfiable",
            "capability_supported",
            "effect_supported",
            "resource_supported",
            "memory_supported",
            "mutation_authority_exact",
            "behavior_contract_fit",
            "behavior_proved",
            "lifecycle_supported",
            "generated",
            "vendor",
            "read_only",
            "cross_root_write",
            "native_semantics_unsupported",
            "lifecycle_native_unsupported",
            "site_placement_admitted",
            "is_reuse",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.is_reuse != (
            self.anchor_kind is PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION
        ):
            raise SupportBehaviorPlacementError(
                "is_reuse must match EXISTING_ADMISSIBLE_IMPLEMENTATION anchor kind"
            )
        if self.score_vector:
            if (
                isinstance(self.score_vector, (str, bytes, bytearray))
                or not isinstance(self.score_vector, Sequence)
                or not self.score_vector
            ):
                raise SupportBehaviorPlacementError(
                    "score_vector must be a non-empty integer sequence when provided"
                )
            scores = tuple(self.score_vector)
            if not all(
                isinstance(item, int) and not isinstance(item, bool) for item in scores
            ):
                raise SupportBehaviorPlacementError(
                    "score_vector must contain plain integers"
                )
            object.__setattr__(self, "score_vector", scores)

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": SUPPORT_PLACEMENT_CANDIDATE_SCHEMA,
                "candidate_id": self.candidate_id,
                "behavior_id": self.behavior_id,
                "subject_symbol_id": self.subject_symbol_id,
                "anchor_kind": self.anchor_kind.value,
                "anchor_id": self.anchor_id,
                "target_path": self.target_path,
                "placement_paths": list(self.placement_paths),
                "owner_id": self.owner_id,
                "module_owner_id": self.module_owner_id,
                "is_reuse": self.is_reuse,
                "roots": self.roots.content_id,
            }
        )

    def default_score_vector(self) -> tuple[int, ...]:
        """Deterministic ranking vector used when no explicit score is supplied."""

        if self.score_vector:
            return self.score_vector
        reuse = SCORE_SCALE if self.is_reuse else 0
        anchor = _ANCHOR_PRIORITY.get(self.anchor_kind, 0)
        fit = 0
        fit += 20 if self.ownership_exact else 0
        fit += 15 if self.owner_unambiguous else 0
        fit += 10 if self.visibility_route_satisfiable else 0
        fit += 10 if self.dependency_direction_legal and self.dependency_acyclic else 0
        fit += 10 if self.registration_export_di_wiring_satisfiable else 0
        fit += 10 if self.behavior_contract_fit and self.behavior_proved else 0
        fit += 5 if self.mutation_authority_exact else 0
        fit += 5 if self.site_placement_admitted else 0
        support = 0
        support += 1 if self.capability_supported else 0
        support += 1 if self.effect_supported else 0
        support += 1 if self.resource_supported else 0
        support += 1 if self.memory_supported else 0
        support += 1 if self.lifecycle_supported else 0
        return (reuse, anchor, fit, support)


@dataclass(frozen=True)
class SupportPlacementDecision:
    """Deterministic placement decision; paths exist only when admitted."""

    disposition: SupportPlacementDisposition
    roots: PropagationAuthorityRoots
    behavior_id: str
    candidate_set_id: str
    selected_candidate_id: str = ""
    action: SupportPlacementAction = SupportPlacementAction.NONE
    target_path: str = ""
    placement_paths: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    eligible_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    margin: int | None = None
    site_placement_decision_ref: str = ""
    schema: str = SUPPORT_PLACEMENT_DECISION_SCHEMA
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SupportPlacementDisposition, "disposition"),
        )
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "behavior_id", _identifier(self.behavior_id, "behavior_id")
        )
        object.__setattr__(
            self, "candidate_set_id", _identifier(self.candidate_set_id, "candidate_set_id")
        )
        object.__setattr__(
            self,
            "action",
            _enum(self.action, SupportPlacementAction, "action"),
        )
        for name in ("selected_candidate_id", "target_path", "site_placement_decision_ref"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.target_path:
            object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        if self.placement_paths:
            object.__setattr__(
                self,
                "placement_paths",
                _paths(self.placement_paths, "placement_paths", required=True),
            )
        else:
            object.__setattr__(self, "placement_paths", ())
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_identifier(item, "reason_code") for item in self.reason_codes})),
        )
        object.__setattr__(
            self, "proof_receipt_ids", _ids(self.proof_receipt_ids, "proof_receipt_ids")
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self,
            "eligible_candidate_ids",
            _ids(self.eligible_candidate_ids, "eligible_candidate_ids"),
        )
        object.__setattr__(
            self,
            "rejected_candidate_ids",
            _ids(self.rejected_candidate_ids, "rejected_candidate_ids"),
        )
        if self.margin is not None:
            if isinstance(self.margin, bool) or not isinstance(self.margin, int) or self.margin < 0:
                raise SupportBehaviorPlacementError(
                    "margin must be a non-negative integer or None"
                )
        if self.schema != SUPPORT_PLACEMENT_DECISION_SCHEMA:
            raise SupportBehaviorPlacementError("unsupported support placement decision schema")
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

        if self.disposition is SupportPlacementDisposition.ADMITTED:
            if not self.selected_candidate_id or not self.target_path:
                raise SupportBehaviorPlacementError(
                    "admission requires selected_candidate_id and target_path"
                )
            if not self.placement_paths:
                raise SupportBehaviorPlacementError(
                    "admission requires exact placement_paths from selection"
                )
            if self.target_path not in self.placement_paths:
                raise SupportBehaviorPlacementError(
                    "admitted target_path must be one of the placement_paths"
                )
            if not self.proof_receipt_ids:
                raise SupportBehaviorPlacementError(
                    "admission requires proof_receipt_ids"
                )
            if self.action is SupportPlacementAction.NONE:
                raise SupportBehaviorPlacementError(
                    "admission requires REUSE_EXISTING or PLACE_NEW action"
                )
        else:
            if self.selected_candidate_id or self.target_path or self.placement_paths:
                raise SupportBehaviorPlacementError(
                    "non-admitted decisions cannot select a target or placement paths"
                )
            if self.action is not SupportPlacementAction.NONE:
                raise SupportBehaviorPlacementError(
                    "non-admitted decisions cannot carry a placement action"
                )
            if self.proof_receipt_ids:
                raise SupportBehaviorPlacementError(
                    "non-admitted decisions cannot carry proof authority"
                )

    @property
    def admitted(self) -> bool:
        return self.disposition is SupportPlacementDisposition.ADMITTED

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "disposition": self.disposition.value,
                "behavior_id": self.behavior_id,
                "candidate_set_id": self.candidate_set_id,
                "selected_candidate_id": self.selected_candidate_id,
                "action": self.action.value,
                "target_path": self.target_path,
                "placement_paths": list(self.placement_paths),
                "reason_codes": list(self.reason_codes),
                "proof_receipt_ids": list(self.proof_receipt_ids),
                "roots": self.roots.content_id,
                "producer_id": self.producer_id,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "disposition": self.disposition.value,
            "behavior_id": self.behavior_id,
            "candidate_set_id": self.candidate_set_id,
            "selected_candidate_id": self.selected_candidate_id,
            "action": self.action.value,
            "target_path": self.target_path,
            "placement_paths": list(self.placement_paths),
            "reason_codes": list(self.reason_codes),
            "proof_receipt_ids": list(self.proof_receipt_ids),
            "evidence_refs": list(self.evidence_refs),
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "margin": self.margin,
            "site_placement_decision_ref": self.site_placement_decision_ref,
            "producer_id": self.producer_id,
            "roots": self.roots.to_dict(),
            "content_id": self.content_id,
        }


# ---------------------------------------------------------------------------
# Candidate-set identity
# ---------------------------------------------------------------------------


def support_placement_candidate_set_identity(
    candidates: Sequence[SupportPlacementCandidate],
) -> str:
    """Bind the complete support-placement candidate set identity."""

    if (
        isinstance(candidates, (str, bytes, bytearray))
        or not isinstance(candidates, Sequence)
        or not candidates
    ):
        raise SupportBehaviorPlacementError(
            "candidates must be a non-empty SupportPlacementCandidate sequence"
        )
    rows = tuple(candidates)
    if not all(isinstance(item, SupportPlacementCandidate) for item in rows):
        raise SupportBehaviorPlacementError(
            "candidates must contain SupportPlacementCandidate values"
        )
    ids = tuple(sorted(item.candidate_id for item in rows))
    if len(set(ids)) != len(ids):
        raise SupportBehaviorPlacementError("candidate ids must be unique")
    if any(item.roots != rows[0].roots for item in rows):
        raise SupportBehaviorPlacementError(
            "candidates must bind one exact authority-root set"
        )
    return content_identity(
        {
            "schema": "support-placement-candidate-set@1",
            "candidate_ids": list(ids),
            "behavior_ids": list(sorted({item.behavior_id for item in rows})),
            "roots": rows[0].roots.content_id,
        }
    )


# ---------------------------------------------------------------------------
# Placement engine
# ---------------------------------------------------------------------------


class SupportBehaviorPlacement:
    """Enumerate and admit exact placement for new support behavior."""

    def __init__(self, *, minimum_margin: int = DEFAULT_MINIMUM_MARGIN) -> None:
        if (
            isinstance(minimum_margin, bool)
            or not isinstance(minimum_margin, int)
            or not 1 <= minimum_margin <= SCORE_SCALE
        ):
            raise SupportBehaviorPlacementError(
                "minimum_margin must be a positive bounded integer"
            )
        self.minimum_margin = minimum_margin

    # -- enumeration -------------------------------------------------------

    def enumerate_candidates(
        self,
        behavior: RequiredBehaviorContract,
        anchors: Sequence[PlacementAnchor] = (),
        *,
        existing_implementations: Sequence[ExistingImplementationFact] = (),
        default_proved: bool = False,
    ) -> tuple[SupportPlacementCandidate, ...]:
        """Build a bounded candidate set from anchors and reusable implementations.

        Enumeration is deterministic and does not admit a placement.  Soft
        nomination sources (vector/KG/LLM) are dropped rather than promoted.
        """

        if not isinstance(behavior, RequiredBehaviorContract):
            raise SupportBehaviorPlacementError(
                "behavior must be RequiredBehaviorContract"
            )
        if behavior.implementation_hypothesis:
            raise SupportBehaviorPlacementAuthorityError(
                "implementation hypotheses cannot authorize placement enumeration"
            )

        if isinstance(anchors, (str, bytes, bytearray)) or not isinstance(
            anchors, Sequence
        ):
            raise SupportBehaviorPlacementError("anchors must be a sequence")
        if isinstance(existing_implementations, (str, bytes, bytearray)) or not isinstance(
            existing_implementations, Sequence
        ):
            raise SupportBehaviorPlacementError(
                "existing_implementations must be a sequence"
            )

        anchor_rows = tuple(anchors)
        reuse_rows = tuple(existing_implementations)
        if len(anchor_rows) > MAX_ANCHORS or len(reuse_rows) > MAX_ANCHORS:
            raise SupportBehaviorPlacementError("anchor enumeration exceeds bound")
        if not all(isinstance(item, PlacementAnchor) for item in anchor_rows):
            raise SupportBehaviorPlacementError(
                "anchors must contain PlacementAnchor values"
            )
        if not all(
            isinstance(item, ExistingImplementationFact) for item in reuse_rows
        ):
            raise SupportBehaviorPlacementError(
                "existing_implementations must contain ExistingImplementationFact values"
            )

        candidates: list[SupportPlacementCandidate] = []
        for fact in sorted(reuse_rows, key=lambda item: item.implementation_id):
            if fact.roots != behavior.roots:
                continue
            if _nomination_forbidden(fact.nomination_source):
                continue
            # Only reuse implementations for the same subject symbol.
            if fact.subject_symbol_id != behavior.subject_symbol_id:
                continue
            candidates.append(
                SupportPlacementCandidate(
                    roots=behavior.roots,
                    candidate_id=f"reuse:{fact.implementation_id}",
                    behavior_id=behavior.behavior_id,
                    subject_symbol_id=behavior.subject_symbol_id,
                    anchor_kind=PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION,
                    anchor_id=fact.implementation_id,
                    target_path=fact.target_path,
                    placement_paths=fact.placement_paths,
                    owner_id=fact.owner_id,
                    module_owner_id=fact.module_owner_id,
                    language_runtime=fact.language_runtime,
                    proof_receipt_ids=fact.proof_receipt_ids,
                    evidence_refs=fact.evidence_refs,
                    nomination_source=fact.nomination_source,
                    ownership_exact=bool(fact.owner_id),
                    owner_unambiguous=bool(fact.owner_id),
                    visibility_route_satisfiable=default_proved or fact.admissible,
                    dependency_direction_legal=default_proved or fact.admissible,
                    dependency_acyclic=default_proved or fact.admissible,
                    registration_export_di_wiring_satisfiable=default_proved
                    or fact.admissible,
                    capability_supported=default_proved or fact.admissible,
                    effect_supported=default_proved or fact.admissible,
                    resource_supported=default_proved or fact.admissible,
                    memory_supported=default_proved or fact.admissible,
                    mutation_authority_exact=default_proved or fact.admissible,
                    behavior_contract_fit=fact.behavior_contract_fit,
                    behavior_proved=fact.behavior_contract_fit and bool(fact.proof_receipt_ids),
                    lifecycle_supported=default_proved or fact.admissible,
                    site_placement_admitted=fact.site_placement_admitted,
                    is_reuse=True,
                )
            )

        for anchor in sorted(anchor_rows, key=lambda item: (item.kind.value, item.anchor_id)):
            if anchor.roots != behavior.roots:
                continue
            if _nomination_forbidden(anchor.nomination_source):
                continue
            if anchor.kind not in _STRUCTURAL_ANCHOR_KINDS:
                continue
            candidates.append(
                SupportPlacementCandidate(
                    roots=behavior.roots,
                    candidate_id=f"anchor:{anchor.kind.value}:{anchor.anchor_id}",
                    behavior_id=behavior.behavior_id,
                    subject_symbol_id=behavior.subject_symbol_id,
                    anchor_kind=anchor.kind,
                    anchor_id=anchor.anchor_id,
                    target_path=anchor.target_path,
                    placement_paths=(anchor.target_path,),
                    owner_id=anchor.owner_id,
                    module_owner_id=anchor.module_owner_id,
                    language_runtime=anchor.language_runtime,
                    evidence_refs=anchor.evidence_refs,
                    interface_id=anchor.interface_id,
                    declaration_id=anchor.declaration_id,
                    nomination_source=anchor.nomination_source,
                    ownership_exact=bool(anchor.owner_id),
                    owner_unambiguous=bool(anchor.owner_id),
                    visibility_route_satisfiable=default_proved,
                    dependency_direction_legal=default_proved,
                    dependency_acyclic=default_proved,
                    registration_export_di_wiring_satisfiable=default_proved
                    or bool(
                        anchor.registration_route_id
                        or anchor.export_route_id
                        or anchor.di_wiring_route_id
                    ),
                    capability_supported=default_proved,
                    effect_supported=default_proved,
                    resource_supported=default_proved,
                    memory_supported=default_proved,
                    mutation_authority_exact=default_proved,
                    behavior_contract_fit=default_proved,
                    behavior_proved=default_proved,
                    lifecycle_supported=default_proved,
                    is_reuse=False,
                )
            )

        if len(candidates) > MAX_CANDIDATES:
            raise SupportBehaviorPlacementError("enumerated candidate set exceeds bound")
        return tuple(candidates)

    # -- admission ---------------------------------------------------------

    def decide(
        self,
        behavior: RequiredBehaviorContract,
        candidates: Sequence[SupportPlacementCandidate],
        *,
        minimum_margin: int | None = None,
        site_decisions: Mapping[str, PlacementDecision] | None = None,
        proof_receipts: Sequence[ProofReceipt] = (),
        dependency_graph_id: str = "",
        write_policy_paths: Sequence[str] | None = None,
    ) -> SupportPlacementDecision:
        """Admit exactly one proved placement site or abstain."""

        try:
            return self._decide(
                behavior,
                candidates,
                minimum_margin=minimum_margin,
                site_decisions=site_decisions,
                proof_receipts=proof_receipts,
                dependency_graph_id=dependency_graph_id,
                write_policy_paths=write_policy_paths,
            )
        except (SupportBehaviorPlacementError, ValueError):
            roots = (
                behavior.roots
                if isinstance(behavior, RequiredBehaviorContract)
                else PropagationAuthorityRoots(
                    repository_id="repository:invalid",
                    base_forest_id="forest:base",
                    base_tree_id="tree:base",
                    base_overlay_id="overlay:base",
                    candidate_forest_id="forest:candidate",
                    candidate_tree_id="tree:candidate",
                    candidate_overlay_id="overlay:candidate",
                    graph_id="graph:invalid",
                    index_id="index:invalid",
                    model_id="model:invalid",
                    config_id="config:invalid",
                    translator_id="translator:invalid",
                    toolchain_id="toolchain:invalid",
                    policy_id="policy:invalid",
                )
            )
            behavior_id = (
                behavior.behavior_id
                if isinstance(behavior, RequiredBehaviorContract)
                else "behavior:invalid"
            )
            return SupportPlacementDecision(
                SupportPlacementDisposition.ABSTAINED,
                roots,
                behavior_id,
                "invalid:candidate-set",
                reason_codes=("invalid_admission_input",),
            )

    assess = decide
    evaluate = decide
    place = decide
    admit = decide

    def _decide(
        self,
        behavior: RequiredBehaviorContract,
        candidates: Sequence[SupportPlacementCandidate],
        *,
        minimum_margin: int | None,
        site_decisions: Mapping[str, PlacementDecision] | None,
        proof_receipts: Sequence[ProofReceipt],
        dependency_graph_id: str,
        write_policy_paths: Sequence[str] | None,
    ) -> SupportPlacementDecision:
        if not isinstance(behavior, RequiredBehaviorContract):
            raise SupportBehaviorPlacementError(
                "behavior must be RequiredBehaviorContract"
            )
        if behavior.implementation_hypothesis:
            return SupportPlacementDecision(
                SupportPlacementDisposition.REVIEW_ONLY,
                behavior.roots,
                behavior.behavior_id,
                "review:implementation-hypothesis",
                reason_codes=("implementation_hypothesis_not_authoritative",),
            )
        if not behavior.proof_refs and not any(
            getattr(behavior, name)
            for name in (
                "field_refs",
                "constructor_refs",
                "method_refs",
                "invariant_refs",
                "state_transition_refs",
            )
        ):
            return SupportPlacementDecision(
                SupportPlacementDisposition.ABSTAINED,
                behavior.roots,
                behavior.behavior_id,
                "invalid:behavior",
                reason_codes=("unproved_behavior",),
            )

        rows = self._candidates(candidates)
        if not rows:
            return SupportPlacementDecision(
                SupportPlacementDisposition.ABSTAINED,
                behavior.roots,
                behavior.behavior_id,
                "empty:candidate-set",
                reason_codes=("no_placement_candidates",),
            )
        candidate_set_id = support_placement_candidate_set_identity(rows)
        margin_floor = (
            self.minimum_margin if minimum_margin is None else minimum_margin
        )
        if (
            isinstance(margin_floor, bool)
            or not isinstance(margin_floor, int)
            or not 1 <= margin_floor <= SCORE_SCALE
        ):
            raise SupportBehaviorPlacementError(
                "minimum_margin must be a positive bounded integer"
            )

        receipt_index = self._proof_receipt_index(proof_receipts, behavior)
        site_map = dict(site_decisions or {})
        if not all(isinstance(key, str) for key in site_map):
            raise SupportBehaviorPlacementError(
                "site_decisions keys must be candidate ids"
            )
        if not all(isinstance(value, PlacementDecision) for value in site_map.values()):
            raise SupportBehaviorPlacementError(
                "site_decisions values must be PlacementDecision"
            )

        policy_paths: frozenset[str] | None = None
        if write_policy_paths is not None:
            policy_paths = frozenset(
                _paths(write_policy_paths, "write_policy_paths", required=False)
            )

        graph_id = _text(dependency_graph_id, "dependency_graph_id", required=False)
        if graph_id and graph_id != behavior.roots.graph_id:
            return SupportPlacementDecision(
                SupportPlacementDisposition.ABSTAINED,
                behavior.roots,
                behavior.behavior_id,
                candidate_set_id,
                reason_codes=("dependency_graph_root_mismatch",),
            )

        eligible: list[tuple[SupportPlacementCandidate, tuple[int, ...]]] = []
        rejected: dict[str, tuple[str, ...]] = {}
        rejection_codes: set[str] = set()

        for candidate in rows:
            reasons = self._rejection_reasons(
                behavior,
                candidate,
                receipt_index=receipt_index,
                site_map=site_map,
                policy_paths=policy_paths,
            )
            if reasons:
                rejected[candidate.candidate_id] = reasons
                rejection_codes.update(reasons)
            else:
                eligible.append((candidate, candidate.default_score_vector()))

        eligible_ids = tuple(sorted(item[0].candidate_id for item in eligible))
        rejected_ids = tuple(sorted(rejected))

        if not eligible:
            disposition = SupportPlacementDisposition.ABSTAINED
            # Thin authority / unsupported surfaces surface as review-only.
            review_markers = {
                "unsupported_lifecycle_native_semantics",
                "unproved_behavior",
                "missing_owner",
                "behavior_contract_mismatch",
            }
            if rejection_codes & review_markers and not (
                rejection_codes - review_markers - {"no_eligible_placement_candidate"}
            ):
                # Only when the sole blockers are authority/semantic gaps.
                if rejection_codes <= review_markers:
                    disposition = SupportPlacementDisposition.REVIEW_ONLY
            return SupportPlacementDecision(
                disposition,
                behavior.roots,
                behavior.behavior_id,
                candidate_set_id,
                reason_codes=tuple(
                    rejection_codes or {"no_eligible_placement_candidate"}
                ),
                eligible_candidate_ids=eligible_ids,
                rejected_candidate_ids=rejected_ids,
            )

        # Prefer reuse of existing admissible implementations before new sites.
        reuse_pool = [item for item in eligible if item[0].is_reuse]
        pool = reuse_pool if reuse_pool else eligible
        ordered = sorted(
            pool,
            key=lambda item: (
                tuple(-value for value in item[1]),
                item[0].candidate_id,
            ),
        )
        winner, winner_score = ordered[0]
        margin: int | None = None
        if len(ordered) > 1:
            margin = self._margin(winner_score, ordered[1][1])
            if margin is None:
                return SupportPlacementDecision(
                    SupportPlacementDisposition.AMBIGUOUS,
                    behavior.roots,
                    behavior.behavior_id,
                    candidate_set_id,
                    reason_codes=("rank_tie", "multiple_equal_admissible_sites"),
                    eligible_candidate_ids=tuple(
                        sorted(item[0].candidate_id for item in pool)
                    ),
                    rejected_candidate_ids=rejected_ids,
                )
            if margin < margin_floor:
                return SupportPlacementDecision(
                    SupportPlacementDisposition.AMBIGUOUS,
                    behavior.roots,
                    behavior.behavior_id,
                    candidate_set_id,
                    reason_codes=("insufficient_rank_margin",),
                    eligible_candidate_ids=tuple(
                        sorted(item[0].candidate_id for item in pool)
                    ),
                    rejected_candidate_ids=rejected_ids,
                    margin=margin,
                )

        action = (
            SupportPlacementAction.REUSE_EXISTING
            if winner.is_reuse
            else SupportPlacementAction.PLACE_NEW
        )
        # Selection alone defines exact placement paths.
        placement_paths = winner.placement_paths
        evidence = tuple(
            sorted(
                {
                    *winner.evidence_refs,
                    *behavior.proof_refs,
                    f"candidate-set:{candidate_set_id}",
                    f"behavior:{behavior.behavior_id}",
                }
            )
        )
        return SupportPlacementDecision(
            SupportPlacementDisposition.ADMITTED,
            behavior.roots,
            behavior.behavior_id,
            candidate_set_id,
            selected_candidate_id=winner.candidate_id,
            action=action,
            target_path=winner.target_path,
            placement_paths=placement_paths,
            proof_receipt_ids=winner.proof_receipt_ids,
            evidence_refs=evidence,
            eligible_candidate_ids=eligible_ids,
            rejected_candidate_ids=rejected_ids,
            margin=margin,
            site_placement_decision_ref=winner.site_placement_decision_ref,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _candidates(
        candidates: Sequence[SupportPlacementCandidate],
    ) -> tuple[SupportPlacementCandidate, ...]:
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(
            candidates, Sequence
        ):
            raise SupportBehaviorPlacementError("candidates must be a sequence")
        rows = tuple(candidates)
        if len(rows) > MAX_CANDIDATES:
            raise SupportBehaviorPlacementError("candidate set exceeds bound")
        if not all(isinstance(item, SupportPlacementCandidate) for item in rows):
            raise SupportBehaviorPlacementError(
                "candidates must contain SupportPlacementCandidate values"
            )
        ids = [item.candidate_id for item in rows]
        if len(set(ids)) != len(ids):
            raise SupportBehaviorPlacementError("duplicate placement candidate ids")
        if rows and any(item.roots != rows[0].roots for item in rows):
            raise SupportBehaviorPlacementError(
                "candidates must bind one exact authority-root set"
            )
        return rows

    @staticmethod
    def _proof_receipt_index(
        proof_receipts: Sequence[ProofReceipt],
        behavior: RequiredBehaviorContract,
    ) -> dict[str, ProofReceipt]:
        if isinstance(proof_receipts, (str, bytes, bytearray)) or not isinstance(
            proof_receipts, Sequence
        ):
            raise SupportBehaviorPlacementError("proof_receipts must be a sequence")
        index: dict[str, ProofReceipt] = {}
        for receipt in proof_receipts:
            if not isinstance(receipt, ProofReceipt):
                raise SupportBehaviorPlacementError(
                    "proof_receipts must contain ProofReceipt values"
                )
            if (
                receipt.repository_id != behavior.roots.repository_id
                or receipt.translator_id != behavior.roots.translator_id
                or receipt.toolchain_id != behavior.roots.toolchain_id
                or receipt.policy_id != behavior.roots.policy_id
            ):
                # Stale/cross-root receipts are ignored rather than promoted.
                continue
            # Accept either candidate tree identity for propagation snapshots.
            if receipt.repository_tree_id not in {
                behavior.roots.candidate_tree_id,
                behavior.roots.base_tree_id,
            }:
                continue
            index[receipt.receipt_id] = receipt
            if receipt.obligation_id:
                index.setdefault(receipt.obligation_id, receipt)
        return index

    def _rejection_reasons(
        self,
        behavior: RequiredBehaviorContract,
        candidate: SupportPlacementCandidate,
        *,
        receipt_index: Mapping[str, ProofReceipt],
        site_map: Mapping[str, PlacementDecision],
        policy_paths: frozenset[str] | None,
    ) -> tuple[str, ...]:
        reasons: set[str] = set()

        if candidate.roots != behavior.roots:
            reasons.add("authority_roots_mismatch")
        if candidate.behavior_id != behavior.behavior_id:
            reasons.add("behavior_id_mismatch")
        if candidate.subject_symbol_id != behavior.subject_symbol_id:
            reasons.add("subject_symbol_mismatch")

        if _nomination_forbidden(candidate.nomination_source):
            reasons.add("vector_kg_llm_nomination_forbidden")

        if not candidate.owner_id or not candidate.module_owner_id:
            reasons.add("missing_owner")
        if not candidate.ownership_exact or not candidate.owner_unambiguous:
            reasons.add("missing_owner")

        if candidate.language_runtime not in _SUPPORTED_RUNTIMES:
            reasons.add("language_runtime_unsupported")

        if (
            candidate.generated
            or candidate.vendor
            or candidate.read_only
            or _path_forbidden(candidate.target_path)
            or any(_path_forbidden(path) for path in candidate.placement_paths)
        ):
            reasons.add("generated_vendor_read_only_target")

        if candidate.cross_root_write:
            reasons.add("cross_root_write")
        if policy_paths is not None:
            if any(path not in policy_paths for path in candidate.placement_paths):
                reasons.add("cross_root_write")

        if not candidate.dependency_acyclic or not candidate.dependency_direction_legal:
            reasons.add("dependency_cycle")

        if (
            candidate.native_semantics_unsupported
            or candidate.lifecycle_native_unsupported
            or not candidate.lifecycle_supported
        ):
            reasons.add("unsupported_lifecycle_native_semantics")

        if not candidate.visibility_route_satisfiable:
            reasons.add("visibility_route_unsatisfied")
        if not candidate.registration_export_di_wiring_satisfiable:
            reasons.add("registration_export_di_unsatisfied")

        if not (
            candidate.capability_supported
            and candidate.effect_supported
            and candidate.resource_supported
            and candidate.memory_supported
        ):
            reasons.add("capability_effect_resource_memory_unsupported")

        if not candidate.mutation_authority_exact:
            reasons.add("mutation_authority_not_exact")

        if not candidate.behavior_contract_fit:
            reasons.add("behavior_contract_mismatch")
        if not candidate.behavior_proved:
            reasons.add("unproved_behavior")

        if not candidate.proof_receipt_ids:
            reasons.add("missing_required_proof_receipt")
        else:
            for receipt_id in candidate.proof_receipt_ids:
                receipt = receipt_index.get(receipt_id)
                if receipt is None:
                    # Allow compact identity-only proof refs when no full
                    # receipts are supplied for the batch (caller may bind
                    # receipt ids without replaying full ProofReceipt objects).
                    if receipt_index:
                        reasons.add("proof_receipt_binding_mismatch")
                    continue
                if receipt.verdict is not ProofVerdict.PROVED:
                    reasons.add("required_proof_not_reconstructed")

        # Optional ImplementationSiteAdmissibility join for non-reuse sites.
        site = site_map.get(candidate.candidate_id)
        if site is not None:
            if (
                site.disposition is not PlacementDisposition.ADMITTED
                or site.selected_candidate_id
                and site.target_path != candidate.target_path
            ):
                reasons.add("site_not_admitted")
            elif site.target_path and site.target_path != candidate.target_path:
                reasons.add("site_target_path_mismatch")
            else:
                # Admitted site decision reinforces mutation/site authority.
                pass
        elif not candidate.is_reuse and not candidate.site_placement_admitted:
            # New placement still needs an explicit site-admitted fact when no
            # PlacementDecision map entry is provided.
            reasons.add("site_not_admitted")

        if candidate.is_reuse and not candidate.site_placement_admitted and site is None:
            # Reuse still requires the existing implementation to be admissible.
            # site_placement_admitted flag covers that without a full PlacementDecision.
            reasons.add("site_not_admitted")

        # Behavior kind sanity: methods require a host module path; classes may
        # introduce a new module path under the owner package.
        if behavior.kind is BehaviorKind.METHOD and not candidate.target_path.endswith(
            (".py", ".pyi")
        ):
            reasons.add("language_runtime_unsupported")

        return tuple(sorted(reasons))

    @staticmethod
    def _margin(
        first: tuple[int, ...], second: tuple[int, ...]
    ) -> int | None:
        for left, right in zip(first, second):
            if left != right:
                return left - right
        if len(first) != len(second):
            longer, shorter = (
                (first, second) if len(first) > len(second) else (second, first)
            )
            # Non-zero trailing components break the tie.
            for value in longer[len(shorter) :]:
                if value != 0:
                    return abs(value) if first == longer else -abs(value)
        return None


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_MINIMUM_MARGIN",
    "EXISTING_ADMISSIBLE_IMPLEMENTATION",
    "MAX_CANDIDATES",
    "PLACEMENT_ANCHOR_SCHEMA",
    "PRODUCER_ID",
    "SUPPORT_BEHAVIOR_PLACEMENT_INTERFACE",
    "SUPPORT_PLACEMENT_CANDIDATE_SCHEMA",
    "SUPPORT_PLACEMENT_DECISION_SCHEMA",
    "ExistingImplementationFact",
    "PlacementAnchor",
    "PlacementAnchorKind",
    "SupportBehaviorPlacement",
    "SupportBehaviorPlacementAuthorityError",
    "SupportBehaviorPlacementError",
    "SupportPlacementAction",
    "SupportPlacementCandidate",
    "SupportPlacementDecision",
    "SupportPlacementDisposition",
    "support_placement_candidate_set_identity",
]

# Convenience alias matching anchor kind spelling in exports.
EXISTING_ADMISSIBLE_IMPLEMENTATION: Final[PlacementAnchorKind] = (
    PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION
)
