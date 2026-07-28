"""Semantic-roundtrip residual / structural bridge (CBP-110).

Interface: ``CodeProofSrtBridge@1``

Thin adapter that projects PLAT residual catalogs, ``PlateauCodexPacket@1``
handles, and ``StructuralAdmission@1`` receipts into the shared CBP claim,
query, context, and ``CodeEditPacket@1`` path **without**:

* treating heterogeneous SRT methods as interchangeable
* coupling gold IR bodies into proof-cache keys or packet metadata
* relocating semantic promotion authority (e2e loss / holdout gates) into CBP
* rewriting sealed PLAT promotion snapshots

PLAT2 holdout artifacts remain separately preregistered and queryable.

Normative method roles (measured, not interchangeable):

* autoencoder / spaCy → bounded guidance / diagnostics
* SyMAI → orchestration
* Leanstral → proposal teacher
* Hammer / cvc5 / Lean → declared structural gates
* deterministic compiler / IR / decompiler → edit target
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .code_claim_contracts import (
    CLAIM_CATALOG_VERSION,
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    EvidenceTier,
    build_invalidation_selectors,
)
from .code_edit_materialize import (
    bridge_plateau_codex_packet,
)
from .code_edit_packet import (
    CODE_EDIT_PACKET_INTERFACE,
    CodeEditPacket,
    ProverBinding,
)
from .code_property_catalog import SRT_STRUCTURAL_TAGS
from .code_proof_obligations import normalize_residual_refs
from .context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextReference,
    ContextTier,
)
from .formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_cache import (
    ProofCacheKey,
    build_proof_cache_key,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces and schemas
# ---------------------------------------------------------------------------

CODE_PROOF_SRT_BRIDGE_INTERFACE: Final = "CodeProofSrtBridge@1"
CODE_PROOF_SRT_BRIDGE_VERSION: Final = "1"
STRUCTURAL_ADMISSION_INTERFACE: Final = "StructuralAdmission@1"
STRUCTURAL_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/structural-admission@1"
)
PLATEAU_CODEX_PACKET_INTERFACE: Final = "PlateauCodexPacket@1"
PLATEAU_CODEX_PACKET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plateau-codex-packet@1"
)
PLAT_RESIDUAL_CATALOG_INTERFACE: Final = "PlatResidualCatalog@1"
PLAT_RESIDUAL_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plat-residual-catalog@1"
)
PLAT2_HOLDOUT_REGISTRY_INTERFACE: Final = "Plat2HoldoutRegistry@1"
PLAT2_HOLDOUT_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plat2-holdout-registry@1"
)
SRT_GRAPH_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/srt-graph-projection@1"
)
SRT_BRIDGE_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/srt-bridge-projection@1"
)
SRT_CACHE_KEY_HANDLES_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/srt-cache-key-handles@1"
)

# Producer / policy identity for bridge-minted records.
SRT_BRIDGE_PRODUCER_ID: Final = "producer:code-proof-srt-bridge@1"
SRT_BRIDGE_POLICY_ID: Final = "policy:srt-bridge-non-semantic@1"
SRT_BRIDGE_TOOLCHAIN_ID: Final = "toolchain:srt-structural@1"

# Authority doctrine: structural gates and bridge projections never replace
# semantic-roundtrip e2e loss or SRT holdout promotion.
# Preferred semantic name; wire value remains plat2_holdout_promotion_gate
# for content-addressed / historical gate identity.
SRT_HOLDOUT_PROMOTION_GATE: Final = "plat2_holdout_promotion_gate"
PLAT2_HOLDOUT_PROMOTION_GATE: Final = SRT_HOLDOUT_PROMOTION_GATE  # deprecated alias

PROMOTION_AUTHORITIES: Final[tuple[str, ...]] = (
    "semantic_roundtrip_e2e_loss",
    SRT_HOLDOUT_PROMOTION_GATE,
)
STRUCTURAL_SEMANTIC_AUTHORITY: Final = False

# Gold / proof body markers that must never enter cache keys or bridge payloads.
_GOLD_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "gold_ir",
        "gold_ir_body",
        "gold_source",
        "gold_body",
        "full_ir",
        "ir_body",
        "proof_body",
        "proof_text",
        "lean_source",
        "lean_source_body",
        "full_receipt",
        "receipt_body",
        "kernel_proof_body",
        "solver_trace",
        "private_witness",
        "repository_dump",
        "source_dump",
        "raw_source",
    }
)


class CodeProofSrtBridgeError(ValueError):
    """Bridge input is malformed, unsafe, or violates authority doctrine."""


# ---------------------------------------------------------------------------
# Heterogeneous method roles (measured, not interchangeable)
# ---------------------------------------------------------------------------


class SrtMethodRole(str, Enum):
    """Measured role of one SRT method in the plateau / holdout loop."""

    BOUNDED_GUIDANCE_DIAGNOSTICS = "bounded_guidance_diagnostics"
    ORCHESTRATION = "orchestration"
    PROPOSAL_TEACHER = "proposal_teacher"
    STRUCTURAL_GATE = "structural_gate"
    EDIT_TARGET = "edit_target"


# Canonical method → role map.  Methods are not substitutes for each other.
_METHOD_ROLE_BY_ID: Final[Mapping[str, SrtMethodRole]] = MappingProxyType(
    {
        "autoencoder": SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS,
        "spacy": SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS,
        "spaCy": SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS,
        "symai": SrtMethodRole.ORCHESTRATION,
        "SyMAI": SrtMethodRole.ORCHESTRATION,
        "leanstral": SrtMethodRole.PROPOSAL_TEACHER,
        "Leanstral": SrtMethodRole.PROPOSAL_TEACHER,
        "hammer": SrtMethodRole.STRUCTURAL_GATE,
        "Hammer": SrtMethodRole.STRUCTURAL_GATE,
        "cvc5": SrtMethodRole.STRUCTURAL_GATE,
        "lean": SrtMethodRole.STRUCTURAL_GATE,
        "Lean": SrtMethodRole.STRUCTURAL_GATE,
        "compiler": SrtMethodRole.EDIT_TARGET,
        "ir": SrtMethodRole.EDIT_TARGET,
        "decompiler": SrtMethodRole.EDIT_TARGET,
        "deterministic_compiler": SrtMethodRole.EDIT_TARGET,
        "deterministic_ir": SrtMethodRole.EDIT_TARGET,
        "deterministic_decompiler": SrtMethodRole.EDIT_TARGET,
    }
)

# Role → representative method ids (stable for docs / fixtures).
METHOD_ROLES_TABLE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS.value: (
            "autoencoder",
            "spacy",
        ),
        SrtMethodRole.ORCHESTRATION.value: ("symai",),
        SrtMethodRole.PROPOSAL_TEACHER.value: ("leanstral",),
        SrtMethodRole.STRUCTURAL_GATE.value: ("hammer", "cvc5", "lean"),
        SrtMethodRole.EDIT_TARGET.value: (
            "compiler",
            "ir",
            "decompiler",
        ),
    }
)


def resolve_method_role(method_id: str | SrtMethodRole) -> SrtMethodRole:
    """Resolve a measured method role; unknown methods fail closed."""

    if isinstance(method_id, SrtMethodRole):
        return method_id
    text = str(method_id or "").strip()
    if not text:
        raise CodeProofSrtBridgeError("method_id is required")
    role = _METHOD_ROLE_BY_ID.get(text)
    if role is not None:
        return role
    # Case-insensitive fallback for canonical lowercased ids.
    role = _METHOD_ROLE_BY_ID.get(text.lower())
    if role is not None:
        return role
    raise CodeProofSrtBridgeError(
        f"unknown SRT method_id {text!r}; methods are not interchangeable — "
        f"register a measured role or use a known id"
    )


def method_role_description(role: SrtMethodRole | str) -> str:
    """Human-readable description of a measured role."""

    resolved = (
        role
        if isinstance(role, SrtMethodRole)
        else SrtMethodRole(str(role))
    )
    return {
        SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS: (
            "Bounded guidance and diagnostics only; never structural admission "
            "or semantic promotion authority."
        ),
        SrtMethodRole.ORCHESTRATION: (
            "Orchestrates residual → gate → packet flow; does not admit "
            "semantics or substitute for e2e loss."
        ),
        SrtMethodRole.PROPOSAL_TEACHER: (
            "Untrusted proposal teacher for candidate repairs; kernel/solver "
            "gates remain independent."
        ),
        SrtMethodRole.STRUCTURAL_GATE: (
            "Declared structural gate (Hammer/cvc5/Lean); semantic_authority="
            "false; never replaces SRT e2e loss or holdout promotion."
        ),
        SrtMethodRole.EDIT_TARGET: (
            "Deterministic compiler / IR / decompiler is the sole edit target "
            "for supervisor CodeEditPacket materialization."
        ),
    }[resolved]


# ---------------------------------------------------------------------------
# Gold-body rejection and handle helpers
# ---------------------------------------------------------------------------


def _sorted_unique(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(sorted(out))


def _norm_text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if required and not text:
        raise CodeProofSrtBridgeError(f"{field_name} is required")
    return text


def reject_gold_ir_bodies(payload: Mapping[str, Any], *, where: str) -> None:
    """Fail closed if gold IR / proof bodies appear in a bridge payload.

    Used for residual catalogs, plateau packets, structural admissions, and
    cache-key handle construction so gold never enters CBP keys or metadata.
    """

    if not isinstance(payload, Mapping):
        raise CodeProofSrtBridgeError(f"{where} must be a mapping")
    for key, raw in payload.items():
        key_l = str(key).lower()
        if key_l in _GOLD_BODY_MARKERS or any(
            marker in key_l for marker in _GOLD_BODY_MARKERS
        ):
            # Allow explicit exclusion flags such as gold_ir_excluded.
            if key_l.endswith("_excluded") or key_l.endswith("_rejected"):
                continue
            if key_l in {"gold_ir_excluded", "gold_bodies_rejected"}:
                continue
            raise CodeProofSrtBridgeError(
                f"{where} must not couple gold IR / proof bodies into bridge "
                f"projections or cache keys ({key})"
            )
        if isinstance(raw, Mapping):
            reject_gold_ir_bodies(raw, where=where)
        elif isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            for item in raw:
                if isinstance(item, Mapping):
                    reject_gold_ir_bodies(item, where=where)


def _extract_handles(
    payload: Mapping[str, Any],
    *keys: str,
) -> tuple[str, ...]:
    collected: list[str] = []
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes, bytearray)):
            collected.append(str(raw))
            continue
        if isinstance(raw, Sequence):
            for item in raw:
                if isinstance(item, Mapping):
                    handle = (
                        item.get("residual_ref_id")
                        or item.get("residual_id")
                        or item.get("id")
                        or item.get("handle")
                        or item.get("artifact_id")
                        or ""
                    )
                    if handle:
                        collected.append(str(handle))
                else:
                    collected.append(str(item))
    return _sorted_unique(collected)


# ---------------------------------------------------------------------------
# StructuralAdmission@1 — non-semantic structural gate receipt
# ---------------------------------------------------------------------------


class StructuralAdmissionDisposition(str, Enum):
    """Outcome of a declared structural gate (non-semantic)."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"
    ERROR = "error"


@dataclass(frozen=True)
class StructuralAdmission:
    """Content-addressed structural gate receipt (``StructuralAdmission@1``).

    Always carries ``semantic_authority=False``.  Structural admission is a
    declared gate over residual facets / tags — it never substitutes for
    semantic-roundtrip e2e loss or PLAT2 holdout promotion.
    """

    residual_ref_ids: tuple[str, ...]
    structural_tags: tuple[str, ...]
    disposition: StructuralAdmissionDisposition
    gate_method_ids: tuple[str, ...] = ()
    repository_tree_id: str = ""
    repository_id: str = ""
    property_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    semantic_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residual_ref_ids",
            normalize_residual_refs(self.residual_ref_ids),
        )
        object.__setattr__(
            self, "structural_tags", _sorted_unique(self.structural_tags)
        )
        object.__setattr__(
            self, "gate_method_ids", _sorted_unique(self.gate_method_ids)
        )
        object.__setattr__(
            self, "property_ids", _sorted_unique(self.property_ids)
        )
        object.__setattr__(
            self, "obligation_ids", _sorted_unique(self.obligation_ids)
        )
        object.__setattr__(
            self, "reason_codes", _sorted_unique(self.reason_codes)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _norm_text(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "receipt_id",
            _norm_text(self.receipt_id, field_name="receipt_id"),
        )
        disposition = self.disposition
        if not isinstance(disposition, StructuralAdmissionDisposition):
            disposition = StructuralAdmissionDisposition(str(disposition))
        object.__setattr__(self, "disposition", disposition)
        # Doctrine: structural receipts are never semantic authority.
        object.__setattr__(self, "semantic_authority", False)
        if not isinstance(self.metadata, Mapping):
            raise CodeProofSrtBridgeError("metadata must be a mapping")
        reject_gold_ir_bodies(self.metadata, where="StructuralAdmission.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        # Gate methods must all resolve to STRUCTURAL_GATE.
        for method_id in self.gate_method_ids:
            role = resolve_method_role(method_id)
            if role is not SrtMethodRole.STRUCTURAL_GATE:
                raise CodeProofSrtBridgeError(
                    f"gate_method_ids entry {method_id!r} has role "
                    f"{role.value}; only structural_gate methods may appear "
                    f"on StructuralAdmission"
                )
        # Structural tags should be known SRT tags when present.
        unknown = set(self.structural_tags) - set(SRT_STRUCTURAL_TAGS)
        if unknown and not self.metadata.get("allow_unknown_structural_tags"):
            raise CodeProofSrtBridgeError(
                f"unknown structural tags: {sorted(unknown)}; "
                f"known tags: {list(SRT_STRUCTURAL_TAGS)}"
            )

    @property
    def interface(self) -> str:
        return STRUCTURAL_ADMISSION_INTERFACE

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def admission_id(self) -> str:
        return self.receipt_id or self.content_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STRUCTURAL_ADMISSION_SCHEMA,
            "interface": STRUCTURAL_ADMISSION_INTERFACE,
            "residual_ref_ids": list(self.residual_ref_ids),
            "structural_tags": list(self.structural_tags),
            "disposition": self.disposition.value,
            "gate_method_ids": list(self.gate_method_ids),
            "repository_tree_id": self.repository_tree_id,
            "repository_id": self.repository_id,
            "property_ids": list(self.property_ids),
            "obligation_ids": list(self.obligation_ids),
            "receipt_id": self.receipt_id,
            "reason_codes": list(self.reason_codes),
            "semantic_authority": False,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuralAdmission":
        if not isinstance(payload, Mapping):
            raise CodeProofSrtBridgeError("structural admission must be a mapping")
        reject_gold_ir_bodies(payload, where="StructuralAdmission")
        schema = payload.get("schema")
        if schema not in (None, "", STRUCTURAL_ADMISSION_SCHEMA):
            raise CodeProofSrtBridgeError(
                f"unsupported StructuralAdmission schema; use {STRUCTURAL_ADMISSION_SCHEMA}"
            )
        return cls(
            residual_ref_ids=tuple(
                payload.get("residual_ref_ids")
                or payload.get("residual_ids")
                or ()
            ),
            structural_tags=tuple(payload.get("structural_tags") or ()),
            disposition=str(
                payload.get("disposition")
                or StructuralAdmissionDisposition.NOT_MEASURED.value
            ),
            gate_method_ids=tuple(payload.get("gate_method_ids") or ()),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            property_ids=tuple(payload.get("property_ids") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            receipt_id=str(payload.get("receipt_id") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            semantic_authority=False,
            metadata=dict(payload.get("metadata") or {}),
        )


def build_structural_admission(
    *,
    residual_ref_ids: Sequence[Any] = (),
    structural_tags: Sequence[str] = (),
    disposition: StructuralAdmissionDisposition | str = (
        StructuralAdmissionDisposition.ADMITTED
    ),
    gate_method_ids: Sequence[str] = ("hammer", "cvc5", "lean"),
    repository_tree_id: str = "",
    repository_id: str = "",
    property_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    receipt_id: str = "",
    reason_codes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> StructuralAdmission:
    """Build a StructuralAdmission@1 receipt (semantic_authority forced false)."""

    return StructuralAdmission(
        residual_ref_ids=tuple(residual_ref_ids),
        structural_tags=tuple(structural_tags),
        disposition=disposition,  # type: ignore[arg-type]
        gate_method_ids=tuple(gate_method_ids),
        repository_tree_id=repository_tree_id,
        repository_id=repository_id,
        property_ids=tuple(property_ids),
        obligation_ids=tuple(obligation_ids),
        receipt_id=receipt_id,
        reason_codes=tuple(reason_codes),
        semantic_authority=False,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# PLAT residual catalog entry (handles + facets only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualCatalogEntry:
    """One PLAT residual-catalog row projected as handles + facets.

    Gold IR bodies are forbidden.  Entries bind residual ref ids, optional
    structural tags, counterexample handles, and predicted edit paths.
    """

    residual_ref_id: str
    residual_kind: str = "structural"
    structural_tags: tuple[str, ...] = ()
    property_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    counterexample_ref_ids: tuple[str, ...] = ()
    predicted_files: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    status: str = ClaimStatus.OPEN.value
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residual_ref_id",
            _norm_text(self.residual_ref_id, field_name="residual_ref_id", required=True),
        )
        # Validate handle shape via residual normalizer.
        normalize_residual_refs((self.residual_ref_id,))
        object.__setattr__(
            self,
            "residual_kind",
            _norm_text(self.residual_kind, field_name="residual_kind") or "structural",
        )
        for name in (
            "structural_tags",
            "property_ids",
            "claim_ids",
            "obligation_ids",
            "counterexample_ref_ids",
            "predicted_files",
            "assumption_ids",
        ):
            object.__setattr__(self, name, _sorted_unique(getattr(self, name)))
        object.__setattr__(
            self,
            "status",
            _norm_text(self.status, field_name="status") or ClaimStatus.OPEN.value,
        )
        object.__setattr__(
            self, "summary", _norm_text(self.summary, field_name="summary")
        )
        if not isinstance(self.metadata, Mapping):
            raise CodeProofSrtBridgeError("metadata must be a mapping")
        reject_gold_ir_bodies(self.metadata, where="ResidualCatalogEntry.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_ref_id": self.residual_ref_id,
            "residual_kind": self.residual_kind,
            "structural_tags": list(self.structural_tags),
            "property_ids": list(self.property_ids),
            "claim_ids": list(self.claim_ids),
            "obligation_ids": list(self.obligation_ids),
            "counterexample_ref_ids": list(self.counterexample_ref_ids),
            "predicted_files": list(self.predicted_files),
            "assumption_ids": list(self.assumption_ids),
            "status": self.status,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualCatalogEntry":
        if not isinstance(payload, Mapping):
            raise CodeProofSrtBridgeError("residual catalog entry must be a mapping")
        reject_gold_ir_bodies(payload, where="ResidualCatalogEntry")
        residual_id = str(
            payload.get("residual_ref_id")
            or payload.get("residual_id")
            or payload.get("id")
            or ""
        )
        return cls(
            residual_ref_id=residual_id,
            residual_kind=str(payload.get("residual_kind") or "structural"),
            structural_tags=tuple(payload.get("structural_tags") or ()),
            property_ids=tuple(payload.get("property_ids") or ()),
            claim_ids=tuple(payload.get("claim_ids") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            counterexample_ref_ids=tuple(
                payload.get("counterexample_ref_ids")
                or payload.get("counterexample_ids")
                or ()
            ),
            predicted_files=tuple(
                payload.get("predicted_files") or payload.get("paths") or ()
            ),
            assumption_ids=tuple(
                payload.get("assumption_ids") or payload.get("assumptions") or ()
            ),
            status=str(payload.get("status") or ClaimStatus.OPEN.value),
            summary=str(payload.get("summary") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SrtResidualCatalog:
    """Content-addressed SRT residual catalog (handles only)."""

    entries: tuple[ResidualCatalogEntry, ...]
    catalog_id: str = ""
    repository_tree_id: str = ""
    repository_id: str = ""
    plateau_packet_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        seen: set[str] = set()
        ordered: list[ResidualCatalogEntry] = []
        for entry in entries:
            if not isinstance(entry, ResidualCatalogEntry):
                if isinstance(entry, Mapping):
                    entry = ResidualCatalogEntry.from_dict(entry)
                else:
                    raise CodeProofSrtBridgeError(
                        "entries must be ResidualCatalogEntry instances"
                    )
            if entry.residual_ref_id in seen:
                raise CodeProofSrtBridgeError(
                    f"duplicate residual_ref_id: {entry.residual_ref_id}"
                )
            seen.add(entry.residual_ref_id)
            ordered.append(entry)
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(ordered, key=lambda e: e.residual_ref_id)),
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _norm_text(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "plateau_packet_id",
            _norm_text(self.plateau_packet_id, field_name="plateau_packet_id"),
        )
        if not isinstance(self.metadata, Mapping):
            raise CodeProofSrtBridgeError("metadata must be a mapping")
        reject_gold_ir_bodies(self.metadata, where="SrtResidualCatalog.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        derived = content_identity(self._identity_payload())
        claimed = _norm_text(self.catalog_id, field_name="catalog_id")
        if claimed and claimed != derived:
            raise CodeProofSrtBridgeError("catalog_id does not match content")
        object.__setattr__(self, "catalog_id", claimed or derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PLAT_RESIDUAL_CATALOG_SCHEMA,
            "interface": PLAT_RESIDUAL_CATALOG_INTERFACE,
            "repository_tree_id": self.repository_tree_id,
            "repository_id": self.repository_id,
            "plateau_packet_id": self.plateau_packet_id,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": dict(self.metadata),
        }

    @property
    def interface(self) -> str:
        return PLAT_RESIDUAL_CATALOG_INTERFACE

    def residual_ref_ids(self) -> tuple[str, ...]:
        return tuple(e.residual_ref_id for e in self.entries)

    def get(self, residual_ref_id: str) -> ResidualCatalogEntry | None:
        for entry in self.entries:
            if entry.residual_ref_id == residual_ref_id:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["catalog_id"] = self.catalog_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SrtResidualCatalog":
        if not isinstance(payload, Mapping):
            raise CodeProofSrtBridgeError("residual catalog must be a mapping")
        reject_gold_ir_bodies(payload, where="SrtResidualCatalog")
        raw_entries = payload.get("entries") or ()
        entries = tuple(
            ResidualCatalogEntry.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in raw_entries
        )
        return cls(
            entries=entries,
            catalog_id=str(payload.get("catalog_id") or ""),
            repository_tree_id=str(
                payload.get("repository_tree_id")
                or payload.get("source_tree_id")
                or ""
            ),
            repository_id=str(payload.get("repository_id") or ""),
            plateau_packet_id=str(
                payload.get("plateau_packet_id")
                or payload.get("packet_id")
                or ""
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


def build_srt_residual_catalog(
    entries: Sequence[ResidualCatalogEntry | Mapping[str, Any]],
    *,
    repository_tree_id: str = "",
    repository_id: str = "",
    plateau_packet_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> SrtResidualCatalog:
    """Build a PLAT residual catalog from entry mappings or objects."""

    normalized: list[ResidualCatalogEntry] = []
    for item in entries:
        if isinstance(item, ResidualCatalogEntry):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(ResidualCatalogEntry.from_dict(item))
        else:
            raise CodeProofSrtBridgeError(
                "entries must be ResidualCatalogEntry or mappings"
            )
    return SrtResidualCatalog(
        entries=tuple(normalized),
        repository_tree_id=repository_tree_id,
        repository_id=repository_id,
        plateau_packet_id=plateau_packet_id,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# SRT holdout registry (separate from residual catalog)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtHoldoutArtifact:
    """One preregistered SRT holdout artifact (handles only)."""

    artifact_id: str
    holdout_split: str
    residual_ref_ids: tuple[str, ...] = ()
    property_ids: tuple[str, ...] = ()
    metric_ids: tuple[str, ...] = ()
    repository_tree_id: str = ""
    preregistered: bool = True
    queryable: bool = True
    promotion_gate: str = SRT_HOLDOUT_PROMOTION_GATE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _norm_text(self.artifact_id, field_name="artifact_id", required=True),
        )
        object.__setattr__(
            self,
            "holdout_split",
            _norm_text(self.holdout_split, field_name="holdout_split", required=True),
        )
        object.__setattr__(
            self,
            "residual_ref_ids",
            normalize_residual_refs(self.residual_ref_ids),
        )
        object.__setattr__(
            self, "property_ids", _sorted_unique(self.property_ids)
        )
        object.__setattr__(
            self, "metric_ids", _sorted_unique(self.metric_ids)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(self, "preregistered", bool(self.preregistered))
        object.__setattr__(self, "queryable", bool(self.queryable))
        object.__setattr__(
            self,
            "promotion_gate",
            _norm_text(self.promotion_gate, field_name="promotion_gate")
            or SRT_HOLDOUT_PROMOTION_GATE,
        )
        if not isinstance(self.metadata, Mapping):
            raise CodeProofSrtBridgeError("metadata must be a mapping")
        reject_gold_ir_bodies(self.metadata, where="SrtHoldoutArtifact.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "holdout_split": self.holdout_split,
            "residual_ref_ids": list(self.residual_ref_ids),
            "property_ids": list(self.property_ids),
            "metric_ids": list(self.metric_ids),
            "repository_tree_id": self.repository_tree_id,
            "preregistered": self.preregistered,
            "queryable": self.queryable,
            "promotion_gate": self.promotion_gate,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SrtHoldoutArtifact":
        if not isinstance(payload, Mapping):
            raise CodeProofSrtBridgeError("holdout artifact must be a mapping")
        reject_gold_ir_bodies(payload, where="SrtHoldoutArtifact")
        return cls(
            artifact_id=str(
                payload.get("artifact_id")
                or payload.get("id")
                or payload.get("holdout_id")
                or ""
            ),
            holdout_split=str(
                payload.get("holdout_split")
                or payload.get("split")
                or "holdout"
            ),
            residual_ref_ids=tuple(
                payload.get("residual_ref_ids") or payload.get("residual_ids") or ()
            ),
            property_ids=tuple(payload.get("property_ids") or ()),
            metric_ids=tuple(payload.get("metric_ids") or ()),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            preregistered=bool(payload.get("preregistered", True)),
            queryable=bool(payload.get("queryable", True)),
            promotion_gate=str(
                payload.get("promotion_gate") or SRT_HOLDOUT_PROMOTION_GATE
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class SrtHoldoutRegistry:
    """In-memory preregistered SRT holdout registry (separate from residual catalog).

    Holdout artifacts are queryable by artifact id, split, residual, or
    property.  Registration is additive and fail-closed on gold bodies.
    """

    _artifacts: MutableMapping[str, SrtHoldoutArtifact] = field(
        default_factory=dict
    )

    @property
    def interface(self) -> str:
        return PLAT2_HOLDOUT_REGISTRY_INTERFACE

    def register(
        self, artifact: SrtHoldoutArtifact | Mapping[str, Any]
    ) -> SrtHoldoutArtifact:
        if isinstance(artifact, Mapping):
            artifact = SrtHoldoutArtifact.from_dict(artifact)
        if not isinstance(artifact, SrtHoldoutArtifact):
            raise CodeProofSrtBridgeError(
                "artifact must be SrtHoldoutArtifact or mapping"
            )
        if not artifact.preregistered:
            raise CodeProofSrtBridgeError(
                "SRT holdout artifacts must be preregistered before query"
            )
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing.to_dict() != artifact.to_dict():
            raise CodeProofSrtBridgeError(
                f"holdout artifact already registered with different content: "
                f"{artifact.artifact_id}"
            )
        self._artifacts[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> SrtHoldoutArtifact | None:
        return self._artifacts.get(str(artifact_id or "").strip())

    def require(self, artifact_id: str) -> SrtHoldoutArtifact:
        found = self.get(artifact_id)
        if found is None:
            raise CodeProofSrtBridgeError(
                f"unknown SRT holdout artifact: {artifact_id!r}"
            )
        return found

    def query(
        self,
        *,
        artifact_id: str = "",
        holdout_split: str = "",
        residual_ref_id: str = "",
        property_id: str = "",
        queryable_only: bool = True,
    ) -> tuple[SrtHoldoutArtifact, ...]:
        """Query preregistered holdout artifacts (never mixes training residuals)."""

        results: list[SrtHoldoutArtifact] = []
        for artifact in self._artifacts.values():
            if queryable_only and not artifact.queryable:
                continue
            if artifact_id and artifact.artifact_id != artifact_id:
                continue
            if holdout_split and artifact.holdout_split != holdout_split:
                continue
            if residual_ref_id and residual_ref_id not in artifact.residual_ref_ids:
                continue
            if property_id and property_id not in artifact.property_ids:
                continue
            results.append(artifact)
        return tuple(sorted(results, key=lambda a: a.artifact_id))

    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAT2_HOLDOUT_REGISTRY_SCHEMA,
            "interface": PLAT2_HOLDOUT_REGISTRY_INTERFACE,
            "artifacts": [a.to_dict() for a in self.query(queryable_only=False)],
        }


# ---------------------------------------------------------------------------
# Cache-key handles (exclude gold IR bodies)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtCacheKeyHandles:
    """Compact handles that may enter a proof-cache key for SRT residuals.

    Gold IR bodies are never included.  Only residual ref handles, obligation
    ids, structural tags, tree id, and gate identity digests participate.
    """

    residual_ref_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...] = ()
    property_ids: tuple[str, ...] = ()
    structural_tags: tuple[str, ...] = ()
    repository_tree_id: str = ""
    gate_method_ids: tuple[str, ...] = ()
    catalog_version: str = CLAIM_CATALOG_VERSION
    gold_ir_excluded: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residual_ref_ids",
            normalize_residual_refs(self.residual_ref_ids),
        )
        for name in (
            "obligation_ids",
            "property_ids",
            "structural_tags",
            "gate_method_ids",
        ):
            object.__setattr__(self, name, _sorted_unique(getattr(self, name)))
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            _norm_text(self.catalog_version, field_name="catalog_version")
            or CLAIM_CATALOG_VERSION,
        )
        object.__setattr__(self, "gold_ir_excluded", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SRT_CACHE_KEY_HANDLES_SCHEMA,
            "residual_ref_ids": list(self.residual_ref_ids),
            "obligation_ids": list(self.obligation_ids),
            "property_ids": list(self.property_ids),
            "structural_tags": list(self.structural_tags),
            "repository_tree_id": self.repository_tree_id,
            "gate_method_ids": list(self.gate_method_ids),
            "catalog_version": self.catalog_version,
            "gold_ir_excluded": True,
        }

    def premise_handles(self) -> tuple[str, ...]:
        """Handles safe to place in ``ProofCacheKey.premises``."""

        return _sorted_unique(
            (
                *self.residual_ref_ids,
                *self.obligation_ids,
                *self.property_ids,
                *self.structural_tags,
            )
        )

    def build_proof_cache_key(
        self,
        *,
        obligation: Any = "",
        translator: Any = "translator:srt-bridge@1",
        solver: Any = "solver:structural-gate@1",
        kernel: Any = "kernel:lean@1",
        toolchain: Any = SRT_BRIDGE_TOOLCHAIN_ID,
        theorem_registry: Any = "theorem-registry:srt-structural@1",
        policy: Any = SRT_BRIDGE_POLICY_ID,
        resource_budget: Any = "budget:srt-bridge@1",
    ) -> ProofCacheKey:
        """Build a trust-aware proof cache key from handles only (no gold)."""

        obl = obligation or (
            self.obligation_ids[0]
            if self.obligation_ids
            else content_identity(
                {
                    "residual_ref_ids": list(self.residual_ref_ids),
                    "structural_tags": list(self.structural_tags),
                }
            )
        )
        return build_proof_cache_key(
            obligation=obl,
            premises=self.premise_handles(),
            translator=translator,
            solver=solver,
            kernel=kernel,
            toolchain=toolchain,
            theorem_registry=theorem_registry,
            policy=policy,
            resource_budget=resource_budget,
            candidate_tree_id=self.repository_tree_id or None,
        )


def build_srt_cache_key_handles(
    *,
    residual_ref_ids: Sequence[Any] = (),
    obligation_ids: Sequence[str] = (),
    property_ids: Sequence[str] = (),
    structural_tags: Sequence[str] = (),
    repository_tree_id: str = "",
    gate_method_ids: Sequence[str] = (),
    catalog_version: str = CLAIM_CATALOG_VERSION,
    payload: Mapping[str, Any] | None = None,
) -> SrtCacheKeyHandles:
    """Construct cache-key handles; reject any gold bodies in optional payload."""

    if payload is not None:
        reject_gold_ir_bodies(payload, where="srt_cache_key_handles.payload")
    return SrtCacheKeyHandles(
        residual_ref_ids=tuple(residual_ref_ids),
        obligation_ids=tuple(obligation_ids),
        property_ids=tuple(property_ids),
        structural_tags=tuple(structural_tags),
        repository_tree_id=repository_tree_id,
        gate_method_ids=tuple(gate_method_ids),
        catalog_version=catalog_version,
        gold_ir_excluded=True,
    )


# ---------------------------------------------------------------------------
# Graph / query projection for StructuralAdmission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtGraphProjection:
    """Non-semantic graph/query projection of a structural admission."""

    nodes: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]
    query_facts: tuple[Mapping[str, Any], ...]
    semantic_authority: bool = False
    admission_id: str = ""
    repository_tree_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "nodes",
            tuple(dict(n) for n in self.nodes),
        )
        object.__setattr__(
            self,
            "edges",
            tuple(dict(e) for e in self.edges),
        )
        object.__setattr__(
            self,
            "query_facts",
            tuple(dict(f) for f in self.query_facts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SRT_GRAPH_PROJECTION_SCHEMA,
            "nodes": [dict(n) for n in self.nodes],
            "edges": [dict(e) for e in self.edges],
            "query_facts": [dict(f) for f in self.query_facts],
            "semantic_authority": False,
            "admission_id": self.admission_id,
            "repository_tree_id": self.repository_tree_id,
            "non_authoritative": True,
        }


def project_structural_admission_to_graph(
    admission: StructuralAdmission | Mapping[str, Any],
) -> SrtGraphProjection:
    """Project StructuralAdmission@1 into graph/query with non-semantic authority.

    GraphRAG / query facts derived from structural receipts are explicitly
    non-authoritative and cannot mint kernel assurance or semantic promotion.
    """

    if isinstance(admission, Mapping):
        admission = StructuralAdmission.from_dict(admission)
    if not isinstance(admission, StructuralAdmission):
        raise CodeProofSrtBridgeError(
            "admission must be StructuralAdmission or mapping"
        )
    if admission.semantic_authority is not False:
        raise CodeProofSrtBridgeError(
            "StructuralAdmission must have semantic_authority=false"
        )

    admission_id = admission.admission_id
    nodes: list[dict[str, Any]] = [
        {
            "node_id": f"admission:{admission_id}",
            "kind": "structural_admission",
            "disposition": admission.disposition.value,
            "semantic_authority": False,
            "receipt_id": admission.receipt_id,
        }
    ]
    edges: list[dict[str, Any]] = []
    query_facts: list[dict[str, Any]] = []

    for residual_id in admission.residual_ref_ids:
        node_id = f"residual:{residual_id}"
        nodes.append(
            {
                "node_id": node_id,
                "kind": "residual_ref",
                "residual_ref_id": residual_id,
                "semantic_authority": False,
            }
        )
        edges.append(
            {
                "edge_id": f"admits:{admission_id}:{residual_id}",
                "source": f"admission:{admission_id}",
                "target": node_id,
                "relation": "structurally_admits",
                "semantic_authority": False,
            }
        )
        query_facts.append(
            {
                "fact_id": f"fact:structural:{admission_id}:{residual_id}",
                "kind": "structural_admission_fact",
                "residual_ref_id": residual_id,
                "admission_id": admission_id,
                "disposition": admission.disposition.value,
                "structural_tags": list(admission.structural_tags),
                "evidence_tier": EvidenceTier.QUERY_FACT.value,
                "semantic_authority": False,
                "non_authoritative": True,
                "promotion_authorities": list(PROMOTION_AUTHORITIES),
            }
        )

    for tag in admission.structural_tags:
        tag_node = f"tag:{tag}"
        nodes.append(
            {
                "node_id": tag_node,
                "kind": "structural_tag",
                "tag": tag,
                "semantic_authority": False,
            }
        )
        edges.append(
            {
                "edge_id": f"tag:{admission_id}:{tag}",
                "source": f"admission:{admission_id}",
                "target": tag_node,
                "relation": "declares_tag",
                "semantic_authority": False,
            }
        )

    for method_id in admission.gate_method_ids:
        method_node = f"gate:{method_id}"
        nodes.append(
            {
                "node_id": method_node,
                "kind": "structural_gate_method",
                "method_id": method_id,
                "role": SrtMethodRole.STRUCTURAL_GATE.value,
                "semantic_authority": False,
            }
        )
        edges.append(
            {
                "edge_id": f"gate:{admission_id}:{method_id}",
                "source": method_node,
                "target": f"admission:{admission_id}",
                "relation": "issued_by",
                "semantic_authority": False,
            }
        )

    # Stable order.
    nodes_sorted = tuple(sorted(nodes, key=lambda n: n["node_id"]))
    edges_sorted = tuple(sorted(edges, key=lambda e: e["edge_id"]))
    facts_sorted = tuple(sorted(query_facts, key=lambda f: f["fact_id"]))

    return SrtGraphProjection(
        nodes=nodes_sorted,
        edges=edges_sorted,
        query_facts=facts_sorted,
        semantic_authority=False,
        admission_id=admission_id,
        repository_tree_id=admission.repository_tree_id,
    )


def structural_admission_to_claims(
    admission: StructuralAdmission | Mapping[str, Any],
    *,
    catalog_version: str = CLAIM_CATALOG_VERSION,
) -> tuple[CodeClaimRecord, ...]:
    """Project structural admission residuals into typed srt_structural claims.

    Claims are open query/observation-tier bindings.  They never set
    semantic_authority and never mint kernel assurance from the admission alone.
    """

    if isinstance(admission, Mapping):
        admission = StructuralAdmission.from_dict(admission)
    if not isinstance(admission, StructuralAdmission):
        raise CodeProofSrtBridgeError(
            "admission must be StructuralAdmission or mapping"
        )

    status_map = {
        StructuralAdmissionDisposition.ADMITTED: ClaimStatus.OPEN,
        StructuralAdmissionDisposition.REJECTED: ClaimStatus.REFUTED,
        StructuralAdmissionDisposition.TIMEOUT: ClaimStatus.NOT_MEASURED,
        StructuralAdmissionDisposition.UNSUPPORTED: ClaimStatus.UNSUPPORTED,
        StructuralAdmissionDisposition.NOT_MEASURED: ClaimStatus.NOT_MEASURED,
        StructuralAdmissionDisposition.ERROR: ClaimStatus.NOT_MEASURED,
    }
    status = status_map.get(admission.disposition, ClaimStatus.OPEN)

    claims: list[CodeClaimRecord] = []
    for residual_id in admission.residual_ref_ids:
        property_id = (
            admission.property_ids[0]
            if admission.property_ids
            else f"property:srt-structural:{residual_id}"
        )
        obligation_id = (
            admission.obligation_ids[0]
            if admission.obligation_ids
            else f"obligation:srt:{residual_id}"
        )
        selectors = build_invalidation_selectors(
            repository_tree_id=admission.repository_tree_id,
            premise_ids=(residual_id,),
            catalog_version=catalog_version,
            property_id=property_id,
            obligation_id=obligation_id,
            producer_id=SRT_BRIDGE_PRODUCER_ID,
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        )
        claims.append(
            CodeClaimRecord(
                claim_family=ClaimFamily.SRT_STRUCTURAL,
                status=status,
                property_id=property_id,
                obligation_id=obligation_id,
                repository_id=admission.repository_id,
                repository_tree_id=admission.repository_tree_id,
                premise_ids=(residual_id,),
                producer_id=SRT_BRIDGE_PRODUCER_ID,
                toolchain_id=SRT_BRIDGE_TOOLCHAIN_ID,
                policy_id=SRT_BRIDGE_POLICY_ID,
                catalog_version=catalog_version,
                evidence_ids=(admission.admission_id,) if admission.receipt_id else (),
                evidence_tiers=(EvidenceTier.QUERY_FACT,)
                if admission.receipt_id
                else (),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                derived_assurance=AssuranceLevel.UNVERIFIED,
                invalidation_selectors=selectors,
                statement=f"srt structural residual {residual_id}",
                receipt_id=admission.receipt_id,
                metadata={
                    "bridge": CODE_PROOF_SRT_BRIDGE_INTERFACE,
                    "structural_admission": True,
                    "semantic_authority": False,
                    "structural_tags": list(admission.structural_tags),
                    "gate_method_ids": list(admission.gate_method_ids),
                    "disposition": admission.disposition.value,
                    "promotion_authorities": list(PROMOTION_AUTHORITIES),
                    "residual_ref_id": residual_id,
                },
            )
        )
    return tuple(claims)


# ---------------------------------------------------------------------------
# Residual → claim / counterexample / context capsule / CodeEditPacket
# ---------------------------------------------------------------------------


def project_residual_to_claim(
    entry: ResidualCatalogEntry | Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    repository_id: str = "",
    catalog_version: str = CLAIM_CATALOG_VERSION,
) -> CodeClaimRecord:
    """Project one residual-catalog entry into a typed ``srt_structural`` claim."""

    if isinstance(entry, Mapping):
        entry = ResidualCatalogEntry.from_dict(entry)
    if not isinstance(entry, ResidualCatalogEntry):
        raise CodeProofSrtBridgeError(
            "entry must be ResidualCatalogEntry or mapping"
        )

    property_id = (
        entry.property_ids[0]
        if entry.property_ids
        else f"property:srt-residual:{entry.residual_ref_id}"
    )
    obligation_id = (
        entry.obligation_ids[0]
        if entry.obligation_ids
        else f"obligation:srt:{entry.residual_ref_id}"
    )
    try:
        status = ClaimStatus(entry.status)
    except ValueError:
        status = ClaimStatus.OPEN

    selectors = build_invalidation_selectors(
        repository_tree_id=repository_tree_id,
        premise_ids=(entry.residual_ref_id,),
        assumption_ids=entry.assumption_ids,
        catalog_version=catalog_version,
        property_id=property_id,
        obligation_id=obligation_id,
        producer_id=SRT_BRIDGE_PRODUCER_ID,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    return CodeClaimRecord(
        claim_family=ClaimFamily.SRT_STRUCTURAL,
        status=status,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        premise_ids=(entry.residual_ref_id,),
        assumption_ids=entry.assumption_ids,
        producer_id=SRT_BRIDGE_PRODUCER_ID,
        toolchain_id=SRT_BRIDGE_TOOLCHAIN_ID,
        policy_id=SRT_BRIDGE_POLICY_ID,
        catalog_version=catalog_version,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=selectors,
        statement=entry.summary or f"srt residual {entry.residual_ref_id}",
        metadata={
            "bridge": CODE_PROOF_SRT_BRIDGE_INTERFACE,
            "residual_ref_id": entry.residual_ref_id,
            "residual_kind": entry.residual_kind,
            "structural_tags": list(entry.structural_tags),
            "semantic_authority": False,
            "promotion_authorities": list(PROMOTION_AUTHORITIES),
            "predicted_files": list(entry.predicted_files),
            "counterexample_ref_ids": list(entry.counterexample_ref_ids),
        },
    )


def project_residual_to_counterexample(
    entry: ResidualCatalogEntry | Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    violated_property: str = "",
    summary: str = "",
) -> FormalCounterexample:
    """Project a residual entry (or its counterexample handle) into a formal CE.

    Payload carries residual handles and structural tags only — never gold IR.
    """

    if isinstance(entry, Mapping):
        entry = ResidualCatalogEntry.from_dict(entry)
    if not isinstance(entry, ResidualCatalogEntry):
        raise CodeProofSrtBridgeError(
            "entry must be ResidualCatalogEntry or mapping"
        )

    property_id = violated_property or (
        entry.property_ids[0]
        if entry.property_ids
        else f"property:srt-residual:{entry.residual_ref_id}"
    )
    payload = {
        "residual_ref_id": entry.residual_ref_id,
        "residual_kind": entry.residual_kind,
        "structural_tags": list(entry.structural_tags),
        "counterexample_ref_ids": list(entry.counterexample_ref_ids),
        "repository_tree_id": repository_tree_id,
        "semantic_authority": False,
        "gold_ir_excluded": True,
    }
    reject_gold_ir_bodies(payload, where="counterexample_payload")
    tree_ids = (repository_tree_id,) if repository_tree_id else ()
    return normalize_counterexample(
        payload,
        kind=CounterexampleKind.GENERIC_FAILURE,
        bindings=CounterexampleBindings(
            tree_ids=tree_ids,
            obligation_ids=entry.obligation_ids,
            assumption_ids=entry.assumption_ids,
        ),
        property_class="srt_structural",
        violated_property=property_id,
        summary=summary
        or entry.summary
        or f"residual counterexample for {entry.residual_ref_id}",
        assumption_ids=entry.assumption_ids,
    )


def project_residual_to_context_capsule(
    entry: ResidualCatalogEntry | Mapping[str, Any],
    *,
    repository_id: str = "repository:srt-bridge",
    repository_tree_id: str = "",
    objective_id: str = "CBP-110",
    objective_revision: str = "1",
    policy_id: str = SRT_BRIDGE_POLICY_ID,
    policy_revision: str = "1",
    caller: str = "code-proof-srt-bridge",
    stage: str = "srt-residual-bridge",
    plateau_packet_id: str = "",
    budget: ContextBudget | None = None,
) -> ContextCapsule:
    """Project residual handles into an obligation-first ContextCapsule.

    Invariant core binds residual/packet handles and non-semantic authority.
    Gold IR bodies are never embedded.
    """

    if isinstance(entry, Mapping):
        entry = ResidualCatalogEntry.from_dict(entry)
    if not isinstance(entry, ResidualCatalogEntry):
        raise CodeProofSrtBridgeError(
            "entry must be ResidualCatalogEntry or mapping"
        )
    tree = repository_tree_id or "tree:unbound"
    repo = repository_id or "repository:srt-bridge"
    active_budget = budget or ContextBudget()

    goal = {
        "kind": "srt_residual_repair",
        "residual_ref_id": entry.residual_ref_id,
        "property_ids": list(entry.property_ids),
        "obligation_ids": list(entry.obligation_ids),
        "summary": entry.summary
        or f"repair structural residual {entry.residual_ref_id}",
    }
    authority = {
        "semantic_authority": False,
        "promotion_authorities": list(PROMOTION_AUTHORITIES),
        "structural_only": True,
        "prover_semantic_authority": False,
    }
    scope = {
        "residual_ref_id": entry.residual_ref_id,
        "structural_tags": list(entry.structural_tags),
        "predicted_files": list(entry.predicted_files),
        "counterexample_ref_ids": list(entry.counterexample_ref_ids),
        "plateau_packet_id": plateau_packet_id,
        "repository_tree_id": tree,
    }
    acceptance = {
        "acceptance_ids": [
            f"accept:srt-residual:{entry.residual_ref_id}",
        ],
        "requires_e2e_loss": True,
        "requires_holdout_gate": True,
        "structural_admission_insufficient": True,
    }

    evidence_items: list[ContextReference] = [
        ContextReference(
            reference_id=f"ref:residual:{entry.residual_ref_id}",
            kind="residual_ref",
            tier=ContextTier.EVIDENCE,
            referenced_content_id=entry.residual_ref_id,
            repository_id=repo,
            tree_id=tree,
            summary=entry.summary or entry.residual_ref_id,
            metadata={
                "structural_tags": list(entry.structural_tags),
                "semantic_authority": False,
            },
        )
    ]
    # Counterexample handles ride as compact evidence references (handles only),
    # not as truncated expansion slots — keeps the capsule fail-closed and free
    # of gold IR bodies.
    for ce_id in entry.counterexample_ref_ids:
        evidence_items.append(
            ContextReference(
                reference_id=f"ref:ce:{ce_id}",
                kind="counterexample_handle",
                tier=ContextTier.EVIDENCE,
                referenced_content_id=ce_id,
                repository_id=repo,
                tree_id=tree,
                summary=f"counterexample handle {ce_id}",
                metadata={"semantic_authority": False, "handle_only": True},
            )
        )

    return ContextCapsule(
        repository_id=repo,
        tree_id=tree,
        objective_id=objective_id,
        objective_revision=objective_revision,
        policy_id=policy_id,
        policy_revision=policy_revision,
        caller=caller,
        stage=stage,
        budget=active_budget,
        goal=goal,
        authority=authority,
        scope=scope,
        acceptance=acceptance,
        evidence=tuple(evidence_items),
        expansion_references=(),
        truncated=False,
        omissions=(),
    )


def project_residual_to_code_edit_packet(
    entry: ResidualCatalogEntry | Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    plateau_packet_id: str = "",
    acceptance_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    prover: ProverBinding | Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Project a residual entry into ``CodeEditPacket@1`` (handles only)."""

    if isinstance(entry, Mapping):
        entry = ResidualCatalogEntry.from_dict(entry)
    if not isinstance(entry, ResidualCatalogEntry):
        raise CodeProofSrtBridgeError(
            "entry must be ResidualCatalogEntry or mapping"
        )

    plateau_like = {
        "packet_id": plateau_packet_id or f"plateau:from-residual:{entry.residual_ref_id}",
        "repository_tree_id": repository_tree_id,
        "residual_ref_ids": (entry.residual_ref_id,),
        "claim_ids": entry.claim_ids,
        "obligation_ids": entry.obligation_ids
        or (f"obligation:srt:{entry.residual_ref_id}",),
        "assumption_ids": entry.assumption_ids,
        "property_ids": entry.property_ids
        or (f"property:srt-residual:{entry.residual_ref_id}",),
        "predicted_files": entry.predicted_files,
        "status": entry.status,
        "acceptance_ids": tuple(acceptance_ids)
        or (f"accept:srt-residual:{entry.residual_ref_id}",),
    }
    packet = bridge_plateau_codex_packet(
        plateau_like,
        repository_tree_id=repository_tree_id,
        predicted_files=entry.predicted_files,
        acceptance_ids=acceptance_ids
        or (f"accept:srt-residual:{entry.residual_ref_id}",),
        required_assurance=required_assurance,
        prover=prover
        or ProverBinding(
            prover_id="prover:structural-gate",
            solver_id="solver:cvc5",
            kernel_id="kernel:lean",
            semantic_authority=False,
        ),
    )
    # Annotate with residual bridge metadata (still no gold).
    meta = dict(packet.metadata)
    meta.update(
        {
            "bridge": CODE_PROOF_SRT_BRIDGE_INTERFACE,
            "source": "plat_residual_catalog",
            "residual_ref_id": entry.residual_ref_id,
            "structural_tags": list(entry.structural_tags),
            "edit_target_role": SrtMethodRole.EDIT_TARGET.value,
            "semantic_authority": False,
            "promotion_authorities": list(PROMOTION_AUTHORITIES),
            "gold_ir_excluded": True,
        }
    )
    return packet.with_updates(metadata=meta)


def project_plateau_codex_packet(
    plateau: Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    prover: ProverBinding | Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Project ``PlateauCodexPacket@1`` ids into ``CodeEditPacket@1``.

    Delegates to the CBP-080 handles-only materializer bridge after gold
    rejection and SRT metadata annotation.
    """

    if not isinstance(plateau, Mapping):
        raise CodeProofSrtBridgeError("plateau packet must be a mapping")
    reject_gold_ir_bodies(plateau, where="PlateauCodexPacket")
    packet = bridge_plateau_codex_packet(
        plateau,
        repository_tree_id=repository_tree_id,
        predicted_files=predicted_files,
        acceptance_ids=acceptance_ids,
        required_assurance=required_assurance,
        prover=prover
        or ProverBinding(
            prover_id="prover:structural-gate",
            solver_id="solver:cvc5",
            kernel_id="kernel:lean",
            semantic_authority=False,
        ),
    )
    meta = dict(packet.metadata)
    meta.update(
        {
            "bridge": CODE_PROOF_SRT_BRIDGE_INTERFACE,
            "source": PLATEAU_CODEX_PACKET_INTERFACE,
            "semantic_authority": False,
            "promotion_authorities": list(PROMOTION_AUTHORITIES),
            "edit_target_role": SrtMethodRole.EDIT_TARGET.value,
            "gold_ir_excluded": True,
        }
    )
    return packet.with_updates(metadata=meta)


# ---------------------------------------------------------------------------
# Aggregate bridge projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtBridgeProjection:
    """Aggregate projection from PLAT residual / plateau sources into CBP."""

    claims: tuple[CodeClaimRecord, ...]
    counterexamples: tuple[FormalCounterexample, ...]
    context_capsules: tuple[ContextCapsule, ...]
    code_edit_packets: tuple[CodeEditPacket, ...]
    graph_projections: tuple[SrtGraphProjection, ...] = ()
    cache_key_handles: tuple[SrtCacheKeyHandles, ...] = ()
    residual_ref_ids: tuple[str, ...] = ()
    plateau_packet_ids: tuple[str, ...] = ()
    semantic_authority: bool = False
    promotion_authorities: tuple[str, ...] = PROMOTION_AUTHORITIES
    gold_ir_excluded: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "gold_ir_excluded", True)
        object.__setattr__(
            self,
            "promotion_authorities",
            tuple(self.promotion_authorities) or PROMOTION_AUTHORITIES,
        )
        object.__setattr__(
            self, "residual_ref_ids", _sorted_unique(self.residual_ref_ids)
        )
        object.__setattr__(
            self, "plateau_packet_ids", _sorted_unique(self.plateau_packet_ids)
        )
        object.__setattr__(self, "notes", _sorted_unique(self.notes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SRT_BRIDGE_PROJECTION_SCHEMA,
            "interface": CODE_PROOF_SRT_BRIDGE_INTERFACE,
            "version": CODE_PROOF_SRT_BRIDGE_VERSION,
            "claims": [c.to_dict() for c in self.claims],
            "counterexamples": [ce.to_dict() for ce in self.counterexamples],
            "context_capsules": [cap.to_dict() for cap in self.context_capsules],
            "code_edit_packets": [
                p.to_dict(include_id=True) for p in self.code_edit_packets
            ],
            "graph_projections": [g.to_dict() for g in self.graph_projections],
            "cache_key_handles": [h.to_dict() for h in self.cache_key_handles],
            "residual_ref_ids": list(self.residual_ref_ids),
            "plateau_packet_ids": list(self.plateau_packet_ids),
            "semantic_authority": False,
            "promotion_authorities": list(self.promotion_authorities),
            "gold_ir_excluded": True,
            "notes": list(self.notes),
        }


def project_srt_residual_catalog(
    catalog: SrtResidualCatalog | Mapping[str, Any],
    *,
    include_counterexamples: bool = True,
    include_capsules: bool = True,
    include_edit_packets: bool = True,
    structural_admission: StructuralAdmission | Mapping[str, Any] | None = None,
) -> SrtBridgeProjection:
    """Project an entire PLAT residual catalog into CBP claim/CE/capsule/packet."""

    if isinstance(catalog, Mapping):
        catalog = SrtResidualCatalog.from_dict(catalog)
    if not isinstance(catalog, SrtResidualCatalog):
        raise CodeProofSrtBridgeError(
            "catalog must be SrtResidualCatalog or mapping"
        )

    claims: list[CodeClaimRecord] = []
    counterexamples: list[FormalCounterexample] = []
    capsules: list[ContextCapsule] = []
    packets: list[CodeEditPacket] = []
    cache_handles: list[SrtCacheKeyHandles] = []
    notes = [
        "semantic_authority_false",
        "gold_ir_excluded",
        "e2e_loss_and_holdout_remain_promotion_authority",
        "methods_not_interchangeable",
    ]

    for entry in catalog.entries:
        claims.append(
            project_residual_to_claim(
                entry,
                repository_tree_id=catalog.repository_tree_id,
                repository_id=catalog.repository_id,
            )
        )
        if include_counterexamples and (
            entry.counterexample_ref_ids
            or entry.status == ClaimStatus.REFUTED.value
        ):
            counterexamples.append(
                project_residual_to_counterexample(
                    entry,
                    repository_tree_id=catalog.repository_tree_id,
                )
            )
        if include_capsules:
            capsules.append(
                project_residual_to_context_capsule(
                    entry,
                    repository_id=catalog.repository_id or "repository:srt-bridge",
                    repository_tree_id=catalog.repository_tree_id,
                    plateau_packet_id=catalog.plateau_packet_id,
                )
            )
        if include_edit_packets:
            packets.append(
                project_residual_to_code_edit_packet(
                    entry,
                    repository_tree_id=catalog.repository_tree_id,
                    plateau_packet_id=catalog.plateau_packet_id,
                )
            )
        cache_handles.append(
            build_srt_cache_key_handles(
                residual_ref_ids=(entry.residual_ref_id,),
                obligation_ids=entry.obligation_ids,
                property_ids=entry.property_ids,
                structural_tags=entry.structural_tags,
                repository_tree_id=catalog.repository_tree_id,
            )
        )

    graph_projections: list[SrtGraphProjection] = []
    if structural_admission is not None:
        graph_projections.append(
            project_structural_admission_to_graph(structural_admission)
        )
        claims.extend(
            structural_admission_to_claims(structural_admission)
        )
        notes.append("structural_admission_projected_non_semantic")

    return SrtBridgeProjection(
        claims=tuple(claims),
        counterexamples=tuple(counterexamples),
        context_capsules=tuple(capsules),
        code_edit_packets=tuple(packets),
        graph_projections=tuple(graph_projections),
        cache_key_handles=tuple(cache_handles),
        residual_ref_ids=catalog.residual_ref_ids(),
        plateau_packet_ids=(catalog.plateau_packet_id,)
        if catalog.plateau_packet_id
        else (),
        semantic_authority=False,
        promotion_authorities=PROMOTION_AUTHORITIES,
        gold_ir_excluded=True,
        notes=tuple(notes),
    )


def project_plateau_packet_bundle(
    plateau: Mapping[str, Any],
    *,
    residual_entries: Sequence[ResidualCatalogEntry | Mapping[str, Any]] = (),
    structural_admission: StructuralAdmission | Mapping[str, Any] | None = None,
) -> SrtBridgeProjection:
    """Project a PlateauCodexPacket@1 (and optional residual rows) into CBP."""

    reject_gold_ir_bodies(plateau, where="PlateauCodexPacket")
    packet = project_plateau_codex_packet(plateau)
    tree = str(
        plateau.get("repository_tree_id")
        or plateau.get("source_tree_id")
        or plateau.get("tree_id")
        or packet.repository_tree_id
        or ""
    )
    residual_ids = _extract_handles(
        plateau, "residual_ref_ids", "residual_ids", "residuals"
    )
    plateau_id = str(
        plateau.get("packet_id")
        or plateau.get("content_id")
        or plateau.get("plateau_packet_id")
        or packet.plateau_packet_id
        or ""
    )

    entries: list[ResidualCatalogEntry] = []
    for item in residual_entries:
        if isinstance(item, ResidualCatalogEntry):
            entries.append(item)
        else:
            entries.append(ResidualCatalogEntry.from_dict(item))
    # If no explicit residual rows, synthesize handle-only rows from the packet.
    if not entries and residual_ids:
        for rid in residual_ids:
            entries.append(
                ResidualCatalogEntry(
                    residual_ref_id=rid,
                    property_ids=tuple(plateau.get("property_ids") or ()),
                    claim_ids=tuple(plateau.get("claim_ids") or ()),
                    obligation_ids=tuple(plateau.get("obligation_ids") or ()),
                    predicted_files=tuple(
                        plateau.get("predicted_files") or plateau.get("paths") or ()
                    ),
                    assumption_ids=tuple(
                        plateau.get("assumption_ids")
                        or plateau.get("assumptions")
                        or ()
                    ),
                    status=str(
                        plateau.get("status")
                        or plateau.get("claim_status")
                        or ClaimStatus.OPEN.value
                    ),
                )
            )

    catalog = build_srt_residual_catalog(
        entries,
        repository_tree_id=tree,
        repository_id=str(plateau.get("repository_id") or ""),
        plateau_packet_id=plateau_id,
        metadata={"source": PLATEAU_CODEX_PACKET_INTERFACE},
    )
    projection = project_srt_residual_catalog(
        catalog,
        structural_admission=structural_admission,
    )
    # Ensure the plateau-derived CodeEditPacket is present even with empty residuals.
    packets = list(projection.code_edit_packets)
    if not packets:
        packets.append(packet)
    elif packet.packet_id not in {p.packet_id for p in packets}:
        packets.insert(0, packet)

    notes = list(projection.notes) + [
        "plateau_codex_packet_projected",
        f"code_edit_packet_interface:{CODE_EDIT_PACKET_INTERFACE}",
    ]
    return SrtBridgeProjection(
        claims=projection.claims,
        counterexamples=projection.counterexamples,
        context_capsules=projection.context_capsules,
        code_edit_packets=tuple(packets),
        graph_projections=projection.graph_projections,
        cache_key_handles=projection.cache_key_handles,
        residual_ref_ids=projection.residual_ref_ids or residual_ids,
        plateau_packet_ids=_sorted_unique(
            (*projection.plateau_packet_ids, plateau_id)
        ),
        semantic_authority=False,
        promotion_authorities=PROMOTION_AUTHORITIES,
        gold_ir_excluded=True,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Method-role registry surface (docs / introspection)
# ---------------------------------------------------------------------------


def method_roles_manifest() -> dict[str, Any]:
    """Return the measured method-role table for documentation and tests."""

    return {
        "interface": CODE_PROOF_SRT_BRIDGE_INTERFACE,
        "version": CODE_PROOF_SRT_BRIDGE_VERSION,
        "methods_interchangeable": False,
        "roles": {
            role: {
                "methods": list(methods),
                "description": method_role_description(role),
            }
            for role, methods in METHOD_ROLES_TABLE.items()
        },
        "promotion_authorities": list(PROMOTION_AUTHORITIES),
        "structural_semantic_authority": STRUCTURAL_SEMANTIC_AUTHORITY,
        "srt_structural_tags": list(SRT_STRUCTURAL_TAGS),
        "gold_ir_in_cache_keys": False,
    }



# ---------------------------------------------------------------------------
# Compatibility aliases (board-prefix spellings; prefer Srt* names)
# ---------------------------------------------------------------------------
PlatResidualCatalog = SrtResidualCatalog
Plat2HoldoutArtifact = SrtHoldoutArtifact
Plat2HoldoutRegistry = SrtHoldoutRegistry
build_plat_residual_catalog = build_srt_residual_catalog
project_plat_residual_catalog = project_srt_residual_catalog

__all__ = [
    "CODE_PROOF_SRT_BRIDGE_INTERFACE",
    "CODE_PROOF_SRT_BRIDGE_VERSION",
    "STRUCTURAL_ADMISSION_INTERFACE",
    "STRUCTURAL_ADMISSION_SCHEMA",
    "PLATEAU_CODEX_PACKET_INTERFACE",
    "PLATEAU_CODEX_PACKET_SCHEMA",
    "PLAT_RESIDUAL_CATALOG_INTERFACE",
    "PLAT2_HOLDOUT_REGISTRY_INTERFACE",
    "PROMOTION_AUTHORITIES",
    "STRUCTURAL_SEMANTIC_AUTHORITY",
    "METHOD_ROLES_TABLE",
    "SRT_STRUCTURAL_TAGS",
    "CodeProofSrtBridgeError",
    "SrtMethodRole",
    "StructuralAdmissionDisposition",
    "StructuralAdmission",
    "ResidualCatalogEntry",
    "SrtResidualCatalog",
    "SrtHoldoutArtifact",
    "SrtHoldoutRegistry",
    "SrtCacheKeyHandles",
    "SrtGraphProjection",
    "SrtBridgeProjection",
    "resolve_method_role",
    "method_role_description",
    "method_roles_manifest",
    "reject_gold_ir_bodies",
    "build_structural_admission",
    "build_plat_residual_catalog",
    "build_srt_cache_key_handles",
    "project_structural_admission_to_graph",
    "structural_admission_to_claims",
    "project_residual_to_claim",
    "project_residual_to_counterexample",
    "project_residual_to_context_capsule",
    "project_residual_to_code_edit_packet",
    "project_plateau_codex_packet",
    "project_plat_residual_catalog",
    "project_plateau_packet_bundle",
    "PlatResidualCatalog",
    "Plat2HoldoutArtifact",
    "Plat2HoldoutRegistry",
    "build_srt_residual_catalog",
    "project_srt_residual_catalog",
    "SRT_HOLDOUT_PROMOTION_GATE",
    "PLAT2_HOLDOUT_PROMOTION_GATE"
]
