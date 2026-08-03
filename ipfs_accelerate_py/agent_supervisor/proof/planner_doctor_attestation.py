"""Planner/Doctor reasoning-run lineage and optional ZKP attestation (PDR-060).

This module defines:

* :class:`ReasoningRunManifest` — a closed, typed link from one reasoning run to
  Planner, Doctor, cache, plan, permit, mutation, fixed-point, benchmark, and
  promotion receipt CIDs **without collapsing evidence types**;
* Merkle / preimage / run-replay verification over that ordered lineage; and
* :class:`PlannerDoctorAttestation` — an optional zero-knowledge envelope that
  may attest only a narrow, threat-model-approved privacy or fixed-computation
  claim over the committed lineage.

Conflict policy (normative):

* Reuse the shared assurance lattice from ``formal_verification_contracts`` and
  the private-witness / sim≠ATTESTED patterns from ``proof_attestation`` and
  ``program_analysis_zkp``.  Do **not** invent new assurance levels.
* Attestation never substitutes for program semantics, inventory completeness,
  or translator soundness.
* Unavailable, failed, or simulated backends stay at
  ``AssuranceLevel.CANDIDATE`` / ``UNVERIFIED`` and never emit production
  ``AssuranceLevel.ATTESTED``.

Threat model:
``docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Final, TypeVar

from ..program_analysis_zkp import (
    ProgramZkpBackendMode as ProgramAnalysisZkpBackendMode,
)
from ..program_analysis_zkp import (
    ProgramZkpVerificationReceipt,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    canonical_json_bytes,
    content_identity,
)
from .proof_attestation import (
    AttestationBackendMode,
    AttestationGate,
    AttestationTrust,
    AttestationVerificationVerdict,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Versioning and schema identities
# ---------------------------------------------------------------------------

PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION

REASONING_RUN_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-run-manifest@1"
)
REASONING_RUN_MANIFEST_INTERFACE: Final[str] = "ReasoningRunManifest@1"

PLANNER_DOCTOR_ATTESTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-attestation@1"
)
PLANNER_DOCTOR_ATTESTATION_INTERFACE: Final[str] = "PlannerDoctorAttestation@1"

PLANNER_DOCTOR_PUBLIC_INPUTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-attestation-public-inputs@1"
)
PLANNER_DOCTOR_VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-attestation-verification@1"
)
PLANNER_DOCTOR_PROGRAM_ZKP_BRIDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-program-zkp-bridge@1"
)
PLANNER_DOCTOR_LINEAGE_SLOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-lineage-slot@1"
)

PUBLIC_INPUT_CODEC_ID: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-attestation-public-input-codec"
)
PUBLIC_INPUT_CODEC_VERSION: Final[str] = "1"

# Threat-model use case for the only currently approved optional ZKP claim.
PLANNER_DOCTOR_ZKP_USE_CASE_ID: Final[str] = (
    "pdr-060-receipt-lineage-private-witness"
)
PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID: Final[str] = (
    "docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md"
)

# Default circuit pin for the optional lineage/possession claim.
DEFAULT_LINEAGE_CIRCUIT_ID: Final[str] = (
    "circuit:planner-doctor-lineage-possession@1"
)
DEFAULT_LINEAGE_CIRCUIT_VERSION: Final[str] = "1"

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_PREIMAGE_BYTES: Final[int] = 1_048_576

# ---------------------------------------------------------------------------
# Closed lineage vocabulary — never collapse these into a single evidence tier
# ---------------------------------------------------------------------------

# Ordered: leaf order is part of the Merkle root commitment.
LINEAGE_EVIDENCE_TYPES: Final[tuple[str, ...]] = (
    "planner",
    "doctor",
    "cache",
    "plan",
    "permit",
    "mutation",
    "fixed_point",
    "benchmark",
    "promotion",
)


class LineageEvidenceType(str, Enum):
    """Distinct evidence kinds linked by a reasoning-run manifest.

    Collapsing these types (for example treating a planner CID as a doctor
    receipt, or a benchmark CID as promotion evidence) is a contract violation.
    """

    PLANNER = "planner"
    DOCTOR = "doctor"
    CACHE = "cache"
    PLAN = "plan"
    PERMIT = "permit"
    MUTATION = "mutation"
    FIXED_POINT = "fixed_point"
    BENCHMARK = "benchmark"
    PROMOTION = "promotion"


assert tuple(item.value for item in LineageEvidenceType) == LINEAGE_EVIDENCE_TYPES


# Explicit non-claims: what lineage integrity or optional ZKP must never imply.
ATTESTATION_DOES_NOT_PROVE: Final[frozenset[str]] = frozenset(
    {
        "semantic_correctness",
        "inventory_completeness",
        "translator_soundness",
        "arbitrary_runtime_semantics",
        "goal_completion",
        "theorem_beyond_committed_circuit",
    }
)

ATTESTATION_SCOPE_STATEMENT: Final[str] = (
    "Planner/Doctor attestation proves only that committed lineage CIDs open to "
    "typed preimages under a fixed ordered Merkle root for one run_id, and "
    "(optionally) that a private witness satisfies an approved fixed circuit "
    "over those public inputs. It does not prove semantic correctness, "
    "inventory completeness, translator soundness, arbitrary runtime "
    "semantics, goal completion, or any theorem beyond the committed circuit."
)

# Field names that must never appear in public artifacts with live values.
_PRIVATE_WITNESS_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "commitment_opening",
        "cookie",
        "credential",
        "hidden_witness",
        "opening",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "proof_trace",
        "refresh_token",
        "secret",
        "session_token",
        "source_text",
        "witness",
        "witness_opening",
    }
)

_SAFE_PUBLIC_REDACTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "private_witness_redacted",
        "witness_redacted",
        "openings_redacted",
    }
)

# Ordered public-input slots for the optional ZKP codec.
PUBLIC_COMMITMENT_KEYS: Final[tuple[str, ...]] = (
    "run_id",
    "manifest_id",
    "lineage_merkle_root",
    "repository_tree_id",
    "policy_id",
    "circuit_id",
    "circuit_version",
    "proving_key_id",
    "verifying_key_id",
    "ceremony_id",
    "use_case_id",
    "threat_model_id",
    "public_input_codec_id",
    "public_input_codec_version",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlannerDoctorAttestationError(ContractValidationError):
    """Base fail-closed error for planner/doctor attestation contracts."""


class LineageValidationError(PlannerDoctorAttestationError):
    """Raised when a reasoning-run manifest or Merkle lineage is invalid."""


class LineagePreimageError(LineageValidationError):
    """Raised when a supplied preimage does not match a committed CID."""


class LineageOrderError(LineageValidationError):
    """Raised when evidence types are reordered, collapsed, or substituted."""


class LineageRootError(LineageValidationError):
    """Raised when a Merkle root does not match recomputed lineage leaves."""


class LineageReplayError(LineageValidationError):
    """Raised when a manifest is replayed against a wrong run or drifted root."""


class WitnessDisclosureError(PlannerDoctorAttestationError):
    """Raised when private witness material reaches a public boundary."""


class AttestationClaimPromotionError(PlannerDoctorAttestationError):
    """Raised when attestation is promoted into a forbidden semantic claim."""


class AttestationBackendError(PlannerDoctorAttestationError):
    """Raised when a backend is unavailable, failed, or not production-eligible."""


# ---------------------------------------------------------------------------
# Status / backend vocabulary
# ---------------------------------------------------------------------------


class PlannerDoctorBackendMode(str, Enum):
    """Trust class of the optional planner/doctor ZKP path."""

    CRYPTOGRAPHIC = "cryptographic"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    SHADOW = "shadow"


class PlannerDoctorAttestationStatus(str, Enum):
    """Typed outcome; only ``attested`` may contribute production ATTESTED."""

    GENERATED = "generated"
    ATTESTED = "attested"
    CANDIDATE = "candidate"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REJECTED = "rejected"
    ERROR = "error"


class PlannerDoctorZkpPredicate(str, Enum):
    """Closed optional predicates approved by the PDR-060 threat model."""

    RECEIPT_LINEAGE = "receipt_lineage"
    PRIVATE_WITNESS_POSSESSION = "private_witness_possession"
    FIXED_BOUNDED_COMPUTATION = "fixed_bounded_computation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        normalized = ""
    elif not isinstance(value, str):
        raise PlannerDoctorAttestationError("%s must be a string" % field_name)
    else:
        normalized = value.strip()
    if required and not normalized:
        raise PlannerDoctorAttestationError("%s is required" % field_name)
    if "\x00" in normalized:
        raise PlannerDoctorAttestationError("%s must not contain NUL" % field_name)
    if len(normalized.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PlannerDoctorAttestationError(
            "%s exceeds %s UTF-8 bytes" % (field_name, MAX_TEXT_BYTES)
        )
    return normalized


def _enum(value: Any, enum_type: type[T], *, field_name: str) -> T:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        raise PlannerDoctorAttestationError(
            "%s must be one of: %s" % (field_name, allowed)
        ) from exc


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerDoctorAttestationError("%s must be a boolean" % field_name)
    return value


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise PlannerDoctorAttestationError("canonical artifact must be an object")
    if payload.get("schema") != expected:
        raise PlannerDoctorAttestationError(
            "canonical artifact has unsupported schema; expected %s" % expected
        )


def cid_for_preimage(preimage: Any) -> str:
    """Content-address a typed preimage into a CIDv1 DAG-JSON/sha2-256 identity."""

    return content_identity(preimage)


def leaf_digest_for_slot(
    evidence_type: LineageEvidenceType | str,
    cid: str,
    *,
    run_id: str,
) -> bytes:
    """Return the binary Merkle leaf digest for one typed lineage slot.

    The leaf binds ``run_id``, evidence type, and CID so that swapping types,
    reordering, or cross-run reuse changes the digest.
    """

    kind = _enum(evidence_type, LineageEvidenceType, field_name="evidence_type")
    material = {
        "domain": "planner-doctor-lineage-leaf@1",
        "run_id": _text(run_id, field_name="run_id"),
        "evidence_type": kind.value,
        "cid": _text(cid, field_name="cid"),
    }
    return hashlib.sha256(canonical_json_bytes(material)).digest()


def merkle_root_from_leaves(leaves: Sequence[bytes]) -> str:
    """Compute a pairwise SHA-256 Merkle root over ordered leaf digests.

    An odd node is duplicated (Bitcoin-style) so the order and length of the
    leaf sequence are both committed.  The returned root is a content identity
    over the final 32-byte digest so consumers can treat it as a CID-like pin.
    """

    if not leaves:
        raise LineageValidationError("merkle tree requires at least one leaf")
    level = [bytes(leaf) for leaf in leaves]
    for leaf in level:
        if not isinstance(leaf, (bytes, bytearray)) or len(leaf) != 32:
            raise LineageValidationError("each merkle leaf must be a 32-byte digest")
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt: list[bytes] = []
        for index in range(0, len(level), 2):
            nxt.append(hashlib.sha256(level[index] + level[index + 1]).digest())
        level = nxt
    return content_identity(
        {
            "domain": "planner-doctor-lineage-merkle-root@1",
            "digest_hex": level[0].hex(),
            "leaf_count": len(leaves),
        }
    )


def attestation_does_not_prove(claim: str) -> bool:
    """Return True when ``claim`` is an explicit non-claim of this surface."""

    return _text(claim, field_name="claim") in ATTESTATION_DOES_NOT_PROVE


def reject_illegal_semantic_claim(claim: str) -> None:
    """Fail closed if a caller tries to treat attestation as a forbidden claim."""

    name = _text(claim, field_name="claim")
    if name in ATTESTATION_DOES_NOT_PROVE:
        raise AttestationClaimPromotionError(
            "planner/doctor attestation cannot claim %s; %s"
            % (name, ATTESTATION_SCOPE_STATEMENT)
        )


# ---------------------------------------------------------------------------
# Lineage slot and reasoning-run manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageSlot(CanonicalContract):
    """One typed body-free receipt handle in a reasoning-run lineage."""

    SCHEMA: ClassVar[str] = PLANNER_DOCTOR_LINEAGE_SLOT_SCHEMA

    evidence_type: LineageEvidenceType
    receipt_cid: str
    span_id: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_type",
            _enum(self.evidence_type, LineageEvidenceType, field_name="evidence_type"),
        )
        # Accept legacy "cid" key only through from_dict; field is receipt_cid so
        # it does not collide with CanonicalContract.cid (content identity).
        object.__setattr__(
            self,
            "receipt_cid",
            _text(self.receipt_cid, field_name="receipt_cid", required=True),
        )
        object.__setattr__(
            self,
            "span_id",
            _text(self.span_id, field_name="span_id", required=False),
        )
        object.__setattr__(
            self, "label", _text(self.label, field_name="label", required=False)
        )

    def leaf_digest(self, *, run_id: str) -> bytes:
        return leaf_digest_for_slot(
            self.evidence_type, self.receipt_cid, run_id=run_id
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION,
            "evidence_type": self.evidence_type.value,
            "receipt_cid": self.receipt_cid,
            "span_id": self.span_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineageSlot":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorAttestationError("lineage slot must be a mapping")
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("content_id", None)
        receipt = data.get("receipt_cid") or data.get("cid") or ""
        return cls(
            evidence_type=data.get("evidence_type", ""),
            receipt_cid=receipt,
            span_id=data.get("span_id", ""),
            label=data.get("label", ""),
        )


def _normalize_slots(
    slots: Mapping[str, Any] | Sequence[Any],
) -> tuple[LineageSlot, ...]:
    """Normalize a mapping or sequence of slots into the closed ordered form.

    Mapping keys must be evidence types.  Sequence items must each declare
    their type.  Missing types, extras, duplicates, and type/position
    mismatches fail closed so evidence kinds cannot be collapsed.
    """

    by_type: dict[LineageEvidenceType, LineageSlot] = {}
    if isinstance(slots, Mapping):
        for raw_key, raw_value in slots.items():
            kind = _enum(raw_key, LineageEvidenceType, field_name="evidence_type")
            if kind in by_type:
                raise LineageOrderError(
                    "duplicate lineage evidence type %s" % kind.value
                )
            if isinstance(raw_value, LineageSlot):
                slot = raw_value
            elif isinstance(raw_value, Mapping):
                material = dict(raw_value)
                material.setdefault("evidence_type", kind.value)
                slot = LineageSlot.from_dict(material)
            elif isinstance(raw_value, str):
                slot = LineageSlot(evidence_type=kind, receipt_cid=raw_value)
            else:
                raise PlannerDoctorAttestationError(
                    "lineage slot for %s must be a CID string, mapping, or LineageSlot"
                    % kind.value
                )
            if slot.evidence_type is not kind:
                raise LineageOrderError(
                    "lineage slot type mismatch for key %s: got %s"
                    % (kind.value, slot.evidence_type.value)
                )
            by_type[kind] = slot
    elif isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
        for index, raw_value in enumerate(slots):
            if isinstance(raw_value, LineageSlot):
                slot = raw_value
            elif isinstance(raw_value, Mapping):
                slot = LineageSlot.from_dict(raw_value)
            else:
                raise PlannerDoctorAttestationError(
                    "sequence lineage slot at index %s must be a mapping or LineageSlot"
                    % index
                )
            if slot.evidence_type in by_type:
                raise LineageOrderError(
                    "duplicate lineage evidence type %s" % slot.evidence_type.value
                )
            by_type[slot.evidence_type] = slot
    else:
        raise PlannerDoctorAttestationError(
            "lineage slots must be a mapping or sequence of typed slots"
        )

    expected = tuple(LineageEvidenceType)
    missing = [item.value for item in expected if item not in by_type]
    extra = sorted(
        kind.value for kind in by_type if kind not in set(expected)
    )
    if missing or extra:
        raise LineageOrderError(
            "lineage must bind every evidence type exactly once; "
            "missing=%s extra=%s" % (missing, extra)
        )
    return tuple(by_type[kind] for kind in expected)


@dataclass(frozen=True)
class ReasoningRunManifest(CanonicalContract):
    """``ReasoningRunManifest@1`` — typed lineage CIDs for one reasoning run.

    The manifest records exact body-free handles for Planner, Doctor, cache,
    plan, permit, mutation, fixed-point, benchmark, and promotion evidence.
    Evidence types remain distinct fields; the ordered Merkle root commits to
    both the CIDs and their types/order for one ``run_id``.
    """

    SCHEMA: ClassVar[str] = REASONING_RUN_MANIFEST_SCHEMA

    run_id: str
    repository_tree_id: str
    policy_id: str
    slots: tuple[LineageSlot, ...]
    lineage_merkle_root: str = ""
    parent_run_id: str = ""
    signature: str = ""
    signer_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_id", _text(self.run_id, field_name="run_id", required=True)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        object.__setattr__(
            self,
            "parent_run_id",
            _text(self.parent_run_id, field_name="parent_run_id", required=False),
        )
        object.__setattr__(
            self,
            "signature",
            _text(self.signature, field_name="signature", required=False),
        )
        object.__setattr__(
            self,
            "signer_id",
            _text(self.signer_id, field_name="signer_id", required=False),
        )
        normalized = _normalize_slots(self.slots)
        object.__setattr__(self, "slots", normalized)
        expected_root = merkle_root_from_leaves(
            [slot.leaf_digest(run_id=self.run_id) for slot in normalized]
        )
        claimed = _text(
            self.lineage_merkle_root,
            field_name="lineage_merkle_root",
            required=False,
        )
        if claimed and claimed != expected_root:
            raise LineageRootError(
                "lineage_merkle_root does not match recomputed ordered leaves"
            )
        object.__setattr__(self, "lineage_merkle_root", expected_root)

    @property
    def manifest_id(self) -> str:
        return self.content_id

    @property
    def interface(self) -> str:
        return REASONING_RUN_MANIFEST_INTERFACE

    def slot_map(self) -> Mapping[str, LineageSlot]:
        return MappingProxyType({slot.evidence_type.value: slot for slot in self.slots})

    def cid_for(self, evidence_type: LineageEvidenceType | str) -> str:
        kind = _enum(evidence_type, LineageEvidenceType, field_name="evidence_type")
        for slot in self.slots:
            if slot.evidence_type is kind:
                return slot.receipt_cid
        raise LineageValidationError("missing lineage slot %s" % kind.value)

    def leaf_digests(self) -> tuple[bytes, ...]:
        return tuple(slot.leaf_digest(run_id=self.run_id) for slot in self.slots)

    def recompute_merkle_root(self) -> str:
        return merkle_root_from_leaves(self.leaf_digests())

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION,
            "interface": REASONING_RUN_MANIFEST_INTERFACE,
            "run_id": self.run_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "parent_run_id": self.parent_run_id,
            "lineage_merkle_root": self.lineage_merkle_root,
            "slots": [slot.to_dict() for slot in self.slots],
            "signature": self.signature,
            "signer_id": self.signer_id,
            "evidence_types": list(LINEAGE_EVIDENCE_TYPES),
            "does_not_prove": sorted(ATTESTATION_DOES_NOT_PROVE),
            "scope": ATTESTATION_SCOPE_STATEMENT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningRunManifest":
        _schema(payload, cls.SCHEMA)
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("interface", None)
        data.pop("evidence_types", None)
        data.pop("does_not_prove", None)
        data.pop("scope", None)
        claimed = data.pop("content_id", None) or data.pop("manifest_id", None)
        result = cls(
            run_id=data.get("run_id", ""),
            repository_tree_id=data.get("repository_tree_id", ""),
            policy_id=data.get("policy_id", ""),
            slots=tuple(data.get("slots") or ()),
            lineage_merkle_root=data.get("lineage_merkle_root", ""),
            parent_run_id=data.get("parent_run_id", ""),
            signature=data.get("signature", ""),
            signer_id=data.get("signer_id", ""),
        )
        if claimed not in (None, "", result.manifest_id):
            raise LineageRootError(
                "reasoning-run manifest identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {**self.to_dict(), "manifest_id": self.manifest_id}
        reject_private_witness_from_public_payload(public)
        return public


def build_reasoning_run_manifest(
    *,
    run_id: str,
    repository_tree_id: str,
    policy_id: str,
    planner_cid: str,
    doctor_cid: str,
    cache_cid: str,
    plan_cid: str,
    permit_cid: str,
    mutation_cid: str,
    fixed_point_cid: str,
    benchmark_cid: str,
    promotion_cid: str,
    parent_run_id: str = "",
    signature: str = "",
    signer_id: str = "",
    spans: Mapping[str, str] | None = None,
) -> ReasoningRunManifest:
    """Build a complete typed lineage manifest from per-type CIDs."""

    span_map = dict(spans or {})
    cids = {
        LineageEvidenceType.PLANNER: planner_cid,
        LineageEvidenceType.DOCTOR: doctor_cid,
        LineageEvidenceType.CACHE: cache_cid,
        LineageEvidenceType.PLAN: plan_cid,
        LineageEvidenceType.PERMIT: permit_cid,
        LineageEvidenceType.MUTATION: mutation_cid,
        LineageEvidenceType.FIXED_POINT: fixed_point_cid,
        LineageEvidenceType.BENCHMARK: benchmark_cid,
        LineageEvidenceType.PROMOTION: promotion_cid,
    }
    slots = tuple(
        LineageSlot(
            evidence_type=kind,
            receipt_cid=cids[kind],
            span_id=str(span_map.get(kind.value, "") or ""),
        )
        for kind in LineageEvidenceType
    )
    return ReasoningRunManifest(
        run_id=run_id,
        repository_tree_id=repository_tree_id,
        policy_id=policy_id,
        slots=slots,
        parent_run_id=parent_run_id,
        signature=signature,
        signer_id=signer_id,
    )


def verify_lineage_preimages(
    manifest: ReasoningRunManifest,
    preimages: Mapping[str, Any],
) -> None:
    """Require every lineage CID to open to its typed preimage.

    ``preimages`` maps evidence-type name → preimage material.  Wrong
    preimages, missing types, or type/CID mismatches fail closed.
    """

    if not isinstance(manifest, ReasoningRunManifest):
        raise PlannerDoctorAttestationError("manifest must be a ReasoningRunManifest")
    if not isinstance(preimages, Mapping):
        raise PlannerDoctorAttestationError("preimages must be a mapping")
    expected_types = {kind.value for kind in LineageEvidenceType}
    provided = {str(key).strip() for key in preimages}
    missing = sorted(expected_types - provided)
    extra = sorted(provided - expected_types)
    if missing or extra:
        raise LineagePreimageError(
            "preimages must cover every lineage evidence type exactly once; "
            "missing=%s extra=%s" % (missing, extra)
        )
    for slot in manifest.slots:
        key = slot.evidence_type.value
        material = preimages[key]
        # Bind type into the preimage identity so collapsing types fails even
        # when the body bytes are identical.
        expected = cid_for_preimage(
            {
                "evidence_type": key,
                "run_id": manifest.run_id,
                "body": material,
            }
        )
        if expected != slot.receipt_cid:
            raise LineagePreimageError(
                "preimage for %s does not match committed CID" % key
            )


def typed_receipt_cid(
    *,
    evidence_type: LineageEvidenceType | str,
    run_id: str,
    body: Any,
) -> str:
    """Content-address a typed receipt body the way preimage checks expect."""

    kind = _enum(evidence_type, LineageEvidenceType, field_name="evidence_type")
    return cid_for_preimage(
        {
            "evidence_type": kind.value,
            "run_id": _text(run_id, field_name="run_id"),
            "body": body,
        }
    )


def verify_lineage_merkle_root(
    manifest: ReasoningRunManifest,
    *,
    expected_root: str | None = None,
) -> str:
    """Recompute the ordered Merkle root and reject root forgery."""

    if not isinstance(manifest, ReasoningRunManifest):
        raise PlannerDoctorAttestationError("manifest must be a ReasoningRunManifest")
    recomputed = manifest.recompute_merkle_root()
    if recomputed != manifest.lineage_merkle_root:
        raise LineageRootError(
            "manifest lineage_merkle_root does not match recomputed ordered leaves"
        )
    if expected_root is not None and expected_root != recomputed:
        raise LineageRootError(
            "supplied lineage root does not match recomputed ordered leaves"
        )
    return recomputed


def require_run_replay(
    manifest: ReasoningRunManifest,
    *,
    run_id: str,
    repository_tree_id: str,
    policy_id: str,
    lineage_merkle_root: str,
    preimages: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on cross-run replay or drifted tree/policy/root bindings.

    Optional ``preimages`` also re-check every typed CID opening.
    """

    if not isinstance(manifest, ReasoningRunManifest):
        raise PlannerDoctorAttestationError("manifest must be a ReasoningRunManifest")
    wanted_run = _text(run_id, field_name="run_id")
    if manifest.run_id != wanted_run:
        raise LineageReplayError(
            "manifest run_id mismatch: expected %s got %s"
            % (wanted_run, manifest.run_id)
        )
    wanted_tree = _text(repository_tree_id, field_name="repository_tree_id")
    if manifest.repository_tree_id != wanted_tree:
        raise LineageReplayError(
            "manifest repository_tree_id mismatch for run replay"
        )
    wanted_policy = _text(policy_id, field_name="policy_id")
    if manifest.policy_id != wanted_policy:
        raise LineageReplayError("manifest policy_id mismatch for run replay")
    verify_lineage_merkle_root(manifest, expected_root=lineage_merkle_root)
    if preimages is not None:
        verify_lineage_preimages(manifest, preimages)


def reject_collapsed_evidence_types(manifest: ReasoningRunManifest) -> None:
    """Reject any attempt to treat distinct lineage CIDs as interchangeable."""

    if not isinstance(manifest, ReasoningRunManifest):
        raise PlannerDoctorAttestationError("manifest must be a ReasoningRunManifest")
    seen_types: set[str] = set()
    seen_cids: dict[str, str] = {}
    for slot in manifest.slots:
        kind = slot.evidence_type.value
        if kind in seen_types:
            raise LineageOrderError("duplicate evidence type %s" % kind)
        seen_types.add(kind)
        # Same CID reused across *different* types is still a collapse risk for
        # consumers that key only by CID; forbid it.
        owner = seen_cids.get(slot.receipt_cid)
        if owner is not None and owner != kind:
            raise LineageOrderError(
                "CID %s is shared by evidence types %s and %s; "
                "evidence types must not collapse"
                % (slot.receipt_cid, owner, kind)
            )
        seen_cids[slot.receipt_cid] = kind
    expected = set(LINEAGE_EVIDENCE_TYPES)
    if seen_types != expected:
        raise LineageOrderError(
            "lineage evidence types must be exactly %s; got %s"
            % (sorted(expected), sorted(seen_types))
        )


# ---------------------------------------------------------------------------
# Optional ZKP public inputs, witness, attestation, verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDoctorPublicInputs(CanonicalContract):
    """Public commitments for an optional planner/doctor ZKP statement."""

    SCHEMA: ClassVar[str] = PLANNER_DOCTOR_PUBLIC_INPUTS_SCHEMA

    run_id: str
    manifest_id: str
    lineage_merkle_root: str
    repository_tree_id: str
    policy_id: str
    circuit_id: str
    circuit_version: str
    proving_key_id: str
    verifying_key_id: str
    ceremony_id: str
    use_case_id: str = PLANNER_DOCTOR_ZKP_USE_CASE_ID
    threat_model_id: str = PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID
    public_input_codec_id: str = PUBLIC_INPUT_CODEC_ID
    public_input_codec_version: str = PUBLIC_INPUT_CODEC_VERSION
    predicate: PlannerDoctorZkpPredicate = (
        PlannerDoctorZkpPredicate.RECEIPT_LINEAGE
    )

    def __post_init__(self) -> None:
        for name in PUBLIC_COMMITMENT_KEYS:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "predicate",
            _enum(self.predicate, PlannerDoctorZkpPredicate, field_name="predicate"),
        )
        if self.public_input_codec_id != PUBLIC_INPUT_CODEC_ID:
            raise PlannerDoctorAttestationError(
                "public_input_codec_id must be %s" % PUBLIC_INPUT_CODEC_ID
            )
        if self.public_input_codec_version != PUBLIC_INPUT_CODEC_VERSION:
            raise PlannerDoctorAttestationError(
                "public_input_codec_version must be %s" % PUBLIC_INPUT_CODEC_VERSION
            )
        if self.use_case_id != PLANNER_DOCTOR_ZKP_USE_CASE_ID:
            raise PlannerDoctorAttestationError(
                "use_case_id must match the approved planner/doctor ZKP use case"
            )
        if self.threat_model_id != PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID:
            raise PlannerDoctorAttestationError(
                "threat_model_id must pin the approved planner/doctor ZKP threat model"
            )

    @property
    def public_inputs(self) -> Mapping[str, str]:
        return MappingProxyType(
            {key: getattr(self, key) for key in PUBLIC_COMMITMENT_KEYS}
        )

    @property
    def public_input_digest(self) -> str:
        return public_input_vector_digest(self.public_inputs)

    def with_overrides(self, **overrides: str) -> "PlannerDoctorPublicInputs":
        payload = {key: getattr(self, key) for key in PUBLIC_COMMITMENT_KEYS}
        payload["predicate"] = self.predicate.value
        payload.update(overrides)
        return PlannerDoctorPublicInputs(**payload)

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION,
            **{key: getattr(self, key) for key in PUBLIC_COMMITMENT_KEYS},
            "predicate": self.predicate.value,
            "public_input_digest": self.public_input_digest,
            "does_not_prove": sorted(ATTESTATION_DOES_NOT_PROVE),
            "scope": ATTESTATION_SCOPE_STATEMENT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorPublicInputs":
        _schema(payload, cls.SCHEMA)
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("does_not_prove", None)
        data.pop("scope", None)
        claimed_digest = data.pop("public_input_digest", None)
        claimed_id = data.pop("content_id", None)
        fields = {key: data.get(key, "") for key in PUBLIC_COMMITMENT_KEYS}
        fields["predicate"] = data.get(
            "predicate", PlannerDoctorZkpPredicate.RECEIPT_LINEAGE.value
        )
        result = cls(**fields)
        if claimed_digest not in (None, "", result.public_input_digest):
            raise LineageRootError(
                "public_input_digest does not match recomputed vector"
            )
        if claimed_id not in (None, "", result.content_id):
            raise LineageRootError(
                "public-input content identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        public = self.to_dict()
        reject_private_witness_from_public_payload(public)
        return public


def encode_public_input_vector(public_inputs: Mapping[str, str]) -> tuple[str, ...]:
    """Encode public inputs into the canonical ordered codec vector."""

    if not isinstance(public_inputs, Mapping):
        raise PlannerDoctorAttestationError("public_inputs must be a mapping")
    keys = tuple(public_inputs.keys())
    if set(keys) != set(PUBLIC_COMMITMENT_KEYS):
        missing = sorted(set(PUBLIC_COMMITMENT_KEYS) - set(keys))
        extra = sorted(set(keys) - set(PUBLIC_COMMITMENT_KEYS))
        raise PlannerDoctorAttestationError(
            "public_inputs keys mismatch; missing=%s extra=%s" % (missing, extra)
        )
    vector: list[str] = []
    for key in PUBLIC_COMMITMENT_KEYS:
        value = public_inputs[key]
        if not isinstance(value, str) or not value.strip():
            raise PlannerDoctorAttestationError(
                "public input %s must be a non-empty string" % key
            )
        vector.append(value.strip())
    return tuple(vector)


def public_input_vector_digest(public_inputs: Mapping[str, str]) -> str:
    """Content-address the canonical ordered public-input vector."""

    vector = encode_public_input_vector(public_inputs)
    return content_identity(
        {
            "codec_id": PUBLIC_INPUT_CODEC_ID,
            "codec_version": PUBLIC_INPUT_CODEC_VERSION,
            "vector": list(vector),
        }
    )


def build_public_inputs_from_manifest(
    manifest: ReasoningRunManifest,
    *,
    circuit_id: str = DEFAULT_LINEAGE_CIRCUIT_ID,
    circuit_version: str = DEFAULT_LINEAGE_CIRCUIT_VERSION,
    proving_key_id: str,
    verifying_key_id: str,
    ceremony_id: str,
    predicate: PlannerDoctorZkpPredicate | str = (
        PlannerDoctorZkpPredicate.RECEIPT_LINEAGE
    ),
) -> PlannerDoctorPublicInputs:
    """Bind optional ZKP public inputs to an existing verified lineage manifest."""

    if not isinstance(manifest, ReasoningRunManifest):
        raise PlannerDoctorAttestationError("manifest must be a ReasoningRunManifest")
    verify_lineage_merkle_root(manifest)
    reject_collapsed_evidence_types(manifest)
    return PlannerDoctorPublicInputs(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        lineage_merkle_root=manifest.lineage_merkle_root,
        repository_tree_id=manifest.repository_tree_id,
        policy_id=manifest.policy_id,
        circuit_id=circuit_id,
        circuit_version=circuit_version,
        proving_key_id=proving_key_id,
        verifying_key_id=verifying_key_id,
        ceremony_id=ceremony_id,
        predicate=predicate,
    )


class PrivatePlannerDoctorWitness:
    """Non-serializable private witness for optional planner/doctor ZKP.

    The witness never enters public envelopes, cache keys, logs, or manifests.
    """

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping) or not values:
            raise PlannerDoctorAttestationError(
                "private witness values must be a non-empty mapping"
            )
        normalized: Dict[str, Any] = {}
        for raw_name, value in values.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise PlannerDoctorAttestationError(
                    "witness field names must be non-empty strings"
                )
            normalized[raw_name.strip()] = value
        self.__values = dict(normalized)

    def __repr__(self) -> str:
        return "<PrivatePlannerDoctorWitness redacted>"

    __str__ = __repr__

    def __copy__(self) -> "PrivatePlannerDoctorWitness":
        raise WitnessDisclosureError("private witness cannot be copied")

    def __deepcopy__(self, memo: Any) -> "PrivatePlannerDoctorWitness":
        del memo
        raise WitnessDisclosureError("private witness cannot be copied")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise WitnessDisclosureError("private witness cannot be serialized or cached")

    def __getstate__(self) -> Any:
        raise WitnessDisclosureError("private witness cannot be serialized or cached")

    def to_dict(self) -> Dict[str, Any]:
        raise WitnessDisclosureError(
            "private witness has no public dictionary representation"
        )

    def use(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        if not callable(consumer):
            raise PlannerDoctorAttestationError("witness consumer must be callable")
        return consumer(MappingProxyType(self.__values))

    def redacted(self) -> Dict[str, bool]:
        return {"private_witness_redacted": True}


def reject_private_witness_from_public_payload(value: Any) -> None:
    """Reject private witness material from public receipts and statements."""

    if isinstance(value, PrivatePlannerDoctorWitness):
        raise WitnessDisclosureError(
            "private witness cannot enter a public planner/doctor artifact"
        )
    if isinstance(value, PlannerDoctorProvingRequest):
        raise WitnessDisclosureError(
            "proving requests cannot enter public planner/doctor artifacts"
        )
    if _public_payload_has_private_witness(value):
        raise WitnessDisclosureError(
            "private witness markers are rejected from public planner/doctor artifacts"
        )


def _public_payload_has_private_witness(value: Any) -> bool:
    if isinstance(value, PrivatePlannerDoctorWitness):
        return True
    if isinstance(value, PlannerDoctorProvingRequest):
        return True
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in _SAFE_PUBLIC_REDACTION_KEYS:
                if isinstance(item, bool):
                    continue
                return True
            if any(
                key == marker or key.endswith("_" + marker) or marker in key
                for marker in _PRIVATE_WITNESS_MARKERS
            ):
                if key not in _SAFE_PUBLIC_REDACTION_KEYS:
                    return True
            if _public_payload_has_private_witness(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_public_payload_has_private_witness(item) for item in value)
    return False


def public_planner_doctor_artifact(value: Any) -> Any:
    """Project a value into a public form, rejecting private witnesses."""

    if isinstance(value, PrivatePlannerDoctorWitness):
        raise WitnessDisclosureError(
            "private witness cannot enter a public planner/doctor artifact"
        )
    if isinstance(value, PlannerDoctorProvingRequest):
        return value.to_public_artifact()
    if isinstance(value, CanonicalContract):
        if hasattr(value, "to_public_artifact"):
            public = value.to_public_artifact()  # type: ignore[attr-defined]
        else:
            public = value.to_dict()
        reject_private_witness_from_public_payload(public)
        return public
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.strip().lower().replace("-", "_")
            if lowered in _SAFE_PUBLIC_REDACTION_KEYS and isinstance(item, bool):
                result[key_text] = item
                continue
            if any(
                lowered == marker or marker in lowered
                for marker in _PRIVATE_WITNESS_MARKERS
            ):
                if lowered not in _SAFE_PUBLIC_REDACTION_KEYS:
                    raise WitnessDisclosureError(
                        "private witness markers are rejected from public artifacts"
                    )
            result[key_text] = public_planner_doctor_artifact(item)
        reject_private_witness_from_public_payload(result)
        return result
    if isinstance(value, (list, tuple)):
        return [public_planner_doctor_artifact(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_public_artifact") and callable(value.to_public_artifact):
        public = value.to_public_artifact()
        reject_private_witness_from_public_payload(public)
        return public
    raise WitnessDisclosureError(
        "value of type %s is not a public planner/doctor artifact"
        % type(value).__name__
    )


@dataclass(frozen=True)
class PlannerDoctorProvingRequest:
    """Local proving request that retains a private witness off the public path."""

    public_inputs: PlannerDoctorPublicInputs
    witness: PrivatePlannerDoctorWitness
    backend_mode: PlannerDoctorBackendMode = PlannerDoctorBackendMode.SHADOW
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.public_inputs, PlannerDoctorPublicInputs):
            raise PlannerDoctorAttestationError(
                "public_inputs must be PlannerDoctorPublicInputs"
            )
        if not isinstance(self.witness, PrivatePlannerDoctorWitness):
            raise PlannerDoctorAttestationError(
                "witness must be a PrivatePlannerDoctorWitness"
            )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, PlannerDoctorBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "production_eligible",
            _boolean(self.production_eligible, field_name="production_eligible"),
        )
        if self.production_eligible and self.backend_mode is not (
            PlannerDoctorBackendMode.CRYPTOGRAPHIC
        ):
            raise AttestationBackendError(
                "only a cryptographic backend may be marked production_eligible"
            )
        if self.backend_mode in (
            PlannerDoctorBackendMode.SIMULATED,
            PlannerDoctorBackendMode.UNAVAILABLE,
            PlannerDoctorBackendMode.SHADOW,
        ):
            # Force non-production for non-cryptographic modes.
            object.__setattr__(self, "production_eligible", False)

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {
            "schema": PLANNER_DOCTOR_ATTESTATION_SCHEMA + "/proving-request-public@1",
            "public_inputs": self.public_inputs.to_public_artifact(),
            "backend_mode": self.backend_mode.value,
            "production_eligible": self.production_eligible,
            "private_witness_redacted": True,
            "does_not_prove": sorted(ATTESTATION_DOES_NOT_PROVE),
        }
        reject_private_witness_from_public_payload(public)
        return public


def prepare_planner_doctor_attestation(
    public_inputs: PlannerDoctorPublicInputs,
    *,
    witness: PrivatePlannerDoctorWitness,
    backend_mode: PlannerDoctorBackendMode | str = PlannerDoctorBackendMode.SHADOW,
    production_eligible: bool = False,
) -> PlannerDoctorProvingRequest:
    """Prepare a proving request without asserting assurance."""

    mode = _enum(backend_mode, PlannerDoctorBackendMode, field_name="backend_mode")
    return PlannerDoctorProvingRequest(
        public_inputs=public_inputs,
        witness=witness,
        backend_mode=mode,
        production_eligible=production_eligible,
    )


def planner_doctor_program_zkp_result_commitment(
    public_inputs: PlannerDoctorPublicInputs,
    *,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str,
) -> str:
    """Commit the exact Planner/Doctor proof claim verified by program ZKP.

    ``ProgramZkpVerificationReceipt`` uses a different, generic public-input
    codec.  Its ``result_commitment`` therefore binds this closed bridge record
    instead of being compared with the unrelated codec digest directly.  The
    bridge covers both the complete Planner/Doctor public-input vector and the
    proof artifact identity, preventing a valid receipt from being replayed
    over a different run, root, proof, circuit, or key.
    """

    if not isinstance(public_inputs, PlannerDoctorPublicInputs):
        raise PlannerDoctorAttestationError(
            "public_inputs must be PlannerDoctorPublicInputs"
        )
    return content_identity(
        {
            "schema": PLANNER_DOCTOR_PROGRAM_ZKP_BRIDGE_SCHEMA,
            "public_input_digest": public_inputs.public_input_digest,
            "proof_artifact_id": _text(
                proof_artifact_id, field_name="proof_artifact_id"
            ),
            "proof_digest": _text(proof_digest, field_name="proof_digest"),
            "prover_id": _text(prover_id, field_name="prover_id"),
            "predicate": public_inputs.predicate.value,
        }
    )


def _require_authoritative_program_zkp_receipt(
    *,
    public_inputs: PlannerDoctorPublicInputs,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str,
    receipt: ProgramZkpVerificationReceipt | None,
) -> ProgramZkpVerificationReceipt:
    """Fail closed unless an independent cryptographic receipt binds the claim."""

    if not isinstance(receipt, ProgramZkpVerificationReceipt):
        raise AttestationBackendError(
            "ATTESTED requires an independently verified "
            "ProgramZkpVerificationReceipt"
        )
    if not receipt.authoritative:
        raise AttestationBackendError(
            "ProgramZkpVerificationReceipt is not authoritative; simulated, "
            "shadow, rejected, stale, or self-verified receipts cannot attest"
        )
    if (
        receipt.backend_mode is not ProgramAnalysisZkpBackendMode.CRYPTOGRAPHIC
        or receipt.statement.backend_mode
        is not ProgramAnalysisZkpBackendMode.CRYPTOGRAPHIC
    ):
        raise AttestationBackendError(
            "ProgramZkpVerificationReceipt must come from a cryptographic "
            "statement and backend"
        )

    pins = receipt.statement.public_inputs
    expected_bindings = {
        "forest_commitment": public_inputs.repository_tree_id,
        "inventory_commitment": public_inputs.manifest_id,
        "contract_commitment": public_inputs.policy_id,
        "call_slice_commitment": public_inputs.lineage_merkle_root,
        "assumptions_commitment": public_inputs.public_input_digest,
        "result_commitment": planner_doctor_program_zkp_result_commitment(
            public_inputs,
            proof_artifact_id=proof_artifact_id,
            proof_digest=proof_digest,
            prover_id=prover_id,
        ),
        "circuit_id": public_inputs.circuit_id,
        "proving_key_id": public_inputs.proving_key_id,
        "verifying_key_id": public_inputs.verifying_key_id,
        "ceremony_id": public_inputs.ceremony_id,
        "public_input_codec_version": public_inputs.public_input_codec_version,
    }
    mismatches = tuple(
        name
        for name, expected in expected_bindings.items()
        if getattr(pins, name) != expected
    )
    if mismatches:
        raise AttestationBackendError(
            "ProgramZkpVerificationReceipt does not bind the Planner/Doctor "
            f"claim: {', '.join(mismatches)}"
        )

    # Reparse the public receipt and replay its exact pins.  Authority flags are
    # deliberately re-derived by ProgramZkpVerificationReceipt.from_dict.
    replayed = ProgramZkpVerificationReceipt.from_dict(receipt.to_public_artifact())
    if replayed.receipt_id != receipt.receipt_id or not replayed.authoritative:
        raise AttestationBackendError(
            "ProgramZkpVerificationReceipt identity or authority failed replay"
        )
    replayed.require_replay(
        public_inputs=pins,
        verifying_key_id=public_inputs.verifying_key_id,
        circuit_id=public_inputs.circuit_id,
        ceremony_id=public_inputs.ceremony_id,
        public_input_codec_version=pins.public_input_codec_version,
        capability_epoch=receipt.capability_epoch,
    )
    return replayed


@dataclass(frozen=True)
class PlannerDoctorAttestation(CanonicalContract):
    """``PlannerDoctorAttestation@1`` — optional public ZKP envelope.

    The envelope binds fixed circuit/code pins and public inputs.  It never
    carries the private witness.  Authority is derived only after independent
    verification of a cryptographic, production-eligible backend.
    """

    SCHEMA: ClassVar[str] = PLANNER_DOCTOR_ATTESTATION_SCHEMA

    public_inputs: PlannerDoctorPublicInputs
    proof_artifact_id: str
    proof_digest: str
    prover_id: str
    backend_mode: PlannerDoctorBackendMode = PlannerDoctorBackendMode.SHADOW
    production_eligible: bool = False
    status: PlannerDoctorAttestationStatus = PlannerDoctorAttestationStatus.GENERATED
    program_zkp_verification_receipt: ProgramZkpVerificationReceipt | None = None

    def __post_init__(self) -> None:
        if isinstance(self.public_inputs, Mapping):
            object.__setattr__(
                self,
                "public_inputs",
                PlannerDoctorPublicInputs.from_dict(self.public_inputs),
            )
        if not isinstance(self.public_inputs, PlannerDoctorPublicInputs):
            raise PlannerDoctorAttestationError(
                "public_inputs must be PlannerDoctorPublicInputs"
            )
        object.__setattr__(
            self,
            "proof_artifact_id",
            _text(self.proof_artifact_id, field_name="proof_artifact_id"),
        )
        object.__setattr__(
            self,
            "proof_digest",
            _text(self.proof_digest, field_name="proof_digest"),
        )
        object.__setattr__(
            self, "prover_id", _text(self.prover_id, field_name="prover_id")
        )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, PlannerDoctorBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "production_eligible",
            _boolean(self.production_eligible, field_name="production_eligible"),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, PlannerDoctorAttestationStatus, field_name="status"),
        )
        receipt = self.program_zkp_verification_receipt
        if isinstance(receipt, Mapping):
            receipt = ProgramZkpVerificationReceipt.from_dict(receipt)
            object.__setattr__(self, "program_zkp_verification_receipt", receipt)
        elif receipt is not None and not isinstance(
            receipt, ProgramZkpVerificationReceipt
        ):
            raise AttestationBackendError(
                "program_zkp_verification_receipt must be a "
                "ProgramZkpVerificationReceipt"
            )
        if self.backend_mode is not PlannerDoctorBackendMode.CRYPTOGRAPHIC:
            object.__setattr__(self, "production_eligible", False)
            if self.status is PlannerDoctorAttestationStatus.ATTESTED:
                raise AttestationBackendError(
                    "non-cryptographic backends cannot emit status=attested"
                )
        if (
            self.production_eligible
            and self.backend_mode is not PlannerDoctorBackendMode.CRYPTOGRAPHIC
        ):
            raise AttestationBackendError(
                "production_eligible requires cryptographic backend_mode"
            )
        if self.status is PlannerDoctorAttestationStatus.ATTESTED and not (
            self.production_eligible
            and self.backend_mode is PlannerDoctorBackendMode.CRYPTOGRAPHIC
        ):
            raise AttestationBackendError(
                "status=attested requires a production-eligible cryptographic backend"
            )
        if self.status is PlannerDoctorAttestationStatus.ATTESTED:
            _require_authoritative_program_zkp_receipt(
                public_inputs=self.public_inputs,
                proof_artifact_id=self.proof_artifact_id,
                proof_digest=self.proof_digest,
                prover_id=self.prover_id,
                receipt=receipt,
            )
        elif receipt is not None:
            raise AttestationBackendError(
                "ProgramZkpVerificationReceipt may only be attached to an "
                "ATTESTED envelope"
            )

    @property
    def attestation_id(self) -> str:
        return self.content_id

    @property
    def interface(self) -> str:
        return PLANNER_DOCTOR_ATTESTATION_INTERFACE

    @property
    def simulated(self) -> bool:
        return self.backend_mode is PlannerDoctorBackendMode.SIMULATED

    @property
    def unavailable(self) -> bool:
        return self.backend_mode is PlannerDoctorBackendMode.UNAVAILABLE or (
            self.status is PlannerDoctorAttestationStatus.UNAVAILABLE
        )

    @property
    def failed(self) -> bool:
        return self.status in (
            PlannerDoctorAttestationStatus.FAILED,
            PlannerDoctorAttestationStatus.ERROR,
            PlannerDoctorAttestationStatus.REJECTED,
        )

    def _payload(self) -> Dict[str, Any]:
        payload = {
            "contract_version": PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION,
            "interface": PLANNER_DOCTOR_ATTESTATION_INTERFACE,
            "public_inputs": self.public_inputs.to_dict(),
            "public_input_digest": self.public_inputs.public_input_digest,
            "proof_artifact_id": self.proof_artifact_id,
            "proof_digest": self.proof_digest,
            "prover_id": self.prover_id,
            "backend_mode": self.backend_mode.value,
            "production_eligible": self.production_eligible,
            "status": self.status.value,
            "private_witness_redacted": True,
            "does_not_prove": sorted(ATTESTATION_DOES_NOT_PROVE),
            "scope": ATTESTATION_SCOPE_STATEMENT,
            "threat_model_id": PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID,
            "use_case_id": PLANNER_DOCTOR_ZKP_USE_CASE_ID,
        }
        # Preserve content identities of existing non-attested @1 envelopes;
        # the additive receipt is present only on the newly admitted seal path.
        if self.program_zkp_verification_receipt is not None:
            payload["program_zkp_verification_receipt"] = (
                self.program_zkp_verification_receipt.to_public_artifact()
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorAttestation":
        _schema(payload, cls.SCHEMA)
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("interface", None)
        data.pop("does_not_prove", None)
        data.pop("scope", None)
        data.pop("threat_model_id", None)
        data.pop("use_case_id", None)
        data.pop("private_witness_redacted", None)
        claimed_digest = data.pop("public_input_digest", None)
        claimed_id = data.pop("content_id", None) or data.pop("attestation_id", None)
        public_raw = data.get("public_inputs") or {}
        if isinstance(public_raw, PlannerDoctorPublicInputs):
            public_inputs = public_raw
        else:
            public_inputs = PlannerDoctorPublicInputs.from_dict(public_raw)
        result = cls(
            public_inputs=public_inputs,
            proof_artifact_id=data.get("proof_artifact_id", ""),
            proof_digest=data.get("proof_digest", ""),
            prover_id=data.get("prover_id", ""),
            backend_mode=data.get("backend_mode", PlannerDoctorBackendMode.SHADOW),
            production_eligible=bool(data.get("production_eligible", False)),
            status=data.get("status", PlannerDoctorAttestationStatus.GENERATED),
            program_zkp_verification_receipt=data.get(
                "program_zkp_verification_receipt"
            ),
        )
        if claimed_digest not in (None, "", result.public_inputs.public_input_digest):
            raise LineageRootError(
                "envelope public_input_digest does not match public inputs"
            )
        if claimed_id not in (None, "", result.attestation_id):
            raise LineageRootError(
                "attestation identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {**self.to_dict(), "attestation_id": self.attestation_id}
        reject_private_witness_from_public_payload(public)
        return public


def create_planner_doctor_attestation(
    request: PlannerDoctorProvingRequest,
    *,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str,
    status: PlannerDoctorAttestationStatus | str | None = None,
) -> PlannerDoctorAttestation:
    """Seal a public envelope from a proving request (witness stays local)."""

    if not isinstance(request, PlannerDoctorProvingRequest):
        raise PlannerDoctorAttestationError(
            "request must be a PlannerDoctorProvingRequest"
        )
    mode = request.backend_mode
    if status is None:
        if mode is PlannerDoctorBackendMode.UNAVAILABLE:
            resolved = PlannerDoctorAttestationStatus.UNAVAILABLE
        elif mode is PlannerDoctorBackendMode.SIMULATED:
            resolved = PlannerDoctorAttestationStatus.SIMULATED
        elif mode is PlannerDoctorBackendMode.SHADOW:
            resolved = PlannerDoctorAttestationStatus.CANDIDATE
        elif mode is PlannerDoctorBackendMode.CRYPTOGRAPHIC:
            # Cryptographic proofs still require independent verification
            # before status may become ATTESTED.
            resolved = PlannerDoctorAttestationStatus.GENERATED
        else:
            resolved = PlannerDoctorAttestationStatus.CANDIDATE
    else:
        resolved = _enum(status, PlannerDoctorAttestationStatus, field_name="status")
    # Hard gate: non-crypto paths never seal as ATTESTED.
    if (
        resolved is PlannerDoctorAttestationStatus.ATTESTED
        and mode is not PlannerDoctorBackendMode.CRYPTOGRAPHIC
    ):
        raise AttestationBackendError(
            "unavailable/failed/simulated/shadow backends cannot emit ATTESTED"
        )
    return PlannerDoctorAttestation(
        public_inputs=request.public_inputs,
        proof_artifact_id=proof_artifact_id,
        proof_digest=proof_digest,
        prover_id=prover_id,
        backend_mode=mode,
        production_eligible=request.production_eligible,
        status=resolved,
    )


def create_unavailable_attestation(
    public_inputs: PlannerDoctorPublicInputs,
    *,
    reason_code: str = "backend_unavailable",
) -> PlannerDoctorAttestation:
    """Emit a typed unavailable envelope that cannot contribute ATTESTED."""

    del reason_code  # retained for call-site diagnostics without public claim
    return PlannerDoctorAttestation(
        public_inputs=public_inputs,
        proof_artifact_id="artifact:unavailable",
        proof_digest="sha256:" + ("00" * 32),
        prover_id="prover:unavailable",
        backend_mode=PlannerDoctorBackendMode.UNAVAILABLE,
        production_eligible=False,
        status=PlannerDoctorAttestationStatus.UNAVAILABLE,
    )


def create_simulated_attestation(
    public_inputs: PlannerDoctorPublicInputs,
    *,
    proof_artifact_id: str = "artifact:zk-proof-simulated",
    proof_digest: str | None = None,
    prover_id: str = "prover:simulated",
) -> PlannerDoctorAttestation:
    """Emit a simulated envelope for serialization tests only (sim ≠ ATTESTED)."""

    digest = proof_digest or ("sha256:" + ("ab" * 32))
    return PlannerDoctorAttestation(
        public_inputs=public_inputs,
        proof_artifact_id=proof_artifact_id,
        proof_digest=digest,
        prover_id=prover_id,
        backend_mode=PlannerDoctorBackendMode.SIMULATED,
        production_eligible=False,
        status=PlannerDoctorAttestationStatus.SIMULATED,
    )


def create_failed_attestation(
    public_inputs: PlannerDoctorPublicInputs,
    *,
    proof_artifact_id: str = "artifact:zk-proof-failed",
    proof_digest: str | None = None,
    prover_id: str = "prover:failed",
) -> PlannerDoctorAttestation:
    """Emit a failed envelope that remains non-authoritative."""

    digest = proof_digest or ("sha256:" + ("cd" * 32))
    return PlannerDoctorAttestation(
        public_inputs=public_inputs,
        proof_artifact_id=proof_artifact_id,
        proof_digest=digest,
        prover_id=prover_id,
        backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
        production_eligible=False,
        status=PlannerDoctorAttestationStatus.FAILED,
    )


@dataclass(frozen=True)
class PlannerDoctorVerification(CanonicalContract):
    """Independent verification result for a planner/doctor attestation envelope.

    Authority is derived here, never asserted by the prover.
    """

    SCHEMA: ClassVar[str] = PLANNER_DOCTOR_VERIFICATION_SCHEMA

    envelope: PlannerDoctorAttestation
    verdict: AttestationVerificationVerdict
    verifier_id: str
    independent: bool = True
    expected_public_input_digest: str = ""
    expected_lineage_merkle_root: str = ""
    expected_run_id: str = ""
    diagnostic_code: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.envelope, Mapping):
            object.__setattr__(
                self,
                "envelope",
                PlannerDoctorAttestation.from_dict(self.envelope),
            )
        if not isinstance(self.envelope, PlannerDoctorAttestation):
            raise PlannerDoctorAttestationError(
                "envelope must be a PlannerDoctorAttestation"
            )
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, AttestationVerificationVerdict, field_name="verdict"),
        )
        object.__setattr__(
            self, "verifier_id", _text(self.verifier_id, field_name="verifier_id")
        )
        object.__setattr__(
            self, "independent", _boolean(self.independent, field_name="independent")
        )
        object.__setattr__(
            self,
            "expected_public_input_digest",
            _text(
                self.expected_public_input_digest,
                field_name="expected_public_input_digest",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "expected_lineage_merkle_root",
            _text(
                self.expected_lineage_merkle_root,
                field_name="expected_lineage_merkle_root",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "expected_run_id",
            _text(self.expected_run_id, field_name="expected_run_id", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_code",
            _text(self.diagnostic_code, field_name="diagnostic_code", required=False),
        )

    @property
    def verification_id(self) -> str:
        return self.content_id

    @property
    def verified(self) -> bool:
        return self.verdict is AttestationVerificationVerdict.VERIFIED

    @property
    def simulated(self) -> bool:
        return self.envelope.simulated

    @property
    def authoritative(self) -> bool:
        receipt = self.envelope.program_zkp_verification_receipt
        return (
            self.verified
            and self.independent
            and self.envelope.production_eligible
            and self.envelope.backend_mode
            is PlannerDoctorBackendMode.CRYPTOGRAPHIC
            and self.envelope.status
            is PlannerDoctorAttestationStatus.ATTESTED
            and receipt is not None
            and receipt.authoritative
        )

    @property
    def trust(self) -> AttestationTrust:
        if self.authoritative:
            return AttestationTrust.AUTHORITATIVE
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        """Assurance contributed by this verification alone.

        Simulated, unavailable, failed, shadow, or non-independent results
        never contribute ``ATTESTED``.  Non-authoritative cryptographic
        candidates contribute at most ``CANDIDATE``.
        """

        if self.authoritative:
            return AssuranceLevel.ATTESTED
        if self.envelope.backend_mode is PlannerDoctorBackendMode.UNAVAILABLE:
            return AssuranceLevel.UNVERIFIED
        if self.envelope.status in (
            PlannerDoctorAttestationStatus.FAILED,
            PlannerDoctorAttestationStatus.ERROR,
            PlannerDoctorAttestationStatus.REJECTED,
            PlannerDoctorAttestationStatus.UNAVAILABLE,
        ):
            return AssuranceLevel.UNVERIFIED
        if self.simulated or self.envelope.backend_mode in (
            PlannerDoctorBackendMode.SIMULATED,
            PlannerDoctorBackendMode.SHADOW,
        ):
            return AssuranceLevel.CANDIDATE
        if self.verified and not self.authoritative:
            return AssuranceLevel.CANDIDATE
        return AssuranceLevel.UNVERIFIED

    def satisfies_gate(self, gate: AttestationGate | str) -> bool:
        normalized = _enum(gate, AttestationGate, field_name="gate")
        if normalized in (AttestationGate.SERIALIZATION, AttestationGate.TEST):
            return self.verified
        return self.authoritative

    def satisfies_production_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.PRODUCTION)

    def satisfies_completion_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.COMPLETION)

    def require_replay(
        self,
        *,
        public_input_digest: str,
        lineage_merkle_root: str,
        run_id: str,
        verifying_key_id: str | None = None,
        circuit_id: str | None = None,
    ) -> None:
        """Independent consumer replay against pinned public bindings."""

        wanted_digest = _text(public_input_digest, field_name="public_input_digest")
        if self.envelope.public_inputs.public_input_digest != wanted_digest:
            raise LineageReplayError(
                "verification replay failed: public_input_digest mismatch"
            )
        if (
            self.expected_public_input_digest
            and self.expected_public_input_digest != wanted_digest
        ):
            raise LineageReplayError(
                "verification replay failed: expected public_input_digest mismatch"
            )
        wanted_root = _text(lineage_merkle_root, field_name="lineage_merkle_root")
        if self.envelope.public_inputs.lineage_merkle_root != wanted_root:
            raise LineageReplayError(
                "verification replay failed: lineage_merkle_root mismatch"
            )
        wanted_run = _text(run_id, field_name="run_id")
        if self.envelope.public_inputs.run_id != wanted_run:
            raise LineageReplayError(
                "verification replay failed: run_id mismatch (cross-run replay)"
            )
        if verifying_key_id is not None:
            wanted_vk = _text(verifying_key_id, field_name="verifying_key_id")
            if self.envelope.public_inputs.verifying_key_id != wanted_vk:
                raise LineageReplayError(
                    "verification replay failed: verifying_key_id mismatch"
                )
        if circuit_id is not None:
            wanted_circuit = _text(circuit_id, field_name="circuit_id")
            if self.envelope.public_inputs.circuit_id != wanted_circuit:
                raise LineageReplayError(
                    "verification replay failed: circuit_id mismatch"
                )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION,
            "envelope": self.envelope.to_dict(),
            "verdict": self.verdict.value,
            "verifier_id": self.verifier_id,
            "independent": self.independent,
            "expected_public_input_digest": self.expected_public_input_digest,
            "expected_lineage_merkle_root": self.expected_lineage_merkle_root,
            "expected_run_id": self.expected_run_id,
            "diagnostic_code": self.diagnostic_code,
            "authoritative": self.authoritative,
            "authoritative_assurance": self.authoritative_assurance.value,
            "trust": self.trust.value,
            "does_not_prove": sorted(ATTESTATION_DOES_NOT_PROVE),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorVerification":
        _schema(payload, cls.SCHEMA)
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("does_not_prove", None)
        # Authority flags in serialized form are never trusted — re-derived.
        data.pop("authoritative", None)
        data.pop("authoritative_assurance", None)
        data.pop("trust", None)
        claimed_id = data.pop("content_id", None) or data.pop("verification_id", None)
        result = cls(
            envelope=data.get("envelope") or {},
            verdict=data.get("verdict", ""),
            verifier_id=data.get("verifier_id", ""),
            independent=bool(data.get("independent", True)),
            expected_public_input_digest=data.get("expected_public_input_digest", ""),
            expected_lineage_merkle_root=data.get("expected_lineage_merkle_root", ""),
            expected_run_id=data.get("expected_run_id", ""),
            diagnostic_code=data.get("diagnostic_code", ""),
        )
        if claimed_id not in (None, "", result.verification_id):
            raise LineageRootError(
                "verification identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {**self.to_dict(), "verification_id": self.verification_id}
        reject_private_witness_from_public_payload(public)
        return public


def verify_planner_doctor_attestation(
    envelope: PlannerDoctorAttestation,
    *,
    verifier_id: str,
    expected_public_input_digest: str,
    expected_lineage_merkle_root: str,
    expected_run_id: str,
    independent: bool = True,
    accept_simulated: bool = False,
) -> PlannerDoctorVerification:
    """Independently verify a public envelope against pinned public bindings.

    Simulated / unavailable / failed backends may yield a typed non-authoritative
    result for test or observation paths, but never production ``ATTESTED``.
    """

    if not isinstance(envelope, PlannerDoctorAttestation):
        if isinstance(envelope, Mapping):
            envelope = PlannerDoctorAttestation.from_dict(envelope)
        else:
            raise PlannerDoctorAttestationError(
                "envelope must be a PlannerDoctorAttestation"
            )

    digest_ok = (
        envelope.public_inputs.public_input_digest == expected_public_input_digest
    )
    root_ok = (
        envelope.public_inputs.lineage_merkle_root == expected_lineage_merkle_root
    )
    run_ok = envelope.public_inputs.run_id == expected_run_id

    if envelope.backend_mode is PlannerDoctorBackendMode.UNAVAILABLE:
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="backend_unavailable",
        )
    if envelope.status in (
        PlannerDoctorAttestationStatus.FAILED,
        PlannerDoctorAttestationStatus.ERROR,
    ):
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="backend_failed",
        )
    if not (digest_ok and root_ok and run_ok):
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.REJECTED,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="public_binding_mismatch",
        )
    if envelope.simulated or envelope.backend_mode is PlannerDoctorBackendMode.SIMULATED:
        # Simulated proofs may be structure-checked for tests when the caller
        # opts in, but still never become authoritative / ATTESTED.
        if not accept_simulated:
            return PlannerDoctorVerification(
                envelope=envelope,
                verdict=AttestationVerificationVerdict.REJECTED,
                verifier_id=verifier_id,
                independent=independent,
                expected_public_input_digest=expected_public_input_digest,
                expected_lineage_merkle_root=expected_lineage_merkle_root,
                expected_run_id=expected_run_id,
                diagnostic_code="simulated_rejected",
            )
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.VERIFIED,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="simulated_non_authoritative",
        )
    if envelope.backend_mode is PlannerDoctorBackendMode.SHADOW:
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.VERIFIED,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="shadow_non_authoritative",
        )
    if envelope.backend_mode is PlannerDoctorBackendMode.CRYPTOGRAPHIC:
        # Without an installed real prover, a cryptographic envelope that is
        # not yet marked ATTESTED stays candidate; promotion to ATTESTED is an
        # explicit sealing step after a real backend verifies.
        if envelope.status is PlannerDoctorAttestationStatus.ATTESTED:
            if not envelope.production_eligible:
                return PlannerDoctorVerification(
                    envelope=envelope,
                    verdict=AttestationVerificationVerdict.REJECTED,
                    verifier_id=verifier_id,
                    independent=independent,
                    expected_public_input_digest=expected_public_input_digest,
                    expected_lineage_merkle_root=expected_lineage_merkle_root,
                    expected_run_id=expected_run_id,
                    diagnostic_code="not_production_eligible",
                )
            return PlannerDoctorVerification(
                envelope=envelope,
                verdict=AttestationVerificationVerdict.VERIFIED,
                verifier_id=verifier_id,
                independent=independent,
                expected_public_input_digest=expected_public_input_digest,
                expected_lineage_merkle_root=expected_lineage_merkle_root,
                expected_run_id=expected_run_id,
                diagnostic_code="cryptographic_verified",
            )
        return PlannerDoctorVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.VERIFIED,
            verifier_id=verifier_id,
            independent=independent,
            expected_public_input_digest=expected_public_input_digest,
            expected_lineage_merkle_root=expected_lineage_merkle_root,
            expected_run_id=expected_run_id,
            diagnostic_code="cryptographic_candidate",
        )
    return PlannerDoctorVerification(
        envelope=envelope,
        verdict=AttestationVerificationVerdict.ERROR,
        verifier_id=verifier_id,
        independent=independent,
        expected_public_input_digest=expected_public_input_digest,
        expected_lineage_merkle_root=expected_lineage_merkle_root,
        expected_run_id=expected_run_id,
        diagnostic_code="unsupported_backend_mode",
    )


def seal_cryptographic_attested(
    envelope: PlannerDoctorAttestation,
    *,
    verification_receipt: ProgramZkpVerificationReceipt | None = None,
) -> PlannerDoctorAttestation:
    """Promote a production-eligible cryptographic envelope to status=attested.

    Promotion consumes and embeds an independently replayable, authoritative
    ``ProgramZkpVerificationReceipt`` bound to the exact public inputs and
    proof artifact.  Caller-supplied mode/status/eligibility flags are never
    sufficient.  Simulated / unavailable / failed envelopes are rejected.
    """

    if not isinstance(envelope, PlannerDoctorAttestation):
        raise PlannerDoctorAttestationError(
            "envelope must be a PlannerDoctorAttestation"
        )
    if envelope.backend_mode is not PlannerDoctorBackendMode.CRYPTOGRAPHIC:
        raise AttestationBackendError(
            "only cryptographic backends may be sealed as ATTESTED"
        )
    if not envelope.production_eligible:
        raise AttestationBackendError(
            "envelope is not production_eligible; cannot seal ATTESTED"
        )
    if envelope.failed or envelope.unavailable or envelope.simulated:
        raise AttestationBackendError(
            "failed/unavailable/simulated envelopes cannot be sealed as ATTESTED"
        )
    checked_receipt = _require_authoritative_program_zkp_receipt(
        public_inputs=envelope.public_inputs,
        proof_artifact_id=envelope.proof_artifact_id,
        proof_digest=envelope.proof_digest,
        prover_id=envelope.prover_id,
        receipt=verification_receipt,
    )
    return PlannerDoctorAttestation(
        public_inputs=envelope.public_inputs,
        proof_artifact_id=envelope.proof_artifact_id,
        proof_digest=envelope.proof_digest,
        prover_id=envelope.prover_id,
        backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
        production_eligible=True,
        status=PlannerDoctorAttestationStatus.ATTESTED,
        program_zkp_verification_receipt=checked_receipt,
    )


def simulated_attestation_cannot_satisfy_attested(
    verification: PlannerDoctorVerification | Mapping[str, Any],
) -> bool:
    """Return True when a simulated path is correctly barred from ATTESTED."""

    checked = (
        verification
        if isinstance(verification, PlannerDoctorVerification)
        else PlannerDoctorVerification.from_dict(verification)
    )
    if checked.simulated or checked.envelope.backend_mode in (
        PlannerDoctorBackendMode.SIMULATED,
        PlannerDoctorBackendMode.SHADOW,
        PlannerDoctorBackendMode.UNAVAILABLE,
    ):
        return (
            checked.authoritative_assurance is not AssuranceLevel.ATTESTED
            and not checked.authoritative
            and not checked.satisfies_gate(AttestationGate.PRODUCTION)
            and not checked.satisfies_gate(AttestationGate.COMPLETION)
        )
    return checked.authoritative_assurance is AssuranceLevel.ATTESTED


def attestation_independent_of_semantic_authority(
    verification: PlannerDoctorVerification | Mapping[str, Any],
) -> bool:
    """Return True when verification does not claim forbidden semantic authority."""

    checked = (
        verification
        if isinstance(verification, PlannerDoctorVerification)
        else PlannerDoctorVerification.from_dict(verification)
    )
    public = checked.to_public_artifact()
    for claim in ATTESTATION_DOES_NOT_PROVE:
        # Scope text may *mention* non-claims; ensure no positive assertion flag.
        if public.get(claim) is True:
            return False
        if public.get("claims", {}).get(claim) is True:  # type: ignore[union-attr]
            return False
    scope = public.get("does_not_prove") or []
    return set(ATTESTATION_DOES_NOT_PROVE).issubset(set(scope))


def backend_mode_to_attestation_backend_mode(
    mode: PlannerDoctorBackendMode | str,
) -> AttestationBackendMode:
    """Map planner/doctor backend modes onto the shared attestation vocabulary."""

    normalized = _enum(mode, PlannerDoctorBackendMode, field_name="backend_mode")
    if normalized is PlannerDoctorBackendMode.CRYPTOGRAPHIC:
        return AttestationBackendMode.CRYPTOGRAPHIC
    return AttestationBackendMode.SIMULATED


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "ATTESTATION_DOES_NOT_PROVE",
    "ATTESTATION_SCOPE_STATEMENT",
    "CONTRACT_VERSION",
    "DEFAULT_LINEAGE_CIRCUIT_ID",
    "DEFAULT_LINEAGE_CIRCUIT_VERSION",
    "LINEAGE_EVIDENCE_TYPES",
    "PLANNER_DOCTOR_ATTESTATION_CONTRACT_VERSION",
    "PLANNER_DOCTOR_ATTESTATION_INTERFACE",
    "PLANNER_DOCTOR_ATTESTATION_SCHEMA",
    "PLANNER_DOCTOR_PUBLIC_INPUTS_SCHEMA",
    "PLANNER_DOCTOR_PROGRAM_ZKP_BRIDGE_SCHEMA",
    "PLANNER_DOCTOR_VERIFICATION_SCHEMA",
    "PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID",
    "PLANNER_DOCTOR_ZKP_USE_CASE_ID",
    "PUBLIC_COMMITMENT_KEYS",
    "PUBLIC_INPUT_CODEC_ID",
    "PUBLIC_INPUT_CODEC_VERSION",
    "REASONING_RUN_MANIFEST_INTERFACE",
    "REASONING_RUN_MANIFEST_SCHEMA",
    "AttestationBackendError",
    "AttestationClaimPromotionError",
    "LineageEvidenceType",
    "LineageOrderError",
    "LineagePreimageError",
    "LineageReplayError",
    "LineageRootError",
    "LineageSlot",
    "LineageValidationError",
    "PlannerDoctorAttestation",
    "PlannerDoctorAttestationError",
    "PlannerDoctorAttestationStatus",
    "PlannerDoctorBackendMode",
    "PlannerDoctorProvingRequest",
    "PlannerDoctorPublicInputs",
    "PlannerDoctorVerification",
    "PlannerDoctorZkpPredicate",
    "PrivatePlannerDoctorWitness",
    "ReasoningRunManifest",
    "WitnessDisclosureError",
    "attestation_does_not_prove",
    "attestation_independent_of_semantic_authority",
    "backend_mode_to_attestation_backend_mode",
    "build_public_inputs_from_manifest",
    "build_reasoning_run_manifest",
    "cid_for_preimage",
    "create_failed_attestation",
    "create_planner_doctor_attestation",
    "create_simulated_attestation",
    "create_unavailable_attestation",
    "encode_public_input_vector",
    "leaf_digest_for_slot",
    "merkle_root_from_leaves",
    "prepare_planner_doctor_attestation",
    "planner_doctor_program_zkp_result_commitment",
    "public_input_vector_digest",
    "public_planner_doctor_artifact",
    "reject_collapsed_evidence_types",
    "reject_illegal_semantic_claim",
    "reject_private_witness_from_public_payload",
    "require_run_replay",
    "seal_cryptographic_attested",
    "simulated_attestation_cannot_satisfy_attested",
    "typed_receipt_cid",
    "verify_lineage_merkle_root",
    "verify_lineage_preimages",
    "verify_planner_doctor_attestation",
]
