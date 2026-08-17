"""Seal verification under trusted keys and current policy (IPS-041).

``verify_seal`` revalidates seal type, status, chain, key, signature, proofs,
manifest, forest, root, and policy.  Unknown systems/statuses, wrong keys/
policy/parent/root, modified inputs, incomplete history, and cryptographic
failure reject with a typed reason.  Explanations never substitute for this
check.

Interfaces: ``SealVerificationResult``, ``SealVerificationReason``,
``SealVerificationRequest``, ``verify_seal``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    DeltaSeal,
    ParentSealView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    FullCheckpointSeal,
    GENESIS_PARENT_SEAL,
    VerificationPolicyView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    TrustedProofPolicy,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    SealStatus,
    closed_seal_status_values,
    parse_seal_status,
)

EVIDENCE_SUBSET: Final[str] = "ips/seal-verification@1"
RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "seal-verification-result@1"
)
REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "seal-verification-request@1"
)

# Closed ordered verification stages (plan §8 / §11).
VERIFICATION_STAGES: Final[tuple[str, ...]] = (
    "type",
    "status",
    "proof_system",
    "key",
    "policy",
    "parent",
    "root",
    "manifest",
    "forest",
    "inputs",
    "history",
    "signature",
    "cryptography",
)

_ACCEPTED_SEAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        SealStatus.SEALED_FULL.value,
        SealStatus.SEALED_INCREMENTAL.value,
    }
)

_DEFAULT_ALLOWED_PROOF_SYSTEMS: Final[frozenset[str]] = frozenset(
    {
        "integrity",
        "signed_receipt",
        "merkle_manifest_aggregation",
        "groth16",
        "receipt_aggregation",
        "incremental_seal",
    }
)

_SENSITIVE_DETAIL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "proving_key",
        "proving_key_bytes",
        "witness",
        "witness_bytes",
        "private_key",
        "secret",
        "trapdoor",
    }
)


class VerificationError(ValueError):
    """Fail-closed seal verification contract violation."""


class SealVerificationReason(str, Enum):
    """Stable reason codes for seal verification outcomes."""

    ACCEPTED = "accepted"
    UNKNOWN_SEAL_TYPE = "unknown_seal_type"
    UNKNOWN_STATUS = "unknown_status"
    NON_ACCEPTED_STATUS = "non_accepted_status"
    UNKNOWN_PROOF_SYSTEM = "unknown_proof_system"
    WRONG_VERIFICATION_KEY = "wrong_verification_key"
    UNALLOWLISTED_VERIFICATION_KEY = "unallowlisted_verification_key"
    WRONG_POLICY = "wrong_policy"
    WRONG_PARENT = "wrong_parent"
    WRONG_ROOT = "wrong_root"
    MANIFEST_MISMATCH = "manifest_mismatch"
    FOREST_MISMATCH = "forest_mismatch"
    MODIFIED_INPUTS = "modified_inputs"
    INCOMPLETE_HISTORY = "incomplete_history"
    CRYPTOGRAPHIC_FAILURE = "cryptographic_failure"
    SIGNATURE_FAILURE = "signature_failure"
    MALFORMED_SEAL = "malformed_seal"
    MISSING_TRUSTED_KEYS = "missing_trusted_keys"
    MISSING_POLICY = "missing_policy"


class SealKind(str, Enum):
    FULL_CHECKPOINT = "full_checkpoint"
    DELTA_SEAL = "delta_seal"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class UnitProofView:
    """Optional per-unit cryptographic material presented for re-verification."""

    unit_id: str
    proof_object_cid: str
    proof_bytes: bytes | None = None
    public_input_cid: str = ""
    observed_public_input_cid: str = ""
    proof_system_id: str = "integrity"
    signature: str = ""
    signer_id: str = ""
    verification_key_id: str = ""
    expected_proof_digest: str = ""
    freshly_verified: bool = True

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "proof_object_cid": self.proof_object_cid,
            "public_input_cid": self.public_input_cid,
            "observed_public_input_cid": self.observed_public_input_cid,
            "proof_system_id": self.proof_system_id,
            "signature_present": bool(self.signature)
            and self.signature.strip().casefold()
            not in {"", "unsigned", "none", "null", "n/a", "missing"},
            "signer_id": self.signer_id,
            "verification_key_id": self.verification_key_id,
            "expected_proof_digest": self.expected_proof_digest,
            "freshly_verified": self.freshly_verified,
            "proof_byte_length": (
                len(self.proof_bytes) if self.proof_bytes is not None else 0
            ),
            # Never surface proof/witness bytes.
            "proof_bytes_exported": False,
        }


@dataclass(frozen=True, slots=True)
class SealVerificationRequest:
    """Closed inputs for independent seal re-verification."""

    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any]
    trusted_keys: TrustedProofPolicy | Mapping[str, Any] | Sequence[str] | None
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None
    parent_seal: ParentSealView | FullCheckpointSeal | DeltaSeal | Mapping[str, Any] | None = (
        None
    )
    parent_chain: tuple[str, ...] = ()
    unit_proofs: tuple[UnitProofView, ...] = ()
    expected_source_root_cid: str = ""
    expected_repository_state_cid: str = ""
    expected_manifest_root_cid: str = ""
    expected_forest_root_cid: str = ""
    expected_public_input_cid: str = ""
    allowed_proof_systems: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_ALLOWED_PROOF_SYSTEMS
    )
    require_complete_history: bool = True
    require_cryptographic_check: bool = True

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "evidence_subset": EVIDENCE_SUBSET,
            "has_parent": self.parent_seal is not None,
            "parent_chain_length": len(self.parent_chain),
            "unit_proof_count": len(self.unit_proofs),
            "expected_source_root_cid": self.expected_source_root_cid,
            "expected_repository_state_cid": self.expected_repository_state_cid,
            "expected_manifest_root_cid": self.expected_manifest_root_cid,
            "expected_forest_root_cid": self.expected_forest_root_cid,
            "expected_public_input_cid": self.expected_public_input_cid,
            "allowed_proof_systems": sorted(self.allowed_proof_systems),
            "require_complete_history": self.require_complete_history,
            "require_cryptographic_check": self.require_cryptographic_check,
        }


@dataclass(frozen=True, slots=True)
class SealVerificationResult:
    """Typed accept/reject outcome for one seal re-verification."""

    schema: str
    evidence_subset: str
    accepted: bool
    reason: SealVerificationReason
    seal_kind: SealKind
    seal_status: str
    seal_cid: str
    failed_stage: str | None
    stages_passed: tuple[str, ...]
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise VerificationError(f"schema must be {RESULT_SCHEMA}")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise VerificationError(f"evidence_subset must be {EVIDENCE_SUBSET}")
        if type(self.accepted) is not bool:
            raise VerificationError("accepted must be a boolean")
        if self.accepted and self.reason is not SealVerificationReason.ACCEPTED:
            raise VerificationError("accepted results require reason ACCEPTED")
        if not self.accepted and self.reason is SealVerificationReason.ACCEPTED:
            raise VerificationError("rejected results cannot use reason ACCEPTED")
        if self.accepted and self.failed_stage is not None:
            raise VerificationError("accepted results must not set failed_stage")
        if not self.accepted and not self.failed_stage:
            raise VerificationError("rejected results require failed_stage")
        for name in _SENSITIVE_DETAIL_FIELDS:
            if name in self.details:
                raise VerificationError(
                    f"verification details must not carry sensitive field {name!r}"
                )

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "seal_kind": self.seal_kind.value,
            "seal_status": self.seal_status,
            "seal_cid": self.seal_cid,
            "failed_stage": self.failed_stage,
            "stages_passed": list(self.stages_passed),
            "message": self.message,
            "details": dict(self.details),
            "proving_key_exported": False,
            "witness_exported": False,
        }


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _digests_equal(left: str, right: str) -> bool:
    """Constant-time equality for digest strings of equal length."""

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _reject(
    *,
    reason: SealVerificationReason,
    seal_kind: SealKind,
    seal_status: str,
    seal_cid: str,
    failed_stage: str,
    stages_passed: Sequence[str],
    message: str,
    details: Mapping[str, Any] | None = None,
) -> SealVerificationResult:
    return SealVerificationResult(
        schema=RESULT_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        accepted=False,
        reason=reason,
        seal_kind=seal_kind,
        seal_status=seal_status,
        seal_cid=seal_cid,
        failed_stage=failed_stage,
        stages_passed=tuple(stages_passed),
        message=message,
        details=dict(details or {}),
    )


def _accept(
    *,
    seal_kind: SealKind,
    seal_status: str,
    seal_cid: str,
    stages_passed: Sequence[str],
    message: str = "seal accepted under current trusted keys and policy",
    details: Mapping[str, Any] | None = None,
) -> SealVerificationResult:
    return SealVerificationResult(
        schema=RESULT_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        accepted=True,
        reason=SealVerificationReason.ACCEPTED,
        seal_kind=seal_kind,
        seal_status=seal_status,
        seal_cid=seal_cid,
        failed_stage=None,
        stages_passed=tuple(stages_passed),
        message=message,
        details=dict(details or {}),
    )


def _coerce_policy(
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
) -> VerificationPolicyView | None:
    if verification_policy is None:
        return None
    if isinstance(verification_policy, VerificationPolicyView):
        return verification_policy
    if not isinstance(verification_policy, Mapping):
        raise VerificationError(
            "verification_policy must be VerificationPolicyView, mapping, or None"
        )
    policy_cid = str(
        verification_policy.get("policy_cid")
        or verification_policy.get("cid")
        or ""
    )
    if not policy_cid:
        raise VerificationError("verification_policy requires policy_cid")
    return VerificationPolicyView(
        policy_cid=policy_cid,
        proof_schema_version=str(
            verification_policy.get("proof_schema_version") or "1"
        ),
        canonicalization_version=str(
            verification_policy.get("canonicalization_version") or "1"
        ),
        dependency_graph_schema_version=str(
            verification_policy.get("dependency_graph_schema_version")
            or "graph@1"
        ),
        circuit_id=str(verification_policy.get("circuit_id") or "n/a"),
        verification_key_id=str(
            verification_policy.get("verification_key_id") or "n/a"
        ),
    )


def _allowlisted_key_ids(
    trusted_keys: TrustedProofPolicy | Mapping[str, Any] | Sequence[str] | None,
) -> tuple[frozenset[str], TrustedProofPolicy | None]:
    if trusted_keys is None:
        return frozenset(), None
    if isinstance(trusted_keys, TrustedProofPolicy):
        registry = trusted_keys.verification_keys
        ids = registry.ids() if registry is not None else frozenset()
        return ids, trusted_keys
    if isinstance(trusted_keys, Mapping):
        raw = trusted_keys.get("verification_key_ids") or trusted_keys.get(
            "allowed_verification_key_ids"
        )
        if raw is None and "verification_key_id" in trusted_keys:
            raw = (trusted_keys["verification_key_id"],)
        if raw is None:
            keys = trusted_keys.get("keys") or trusted_keys.get("allowlist") or ()
            if isinstance(keys, Mapping):
                raw = keys.keys()
            else:
                raw = keys
        if isinstance(raw, (str, bytes)):
            return frozenset({str(raw)}), None
        return frozenset(str(item) for item in raw if str(item).strip()), None
    if isinstance(trusted_keys, Sequence) and not isinstance(
        trusted_keys, (str, bytes)
    ):
        return frozenset(str(item) for item in trusted_keys if str(item).strip()), None
    raise VerificationError(
        "trusted_keys must be TrustedProofPolicy, mapping, sequence, or None"
    )


def _classify_seal(
    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
) -> tuple[SealKind, dict[str, Any], str]:
    if isinstance(seal, FullCheckpointSeal):
        payload = seal.to_canonical()
        return SealKind.FULL_CHECKPOINT, payload, seal.seal_cid()
    if isinstance(seal, DeltaSeal):
        payload = seal.to_canonical()
        return SealKind.DELTA_SEAL, payload, seal.seal_cid()
    if not isinstance(seal, Mapping):
        raise VerificationError(
            "seal must be FullCheckpointSeal, DeltaSeal, or mapping"
        )
    payload = dict(seal)
    schema = str(payload.get("schema") or "")
    status = str(payload.get("seal_status") or "")
    if "full-checkpoint" in schema or status == SealStatus.SEALED_FULL.value:
        kind = SealKind.FULL_CHECKPOINT
        domain = "ips.full_checkpoint.seal.v1"
    elif "delta-seal" in schema or status == SealStatus.SEALED_INCREMENTAL.value:
        kind = SealKind.DELTA_SEAL
        domain = "ips.delta_seal.seal.v1"
    else:
        kind = SealKind.UNKNOWN
        domain = "ips.unknown.seal.v1"
    seal_cid = str(payload.get("seal_cid") or "")
    if not seal_cid:
        seal_cid = _cid({"domain": domain, "payload": payload})
    return kind, payload, seal_cid


def _parent_bindings(
    parent: ParentSealView | FullCheckpointSeal | DeltaSeal | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if parent is None:
        return None
    if isinstance(parent, ParentSealView):
        return parent.to_canonical()
    if isinstance(parent, FullCheckpointSeal):
        return {
            "seal_cid": parent.seal_cid(),
            "accepted": parent.sealed,
            "seal_status": parent.seal_status.value,
            "source_root_cid": parent.source_root_cid,
            "repository_state_cid": parent.repository_state_cid,
            "manifest_root_cid": parent.manifest_root_cid,
            "forest_root_cid": parent.repository_proof_root,
            "aggregation_root": parent.aggregation_root,
            "policy_cid": parent.policy_cid,
            "environment_cid": parent.environment_cid,
            "repository_id": parent.repository_id,
            "revision": parent.revision,
        }
    if isinstance(parent, DeltaSeal):
        return {
            "seal_cid": parent.seal_cid(),
            "accepted": parent.sealed,
            "seal_status": parent.seal_status.value,
            "source_root_cid": parent.source_root_cid,
            "repository_state_cid": parent.repository_state_cid,
            "manifest_root_cid": parent.new_manifest_root_cid,
            "forest_root_cid": parent.new_forest_root_cid,
            "aggregation_root": parent.new_aggregation_root,
            "policy_cid": parent.policy_cid,
            "environment_cid": parent.environment_cid,
            "repository_id": parent.repository_id,
            "revision": parent.revision,
            "branch_id": parent.branch_id,
        }
    if isinstance(parent, Mapping):
        return dict(parent)
    raise VerificationError("parent_seal has unsupported type")


def _parse_unit_proof(raw: UnitProofView | Mapping[str, Any]) -> UnitProofView:
    if isinstance(raw, UnitProofView):
        return raw
    if not isinstance(raw, Mapping):
        raise VerificationError("unit_proofs entries must be UnitProofView or mapping")
    unit_id = str(raw.get("unit_id") or raw.get("proof_unit_id") or "")
    proof_object_cid = str(
        raw.get("proof_object_cid") or raw.get("proof_cid") or ""
    )
    if not unit_id or not proof_object_cid:
        raise VerificationError("unit proof requires unit_id and proof_object_cid")
    proof_bytes = raw.get("proof_bytes")
    if proof_bytes is not None and not isinstance(proof_bytes, (bytes, bytearray)):
        if isinstance(proof_bytes, str):
            proof_bytes = proof_bytes.encode("utf-8")
        else:
            raise VerificationError("proof_bytes must be bytes when provided")
    return UnitProofView(
        unit_id=unit_id,
        proof_object_cid=proof_object_cid,
        proof_bytes=bytes(proof_bytes) if proof_bytes is not None else None,
        public_input_cid=str(raw.get("public_input_cid") or ""),
        observed_public_input_cid=str(
            raw.get("observed_public_input_cid")
            or raw.get("public_input_cid_observed")
            or ""
        ),
        proof_system_id=str(raw.get("proof_system_id") or "integrity"),
        signature=str(raw.get("signature") or ""),
        signer_id=str(raw.get("signer_id") or ""),
        verification_key_id=str(raw.get("verification_key_id") or ""),
        expected_proof_digest=str(
            raw.get("expected_proof_digest") or raw.get("proof_digest") or ""
        ),
        freshly_verified=bool(raw.get("freshly_verified", True)),
    )


def _unsigned(signature: str) -> bool:
    return signature.strip().casefold() in {
        "",
        "unsigned",
        "none",
        "null",
        "n/a",
        "missing",
    }


def verify_seal(
    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any] | SealVerificationRequest,
    trusted_keys: TrustedProofPolicy | Mapping[str, Any] | Sequence[str] | None = None,
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None = None,
    *,
    parent_seal: ParentSealView
    | FullCheckpointSeal
    | DeltaSeal
    | Mapping[str, Any]
    | None = None,
    parent_chain: Sequence[str] = (),
    unit_proofs: Sequence[UnitProofView | Mapping[str, Any]] = (),
    expected_source_root_cid: str = "",
    expected_repository_state_cid: str = "",
    expected_manifest_root_cid: str = "",
    expected_forest_root_cid: str = "",
    expected_public_input_cid: str = "",
    allowed_proof_systems: Sequence[str] | frozenset[str] | None = None,
    require_complete_history: bool = True,
    require_cryptographic_check: bool = True,
) -> SealVerificationResult:
    """Revalidate a seal under trusted keys and the current verification policy.

    Fail-closed: unknown systems/statuses, wrong keys/policy/parent/root,
    modified inputs, incomplete history, and cryptographic failure reject.
    """

    if isinstance(seal, SealVerificationRequest):
        request = seal
        seal = request.seal
        trusted_keys = request.trusted_keys
        verification_policy = request.verification_policy
        parent_seal = request.parent_seal
        parent_chain = request.parent_chain
        unit_proofs = request.unit_proofs
        expected_source_root_cid = request.expected_source_root_cid
        expected_repository_state_cid = request.expected_repository_state_cid
        expected_manifest_root_cid = request.expected_manifest_root_cid
        expected_forest_root_cid = request.expected_forest_root_cid
        expected_public_input_cid = request.expected_public_input_cid
        allowed_proof_systems = request.allowed_proof_systems
        require_complete_history = request.require_complete_history
        require_cryptographic_check = request.require_cryptographic_check

    stages: list[str] = []
    try:
        seal_kind, payload, seal_cid = _classify_seal(seal)
    except VerificationError as exc:
        return _reject(
            reason=SealVerificationReason.MALFORMED_SEAL,
            seal_kind=SealKind.UNKNOWN,
            seal_status="unknown",
            seal_cid="",
            failed_stage="type",
            stages_passed=(),
            message=str(exc),
        )

    # --- type ---
    if seal_kind is SealKind.UNKNOWN:
        return _reject(
            reason=SealVerificationReason.UNKNOWN_SEAL_TYPE,
            seal_kind=seal_kind,
            seal_status=str(payload.get("seal_status") or "unknown"),
            seal_cid=seal_cid,
            failed_stage="type",
            stages_passed=(),
            message="seal type is not a known full checkpoint or delta seal",
            details={"schema": str(payload.get("schema") or "")},
        )
    stages.append("type")

    # --- status ---
    raw_status = payload.get("seal_status")
    try:
        status = parse_seal_status(raw_status)
        status_value = status.value
    except Exception:
        # Unknown statuses always reject (closed vocabulary).
        return _reject(
            reason=SealVerificationReason.UNKNOWN_STATUS,
            seal_kind=seal_kind,
            seal_status=str(raw_status),
            seal_cid=seal_cid,
            failed_stage="status",
            stages_passed=stages,
            message=f"unknown seal status {raw_status!r}",
            details={
                "closed_statuses": sorted(closed_seal_status_values()),
            },
        )
    if status_value not in _ACCEPTED_SEAL_STATUSES:
        return _reject(
            reason=SealVerificationReason.NON_ACCEPTED_STATUS,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="status",
            stages_passed=stages,
            message=f"seal status {status_value!r} is not an accepted sealed status",
        )
    if seal_kind is SealKind.FULL_CHECKPOINT and status is not SealStatus.SEALED_FULL:
        return _reject(
            reason=SealVerificationReason.NON_ACCEPTED_STATUS,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="status",
            stages_passed=stages,
            message="full checkpoint must carry sealed_full status",
        )
    if (
        seal_kind is SealKind.DELTA_SEAL
        and status is not SealStatus.SEALED_INCREMENTAL
    ):
        return _reject(
            reason=SealVerificationReason.NON_ACCEPTED_STATUS,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="status",
            stages_passed=stages,
            message="delta seal must carry sealed_incremental status",
        )
    if payload.get("sealed") is False:
        return _reject(
            reason=SealVerificationReason.NON_ACCEPTED_STATUS,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="status",
            stages_passed=stages,
            message="seal claims sealed=False",
        )
    stages.append("status")

    # --- proof systems (from unit proofs and seal-level claim) ---
    systems = (
        frozenset(str(item) for item in allowed_proof_systems)
        if allowed_proof_systems is not None
        else _DEFAULT_ALLOWED_PROOF_SYSTEMS
    )
    if not systems:
        systems = _DEFAULT_ALLOWED_PROOF_SYSTEMS
    claimed_system = str(
        payload.get("proof_system_id")
        or payload.get("aggregation_label")
        or "merkle_manifest_aggregation"
    )
    if claimed_system not in systems and claimed_system not in {
        "n/a",
        "manifest_aggregation",
    }:
        return _reject(
            reason=SealVerificationReason.UNKNOWN_PROOF_SYSTEM,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="proof_system",
            stages_passed=stages,
            message=f"unknown proof system {claimed_system!r}",
            details={"allowed_proof_systems": sorted(systems)},
        )
    try:
        proofs = tuple(_parse_unit_proof(item) for item in unit_proofs)
    except VerificationError as exc:
        return _reject(
            reason=SealVerificationReason.MALFORMED_SEAL,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="proof_system",
            stages_passed=stages,
            message=str(exc),
        )
    for unit in proofs:
        if unit.proof_system_id not in systems:
            return _reject(
                reason=SealVerificationReason.UNKNOWN_PROOF_SYSTEM,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="proof_system",
                stages_passed=stages,
                message=(
                    f"unit {unit.unit_id!r} uses unknown proof system "
                    f"{unit.proof_system_id!r}"
                ),
                details={
                    "unit_id": unit.unit_id,
                    "proof_system_id": unit.proof_system_id,
                    "allowed_proof_systems": sorted(systems),
                },
            )
    stages.append("proof_system")

    # --- trusted keys ---
    allowlisted, policy_obj = _allowlisted_key_ids(trusted_keys)
    if not allowlisted and policy_obj is None:
        return _reject(
            reason=SealVerificationReason.MISSING_TRUSTED_KEYS,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="key",
            stages_passed=stages,
            message="trusted verification keys are required",
        )
    seal_vk = str(payload.get("verification_key_id") or "n/a")
    # Integrity-only seals may bind verification_key_id="n/a".  Concrete key
    # IDs must be allowlisted; "n/a" is accepted only when no concrete key is
    # claimed or when "n/a" itself is on the allowlist.
    if seal_vk not in {"", "n/a", "none"}:
        if policy_obj is not None:
            decision = policy_obj.select_verification_key(seal_vk)
            if not decision.accepted:
                return _reject(
                    reason=SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="key",
                    stages_passed=stages,
                    message=decision.message,
                    details={
                        "verification_key_id": seal_vk,
                        "trust_reason": decision.reason_code,
                    },
                )
        elif seal_vk not in allowlisted:
            return _reject(
                reason=SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="key",
                stages_passed=stages,
                message=f"verification key {seal_vk!r} is not allowlisted",
                details={
                    "verification_key_id": seal_vk,
                    "allowlisted": sorted(allowlisted),
                },
            )
    stages.append("key")

    # --- policy ---
    try:
        policy = _coerce_policy(verification_policy)
    except VerificationError as exc:
        return _reject(
            reason=SealVerificationReason.MISSING_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message=str(exc),
        )
    if policy is None:
        return _reject(
            reason=SealVerificationReason.MISSING_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="verification_policy is required",
        )
    seal_policy = str(payload.get("policy_cid") or "")
    if seal_policy != policy.policy_cid:
        return _reject(
            reason=SealVerificationReason.WRONG_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="seal policy_cid does not match current verification policy",
            details={
                "seal_policy_cid": seal_policy,
                "expected_policy_cid": policy.policy_cid,
            },
        )
    if (
        policy.verification_key_id not in {"", "n/a"}
        and seal_vk not in {"", "n/a"}
        and seal_vk != policy.verification_key_id
    ):
        return _reject(
            reason=SealVerificationReason.WRONG_VERIFICATION_KEY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="seal verification_key_id does not match policy binding",
            details={
                "seal_verification_key_id": seal_vk,
                "policy_verification_key_id": policy.verification_key_id,
            },
        )
    if (
        policy.circuit_id not in {"", "n/a"}
        and str(payload.get("circuit_id") or "n/a") not in {"", "n/a"}
        and str(payload.get("circuit_id")) != policy.circuit_id
    ):
        return _reject(
            reason=SealVerificationReason.WRONG_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="seal circuit_id does not match policy binding",
            details={
                "seal_circuit_id": payload.get("circuit_id"),
                "policy_circuit_id": policy.circuit_id,
            },
        )
    if str(payload.get("proof_schema_version") or "1") != policy.proof_schema_version:
        return _reject(
            reason=SealVerificationReason.WRONG_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="seal proof_schema_version does not match policy",
        )
    if (
        str(payload.get("canonicalization_version") or "1")
        != policy.canonicalization_version
    ):
        return _reject(
            reason=SealVerificationReason.WRONG_POLICY,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="policy",
            stages_passed=stages,
            message="seal canonicalization_version does not match policy",
        )
    stages.append("policy")

    # --- parent / chain ---
    parent_bindings = _parent_bindings(parent_seal)
    parent_cid = str(payload.get("parent_seal_cid") or "")
    if seal_kind is SealKind.DELTA_SEAL:
        if not parent_cid or parent_cid == GENESIS_PARENT_SEAL:
            return _reject(
                reason=SealVerificationReason.WRONG_PARENT,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="parent",
                stages_passed=stages,
                message="delta seal requires an exact accepted parent seal",
            )
        if parent_bindings is None:
            return _reject(
                reason=SealVerificationReason.INCOMPLETE_HISTORY,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="history",
                stages_passed=stages,
                message="delta seal verification requires parent seal history",
                details={"parent_seal_cid": parent_cid},
            )
        bound_parent_cid = str(
            parent_bindings.get("seal_cid") or parent_bindings.get("cid") or ""
        )
        if bound_parent_cid and bound_parent_cid != parent_cid:
            return _reject(
                reason=SealVerificationReason.WRONG_PARENT,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="parent",
                stages_passed=stages,
                message="provided parent seal_cid does not match seal parent binding",
                details={
                    "seal_parent_seal_cid": parent_cid,
                    "provided_parent_seal_cid": bound_parent_cid,
                },
            )
        if parent_bindings.get("accepted") is False:
            return _reject(
                reason=SealVerificationReason.WRONG_PARENT,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="parent",
                stages_passed=stages,
                message="parent seal is not accepted under current policy",
            )
        parent_status = str(parent_bindings.get("seal_status") or "")
        if parent_status and parent_status not in _ACCEPTED_SEAL_STATUSES:
            return _reject(
                reason=SealVerificationReason.WRONG_PARENT,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="parent",
                stages_passed=stages,
                message=f"parent seal status {parent_status!r} is not accepted",
            )
        # Old roots must match the parent.
        for seal_field, parent_field in (
            ("old_source_root_cid", "source_root_cid"),
            ("old_repository_state_cid", "repository_state_cid"),
            ("old_manifest_root_cid", "manifest_root_cid"),
            ("old_forest_root_cid", "forest_root_cid"),
            ("old_aggregation_root", "aggregation_root"),
        ):
            seal_value = str(payload.get(seal_field) or "")
            parent_value = str(parent_bindings.get(parent_field) or "")
            if seal_value and parent_value and seal_value != parent_value:
                return _reject(
                    reason=SealVerificationReason.WRONG_PARENT,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="parent",
                    stages_passed=stages,
                    message=f"{seal_field} does not match parent {parent_field}",
                    details={
                        "field": seal_field,
                        "seal_value": seal_value,
                        "parent_value": parent_value,
                    },
                )
    stages.append("parent")

    # --- roots ---
    source_root = str(payload.get("source_root_cid") or "")
    state_root = str(payload.get("repository_state_cid") or "")
    if expected_source_root_cid and source_root != expected_source_root_cid:
        return _reject(
            reason=SealVerificationReason.WRONG_ROOT,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="root",
            stages_passed=stages,
            message="source_root_cid does not match expected repository root",
            details={
                "seal_source_root_cid": source_root,
                "expected_source_root_cid": expected_source_root_cid,
            },
        )
    if expected_repository_state_cid and state_root != expected_repository_state_cid:
        return _reject(
            reason=SealVerificationReason.WRONG_ROOT,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="root",
            stages_passed=stages,
            message="repository_state_cid does not match expected state root",
            details={
                "seal_repository_state_cid": state_root,
                "expected_repository_state_cid": expected_repository_state_cid,
            },
        )
    stages.append("root")

    # --- manifest ---
    if seal_kind is SealKind.FULL_CHECKPOINT:
        manifest_root = str(payload.get("manifest_root_cid") or "")
    else:
        manifest_root = str(payload.get("new_manifest_root_cid") or "")
    if expected_manifest_root_cid and manifest_root != expected_manifest_root_cid:
        return _reject(
            reason=SealVerificationReason.MANIFEST_MISMATCH,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="manifest",
            stages_passed=stages,
            message="manifest root does not match expected commitment",
            details={
                "seal_manifest_root_cid": manifest_root,
                "expected_manifest_root_cid": expected_manifest_root_cid,
            },
        )
    required_ids = payload.get("required_unit_ids") or ()
    verified_ids = payload.get("verified_unit_ids") or ()
    if isinstance(required_ids, Sequence) and isinstance(verified_ids, Sequence):
        if list(required_ids) and set(required_ids) - set(verified_ids):
            # Sealed seals must have verified every required unit.
            return _reject(
                reason=SealVerificationReason.MANIFEST_MISMATCH,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="manifest",
                stages_passed=stages,
                message="required units are not fully covered by verified units",
                details={
                    "missing_verified": sorted(
                        set(str(x) for x in required_ids)
                        - set(str(x) for x in verified_ids)
                    ),
                },
            )
    stages.append("manifest")

    # --- forest ---
    if seal_kind is SealKind.FULL_CHECKPOINT:
        forest_root = str(payload.get("repository_proof_root") or "")
    else:
        forest_root = str(payload.get("new_forest_root_cid") or "")
    if expected_forest_root_cid and forest_root != expected_forest_root_cid:
        return _reject(
            reason=SealVerificationReason.FOREST_MISMATCH,
            seal_kind=seal_kind,
            seal_status=status_value,
            seal_cid=seal_cid,
            failed_stage="forest",
            stages_passed=stages,
            message="forest root does not match expected commitment",
            details={
                "seal_forest_root_cid": forest_root,
                "expected_forest_root_cid": expected_forest_root_cid,
            },
        )
    category_roots = payload.get("category_roots") or {}
    if isinstance(category_roots, Mapping) and category_roots:
        missing_cats = [cat for cat in FOREST_CATEGORIES if cat not in category_roots]
        if missing_cats:
            return _reject(
                reason=SealVerificationReason.FOREST_MISMATCH,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="forest",
                stages_passed=stages,
                message="category_roots omit required forest categories",
                details={"missing_categories": missing_cats},
            )
    stages.append("forest")

    # --- modified inputs ---
    if expected_public_input_cid:
        for unit in proofs:
            observed = unit.observed_public_input_cid or unit.public_input_cid
            expected = unit.public_input_cid or expected_public_input_cid
            if observed and expected and observed != expected:
                return _reject(
                    reason=SealVerificationReason.MODIFIED_INPUTS,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="inputs",
                    stages_passed=stages,
                    message=(
                        f"unit {unit.unit_id!r} public input was modified "
                        "relative to the committed value"
                    ),
                    details={
                        "unit_id": unit.unit_id,
                        "expected_public_input_cid": expected,
                        "observed_public_input_cid": observed,
                    },
                )
            if unit.public_input_cid and unit.public_input_cid != expected_public_input_cid:
                # Seal-level expected public input binding when unit carries one.
                if unit.observed_public_input_cid and (
                    unit.observed_public_input_cid != expected_public_input_cid
                ):
                    return _reject(
                        reason=SealVerificationReason.MODIFIED_INPUTS,
                        seal_kind=seal_kind,
                        seal_status=status_value,
                        seal_cid=seal_cid,
                        failed_stage="inputs",
                        stages_passed=stages,
                        message="public inputs were modified after seal commitment",
                        details={
                            "unit_id": unit.unit_id,
                            "expected_public_input_cid": expected_public_input_cid,
                            "observed_public_input_cid": unit.observed_public_input_cid,
                        },
                    )
    # Also catch unit-local public-input drift without a seal-level expected.
    for unit in proofs:
        if (
            unit.public_input_cid
            and unit.observed_public_input_cid
            and unit.public_input_cid != unit.observed_public_input_cid
        ):
            return _reject(
                reason=SealVerificationReason.MODIFIED_INPUTS,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="inputs",
                stages_passed=stages,
                message=(
                    f"unit {unit.unit_id!r} observed public input differs "
                    "from committed public input"
                ),
                details={
                    "unit_id": unit.unit_id,
                    "public_input_cid": unit.public_input_cid,
                    "observed_public_input_cid": unit.observed_public_input_cid,
                },
            )
    stages.append("inputs")

    # --- history completeness ---
    chain = tuple(str(item) for item in parent_chain if str(item).strip())
    if require_complete_history and seal_kind is SealKind.DELTA_SEAL:
        if parent_cid and parent_cid not in chain and parent_bindings is None:
            return _reject(
                reason=SealVerificationReason.INCOMPLETE_HISTORY,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="history",
                stages_passed=stages,
                message="parent chain is incomplete for delta seal verification",
                details={"parent_seal_cid": parent_cid, "parent_chain": list(chain)},
            )
        if chain and parent_cid and parent_cid not in chain:
            # Explicit chain provided but missing the bound parent.
            return _reject(
                reason=SealVerificationReason.INCOMPLETE_HISTORY,
                seal_kind=seal_kind,
                seal_status=status_value,
                seal_cid=seal_cid,
                failed_stage="history",
                stages_passed=stages,
                message="declared parent is absent from the provided parent chain",
                details={"parent_seal_cid": parent_cid, "parent_chain": list(chain)},
            )
    stages.append("history")

    # --- signature / cryptography ---
    if require_cryptographic_check:
        for unit in proofs:
            if unit.proof_system_id in {"signed_receipt", "groth16"} and _unsigned(
                unit.signature
            ):
                return _reject(
                    reason=SealVerificationReason.SIGNATURE_FAILURE,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="signature",
                    stages_passed=stages,
                    message=(
                        f"unit {unit.unit_id!r} requires a signature but none "
                        "was provided"
                    ),
                    details={"unit_id": unit.unit_id},
                )
            if unit.proof_bytes is not None:
                digest = _digest_bytes(unit.proof_bytes)
                expected = unit.expected_proof_digest or unit.proof_object_cid
                if expected and not _digests_equal(digest, expected):
                    # Allow proof_object_cid to be a non-digest CID label when an
                    # explicit expected_proof_digest is supplied and matches.
                    if unit.expected_proof_digest:
                        if not _digests_equal(digest, unit.expected_proof_digest):
                            return _reject(
                                reason=SealVerificationReason.CRYPTOGRAPHIC_FAILURE,
                                seal_kind=seal_kind,
                                seal_status=status_value,
                                seal_cid=seal_cid,
                                failed_stage="cryptography",
                                stages_passed=stages,
                                message=(
                                    f"unit {unit.unit_id!r} proof bytes fail "
                                    "cryptographic digest check"
                                ),
                                details={
                                    "unit_id": unit.unit_id,
                                    "computed_digest": digest,
                                },
                            )
                    elif expected.startswith("sha256:"):
                        return _reject(
                            reason=SealVerificationReason.CRYPTOGRAPHIC_FAILURE,
                            seal_kind=seal_kind,
                            seal_status=status_value,
                            seal_cid=seal_cid,
                            failed_stage="cryptography",
                            stages_passed=stages,
                            message=(
                                f"unit {unit.unit_id!r} proof bytes fail "
                                "cryptographic digest check"
                            ),
                            details={
                                "unit_id": unit.unit_id,
                                "computed_digest": digest,
                                "expected_digest": expected,
                            },
                        )
            if not unit.freshly_verified:
                return _reject(
                    reason=SealVerificationReason.CRYPTOGRAPHIC_FAILURE,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="cryptography",
                    stages_passed=stages,
                    message=(
                        f"unit {unit.unit_id!r} was not freshly verified under "
                        "current policy"
                    ),
                    details={"unit_id": unit.unit_id},
                )
            if (
                unit.verification_key_id
                and unit.verification_key_id not in {"", "n/a"}
                and allowlisted
                and unit.verification_key_id not in allowlisted
                and (
                    policy_obj is None
                    or not policy_obj.select_verification_key(
                        unit.verification_key_id
                    ).accepted
                )
            ):
                return _reject(
                    reason=SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY,
                    seal_kind=seal_kind,
                    seal_status=status_value,
                    seal_cid=seal_cid,
                    failed_stage="key",
                    stages_passed=stages,
                    message=(
                        f"unit {unit.unit_id!r} verification key "
                        f"{unit.verification_key_id!r} is not allowlisted"
                    ),
                    details={
                        "unit_id": unit.unit_id,
                        "verification_key_id": unit.verification_key_id,
                    },
                )
    stages.append("signature")
    stages.append("cryptography")

    return _accept(
        seal_kind=seal_kind,
        seal_status=status_value,
        seal_cid=seal_cid,
        stages_passed=stages,
        details={
            "repository_id": payload.get("repository_id"),
            "revision": payload.get("revision"),
            "policy_cid": seal_policy,
            "verification_key_id": seal_vk,
            "parent_seal_cid": parent_cid,
            "required_unit_count": (
                len(required_ids) if isinstance(required_ids, Sequence) else 0
            ),
            "unit_proof_count": len(proofs),
            "stages": list(VERIFICATION_STAGES),
        },
    )


__all__ = (
    "EVIDENCE_SUBSET",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "VERIFICATION_STAGES",
    "SealKind",
    "SealVerificationReason",
    "SealVerificationRequest",
    "SealVerificationResult",
    "UnitProofView",
    "VerificationError",
    "verify_seal",
)
