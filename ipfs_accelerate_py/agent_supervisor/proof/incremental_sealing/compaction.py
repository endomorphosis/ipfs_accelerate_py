"""Delta-chain compaction into a verified full checkpoint (IPS-042).

``compact_seal_chain`` verifies the complete seal chain, the current required
manifest, and every current proof unit; rebuilds a full forest; writes and
verifies a new full checkpoint; and retains historical seal references and all
evidence required by the retention policy.  Broken chains or required evidence
loss reject rather than compact.  History is never rewritten and evidence is
never silently deleted.

Interfaces: ``RetentionPolicy``, ``CompactionOutcome``, ``compact_seal_chain``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    DeltaSeal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    GENESIS_PARENT_SEAL,
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
    SealKind,
    SealVerificationReason,
    UnitProofView,
    verify_seal,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    SealStatus,
    parse_seal_status,
)

EVIDENCE_SUBSET: Final[str] = "ips/chain-compaction@1"
GOAL_EVIDENCE_SUBSET: Final[str] = "ips/compaction@1"
OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "compaction-outcome@1"
)
RETENTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "retention-policy@1"
)
CHAIN_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "seal-chain-entry@1"
)

_ACCEPTED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        SealStatus.SEALED_FULL.value,
        SealStatus.SEALED_INCREMENTAL.value,
    }
)


class CompactionError(ValueError):
    """Fail-closed chain-compaction contract violation."""


class CompactionReason(str, Enum):
    """Stable reason codes for compaction outcomes."""

    COMPACTED = "compacted"
    BROKEN_CHAIN = "broken_chain"
    INCOMPLETE_HISTORY = "incomplete_history"
    CURRENT_SEAL_REJECTED = "current_seal_rejected"
    CURRENT_SEAL_NOT_ACCEPTED = "current_seal_not_accepted"
    MANIFEST_INCOMPLETE = "manifest_incomplete"
    UNIT_VERIFICATION_FAILED = "unit_verification_failed"
    FOREST_VERIFICATION_FAILED = "forest_verification_failed"
    REQUIRED_EVIDENCE_LOST = "required_evidence_lost"
    RETENTION_REFERENCE_MISSING = "retention_reference_missing"
    EMPTY_CHAIN = "empty_chain"
    MALFORMED_INPUT = "malformed_input"
    NEW_CHECKPOINT_FAILED = "new_checkpoint_failed"


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """What historical material must survive compaction.

    Compaction never rewrites history.  Required seal CIDs and evidence CIDs
    must be present in the supplied retention index or the attempt rejects.
    """

    retain_historical_seal_references: bool = True
    retain_unit_proof_references: bool = True
    retain_manifest_and_forest_roots: bool = True
    required_historical_seal_cids: tuple[str, ...] = ()
    required_evidence_cids: tuple[str, ...] = ()
    # When true, every seal CID observed in the verified chain is retained.
    retain_entire_verified_chain: bool = True
    schema: str = RETENTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RETENTION_SCHEMA:
            raise CompactionError(f"retention schema must be {RETENTION_SCHEMA}")
        for field_name in (
            "retain_historical_seal_references",
            "retain_unit_proof_references",
            "retain_manifest_and_forest_roots",
            "retain_entire_verified_chain",
        ):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise CompactionError(f"{field_name} must be a boolean")
        seals = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in self.required_historical_seal_cids
                    if str(item).strip()
                }
            )
        )
        evidence = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in self.required_evidence_cids
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "required_historical_seal_cids", seals)
        object.__setattr__(self, "required_evidence_cids", evidence)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "retain_historical_seal_references": (
                self.retain_historical_seal_references
            ),
            "retain_unit_proof_references": self.retain_unit_proof_references,
            "retain_manifest_and_forest_roots": (
                self.retain_manifest_and_forest_roots
            ),
            "required_historical_seal_cids": list(
                self.required_historical_seal_cids
            ),
            "required_evidence_cids": list(self.required_evidence_cids),
            "retain_entire_verified_chain": self.retain_entire_verified_chain,
        }

    def policy_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.retention_policy.v1",
                "payload": self.to_canonical(),
            }
        )

    @classmethod
    def from_canonical(cls, payload: Mapping[str, Any]) -> RetentionPolicy:
        if not isinstance(payload, Mapping):
            raise CompactionError("RetentionPolicy payload must be a mapping")
        return cls(
            retain_historical_seal_references=bool(
                payload.get("retain_historical_seal_references", True)
            ),
            retain_unit_proof_references=bool(
                payload.get("retain_unit_proof_references", True)
            ),
            retain_manifest_and_forest_roots=bool(
                payload.get("retain_manifest_and_forest_roots", True)
            ),
            required_historical_seal_cids=tuple(
                payload.get("required_historical_seal_cids") or ()
            ),
            required_evidence_cids=tuple(
                payload.get("required_evidence_cids") or ()
            ),
            retain_entire_verified_chain=bool(
                payload.get("retain_entire_verified_chain", True)
            ),
            schema=str(payload.get("schema") or RETENTION_SCHEMA),
        )

    @classmethod
    def default(cls) -> RetentionPolicy:
        return cls()


@dataclass(frozen=True, slots=True)
class SealChainEntry:
    """One historical seal reference presented for chain verification."""

    seal_cid: str
    parent_seal_cid: str
    seal_status: str
    seal_kind: str
    accepted: bool
    repository_id: str = ""
    revision: str = ""
    source_root_cid: str = ""
    repository_state_cid: str = ""
    environment_cid: str = ""
    policy_cid: str = ""
    manifest_root_cid: str = ""
    forest_root_cid: str = ""
    aggregation_root: str = ""
    required_unit_ids: tuple[str, ...] = ()
    unit_proof_cids: Mapping[str, str] = field(default_factory=dict)
    verification_key_id: str = "n/a"
    proof_schema_version: str = "1"
    canonicalization_version: str = "1"
    dependency_graph_schema_version: str = "graph@1"
    circuit_id: str = "n/a"
    schema: str = CHAIN_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        seal_cid = str(self.seal_cid).strip()
        if not seal_cid:
            raise CompactionError("seal_cid must be non-empty")
        object.__setattr__(self, "seal_cid", seal_cid)
        parent = str(self.parent_seal_cid).strip() or GENESIS_PARENT_SEAL
        object.__setattr__(self, "parent_seal_cid", parent)
        if type(self.accepted) is not bool:
            raise CompactionError("accepted must be a boolean")
        object.__setattr__(
            self,
            "required_unit_ids",
            tuple(str(item) for item in self.required_unit_ids),
        )
        object.__setattr__(
            self,
            "unit_proof_cids",
            {
                str(key): str(value)
                for key, value in dict(self.unit_proof_cids).items()
            },
        )
        if self.schema != CHAIN_ENTRY_SCHEMA:
            raise CompactionError(f"chain entry schema must be {CHAIN_ENTRY_SCHEMA}")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seal_cid": self.seal_cid,
            "parent_seal_cid": self.parent_seal_cid,
            "seal_status": self.seal_status,
            "seal_kind": self.seal_kind,
            "accepted": self.accepted,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "source_root_cid": self.source_root_cid,
            "repository_state_cid": self.repository_state_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "manifest_root_cid": self.manifest_root_cid,
            "forest_root_cid": self.forest_root_cid,
            "aggregation_root": self.aggregation_root,
            "required_unit_ids": list(self.required_unit_ids),
            "unit_proof_cids": dict(self.unit_proof_cids),
            "verification_key_id": self.verification_key_id,
            "proof_schema_version": self.proof_schema_version,
            "canonicalization_version": self.canonicalization_version,
            "dependency_graph_schema_version": self.dependency_graph_schema_version,
            "circuit_id": self.circuit_id,
        }


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """Typed accept/reject result of a compaction attempt.

    On success ``sealed`` is true and ``seal`` is a verified full checkpoint.
    On rejection the attempt does not publish a compacted checkpoint; retained
    references still list whatever was verified before the failure when known.
    """

    schema: str
    evidence_subset: str
    goal_evidence_subset: str
    sealed: bool
    reason: CompactionReason
    seal: FullCheckpointSeal | None
    current_seal_cid: str
    compacted_seal_cid: str
    parent_of_compacted: str
    chain_verified: bool
    manifest_verified: bool
    units_verified: bool
    forest_verified: bool
    retention_satisfied: bool
    verified_chain_seal_cids: tuple[str, ...]
    retained_historical_seal_cids: tuple[str, ...]
    retained_evidence_cids: tuple[str, ...]
    missing_required_references: tuple[str, ...]
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != OUTCOME_SCHEMA:
            raise CompactionError(f"schema must be {OUTCOME_SCHEMA}")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise CompactionError(f"evidence_subset must be {EVIDENCE_SUBSET}")
        if self.goal_evidence_subset != GOAL_EVIDENCE_SUBSET:
            raise CompactionError(
                f"goal_evidence_subset must be {GOAL_EVIDENCE_SUBSET}"
            )
        if type(self.sealed) is not bool:
            raise CompactionError("sealed must be a boolean")
        if self.sealed:
            if self.seal is None or not self.seal.sealed:
                raise CompactionError(
                    "sealed compaction requires a sealed full checkpoint"
                )
            if self.reason is not CompactionReason.COMPACTED:
                raise CompactionError(
                    "sealed compaction requires reason compacted"
                )
            if not (
                self.chain_verified
                and self.manifest_verified
                and self.units_verified
                and self.forest_verified
                and self.retention_satisfied
            ):
                raise CompactionError(
                    "sealed compaction requires chain/manifest/units/forest/"
                    "retention verification"
                )
        else:
            if self.reason is CompactionReason.COMPACTED:
                raise CompactionError(
                    "rejected compaction cannot use reason compacted"
                )

    @property
    def rejected(self) -> bool:
        return not self.sealed

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "goal_evidence_subset": self.goal_evidence_subset,
            "sealed": self.sealed,
            "reason": self.reason.value,
            "current_seal_cid": self.current_seal_cid,
            "compacted_seal_cid": self.compacted_seal_cid,
            "parent_of_compacted": self.parent_of_compacted,
            "chain_verified": self.chain_verified,
            "manifest_verified": self.manifest_verified,
            "units_verified": self.units_verified,
            "forest_verified": self.forest_verified,
            "retention_satisfied": self.retention_satisfied,
            "verified_chain_seal_cids": list(self.verified_chain_seal_cids),
            "retained_historical_seal_cids": list(
                self.retained_historical_seal_cids
            ),
            "retained_evidence_cids": list(self.retained_evidence_cids),
            "missing_required_references": list(self.missing_required_references),
            "message": self.message,
            "details": dict(self.details),
            "seal": None if self.seal is None else self.seal.to_canonical(),
            "history_rewritten": False,
            "evidence_silently_deleted": False,
        }

    def outcome_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.compaction_outcome.v1",
                "payload": self.to_canonical(),
            }
        )


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reject(
    *,
    reason: CompactionReason,
    current_seal_cid: str = "",
    compacted_seal_cid: str = "",
    parent_of_compacted: str = "",
    chain_verified: bool = False,
    manifest_verified: bool = False,
    units_verified: bool = False,
    forest_verified: bool = False,
    retention_satisfied: bool = False,
    verified_chain_seal_cids: Sequence[str] = (),
    retained_historical_seal_cids: Sequence[str] = (),
    retained_evidence_cids: Sequence[str] = (),
    missing_required_references: Sequence[str] = (),
    message: str,
    details: Mapping[str, Any] | None = None,
    seal: FullCheckpointSeal | None = None,
) -> CompactionOutcome:
    return CompactionOutcome(
        schema=OUTCOME_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        goal_evidence_subset=GOAL_EVIDENCE_SUBSET,
        sealed=False,
        reason=reason,
        seal=seal,
        current_seal_cid=current_seal_cid,
        compacted_seal_cid=compacted_seal_cid,
        parent_of_compacted=parent_of_compacted,
        chain_verified=chain_verified,
        manifest_verified=manifest_verified,
        units_verified=units_verified,
        forest_verified=forest_verified,
        retention_satisfied=retention_satisfied,
        verified_chain_seal_cids=tuple(verified_chain_seal_cids),
        retained_historical_seal_cids=tuple(retained_historical_seal_cids),
        retained_evidence_cids=tuple(retained_evidence_cids),
        missing_required_references=tuple(missing_required_references),
        message=message,
        details=dict(details or {}),
    )


def _accept(
    *,
    seal: FullCheckpointSeal,
    current_seal_cid: str,
    verified_chain_seal_cids: Sequence[str],
    retained_historical_seal_cids: Sequence[str],
    retained_evidence_cids: Sequence[str],
    message: str = "chain compacted into verified full checkpoint",
    details: Mapping[str, Any] | None = None,
) -> CompactionOutcome:
    return CompactionOutcome(
        schema=OUTCOME_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        goal_evidence_subset=GOAL_EVIDENCE_SUBSET,
        sealed=True,
        reason=CompactionReason.COMPACTED,
        seal=seal,
        current_seal_cid=current_seal_cid,
        compacted_seal_cid=seal.seal_cid(),
        parent_of_compacted=seal.parent_seal_cid,
        chain_verified=True,
        manifest_verified=True,
        units_verified=True,
        forest_verified=True,
        retention_satisfied=True,
        verified_chain_seal_cids=tuple(verified_chain_seal_cids),
        retained_historical_seal_cids=tuple(retained_historical_seal_cids),
        retained_evidence_cids=tuple(retained_evidence_cids),
        missing_required_references=(),
        message=message,
        details=dict(details or {}),
    )


def _coerce_retention(
    retention_policy: RetentionPolicy | Mapping[str, Any] | None,
) -> RetentionPolicy:
    if retention_policy is None:
        return RetentionPolicy.default()
    if isinstance(retention_policy, RetentionPolicy):
        return retention_policy
    if isinstance(retention_policy, Mapping):
        return RetentionPolicy.from_canonical(retention_policy)
    raise CompactionError(
        "retention_policy must be RetentionPolicy, mapping, or None"
    )


def _coerce_verification_policy(
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
) -> VerificationPolicyView:
    if verification_policy is None:
        raise CompactionError("verification_policy is required")
    if isinstance(verification_policy, VerificationPolicyView):
        return verification_policy
    if not isinstance(verification_policy, Mapping):
        raise CompactionError(
            "verification_policy must be VerificationPolicyView or mapping"
        )
    policy_cid = str(
        verification_policy.get("policy_cid")
        or verification_policy.get("cid")
        or ""
    )
    if not policy_cid:
        raise CompactionError("verification_policy requires policy_cid")
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


def _seal_status_value(
    seal: FullCheckpointSeal | DeltaSeal | SealChainEntry | Mapping[str, Any],
) -> str:
    if isinstance(seal, SealChainEntry):
        return seal.seal_status
    if isinstance(seal, (FullCheckpointSeal, DeltaSeal)):
        return seal.seal_status.value
    if isinstance(seal, Mapping):
        return str(seal.get("seal_status") or "")
    raise CompactionError("unsupported seal type")


def _seal_cid_of(
    seal: FullCheckpointSeal | DeltaSeal | SealChainEntry | Mapping[str, Any],
) -> str:
    if isinstance(seal, SealChainEntry):
        return seal.seal_cid
    if isinstance(seal, (FullCheckpointSeal, DeltaSeal)):
        return seal.seal_cid()
    if isinstance(seal, Mapping):
        explicit = str(seal.get("seal_cid") or "").strip()
        if explicit:
            return explicit
        # Content-address mapping payload when no explicit seal_cid is present.
        return _cid(
            {
                "domain": "ips.compaction.mapping_seal.v1",
                "payload": dict(seal),
            }
        )
    raise CompactionError("unsupported seal type")


def _parent_cid_of(
    seal: FullCheckpointSeal | DeltaSeal | SealChainEntry | Mapping[str, Any],
) -> str:
    if isinstance(seal, SealChainEntry):
        return seal.parent_seal_cid
    if isinstance(seal, (FullCheckpointSeal, DeltaSeal)):
        return seal.parent_seal_cid
    if isinstance(seal, Mapping):
        text = str(seal.get("parent_seal_cid") or "").strip()
        return text or GENESIS_PARENT_SEAL
    raise CompactionError("unsupported seal type")


def _seal_kind_of(
    seal: FullCheckpointSeal | DeltaSeal | SealChainEntry | Mapping[str, Any],
) -> str:
    if isinstance(seal, SealChainEntry):
        return seal.seal_kind
    if isinstance(seal, FullCheckpointSeal):
        return SealKind.FULL_CHECKPOINT.value
    if isinstance(seal, DeltaSeal):
        return SealKind.DELTA_SEAL.value
    if isinstance(seal, Mapping):
        schema = str(seal.get("schema") or "")
        status = str(seal.get("seal_status") or "")
        if "full-checkpoint" in schema or status == SealStatus.SEALED_FULL.value:
            return SealKind.FULL_CHECKPOINT.value
        if "delta-seal" in schema or status == SealStatus.SEALED_INCREMENTAL.value:
            return SealKind.DELTA_SEAL.value
        return str(seal.get("seal_kind") or SealKind.UNKNOWN.value)
    raise CompactionError("unsupported seal type")


def _mapping_get(seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any], key: str) -> Any:
    if isinstance(seal, Mapping):
        return seal.get(key)
    return getattr(seal, key, None)


def entry_from_seal(
    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
    *,
    accepted: bool | None = None,
) -> SealChainEntry:
    """Project a seal object into a chain entry for history verification."""

    status = _seal_status_value(seal)
    if accepted is None:
        accepted = status in _ACCEPTED_STATUSES and (
            not isinstance(seal, (FullCheckpointSeal, DeltaSeal)) or bool(seal.sealed)
        )
    required = _mapping_get(seal, "required_unit_ids") or ()
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        required_ids = tuple(str(item) for item in required)
    else:
        required_ids = ()

    unit_proofs: dict[str, str] = {}
    raw_proofs = _mapping_get(seal, "unit_proof_cids")
    if isinstance(raw_proofs, Mapping):
        unit_proofs = {str(k): str(v) for k, v in raw_proofs.items()}

    forest = (
        _mapping_get(seal, "repository_proof_root")
        or _mapping_get(seal, "new_forest_root_cid")
        or _mapping_get(seal, "forest_root_cid")
        or ""
    )
    manifest = (
        _mapping_get(seal, "manifest_root_cid")
        or _mapping_get(seal, "new_manifest_root_cid")
        or ""
    )
    aggregation = (
        _mapping_get(seal, "aggregation_root")
        or _mapping_get(seal, "new_aggregation_root")
        or ""
    )

    return SealChainEntry(
        seal_cid=_seal_cid_of(seal),
        parent_seal_cid=_parent_cid_of(seal),
        seal_status=status,
        seal_kind=_seal_kind_of(seal),
        accepted=bool(accepted),
        repository_id=str(_mapping_get(seal, "repository_id") or ""),
        revision=str(_mapping_get(seal, "revision") or ""),
        source_root_cid=str(_mapping_get(seal, "source_root_cid") or ""),
        repository_state_cid=str(_mapping_get(seal, "repository_state_cid") or ""),
        environment_cid=str(_mapping_get(seal, "environment_cid") or ""),
        policy_cid=str(_mapping_get(seal, "policy_cid") or ""),
        manifest_root_cid=str(manifest),
        forest_root_cid=str(forest),
        aggregation_root=str(aggregation),
        required_unit_ids=required_ids,
        unit_proof_cids=unit_proofs,
        verification_key_id=str(_mapping_get(seal, "verification_key_id") or "n/a"),
        proof_schema_version=str(
            _mapping_get(seal, "proof_schema_version") or "1"
        ),
        canonicalization_version=str(
            _mapping_get(seal, "canonicalization_version") or "1"
        ),
        dependency_graph_schema_version=str(
            _mapping_get(seal, "dependency_graph_schema_version") or "graph@1"
        ),
        circuit_id=str(_mapping_get(seal, "circuit_id") or "n/a"),
    )


def _coerce_chain_entry(
    item: SealChainEntry | FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
) -> SealChainEntry:
    if isinstance(item, SealChainEntry):
        return item
    if isinstance(item, (FullCheckpointSeal, DeltaSeal)):
        return entry_from_seal(item)
    if isinstance(item, Mapping):
        if "seal_cid" in item and "parent_seal_cid" in item and "seal_status" in item:
            return SealChainEntry(
                seal_cid=str(item.get("seal_cid") or ""),
                parent_seal_cid=str(item.get("parent_seal_cid") or GENESIS_PARENT_SEAL),
                seal_status=str(item.get("seal_status") or ""),
                seal_kind=str(item.get("seal_kind") or SealKind.UNKNOWN.value),
                accepted=bool(
                    item.get(
                        "accepted",
                        str(item.get("seal_status") or "") in _ACCEPTED_STATUSES,
                    )
                ),
                repository_id=str(item.get("repository_id") or ""),
                revision=str(item.get("revision") or ""),
                source_root_cid=str(item.get("source_root_cid") or ""),
                repository_state_cid=str(item.get("repository_state_cid") or ""),
                environment_cid=str(item.get("environment_cid") or ""),
                policy_cid=str(item.get("policy_cid") or ""),
                manifest_root_cid=str(item.get("manifest_root_cid") or ""),
                forest_root_cid=str(
                    item.get("forest_root_cid")
                    or item.get("repository_proof_root")
                    or ""
                ),
                aggregation_root=str(item.get("aggregation_root") or ""),
                required_unit_ids=tuple(item.get("required_unit_ids") or ()),
                unit_proof_cids=dict(item.get("unit_proof_cids") or {}),
                verification_key_id=str(item.get("verification_key_id") or "n/a"),
                proof_schema_version=str(item.get("proof_schema_version") or "1"),
                canonicalization_version=str(
                    item.get("canonicalization_version") or "1"
                ),
                dependency_graph_schema_version=str(
                    item.get("dependency_graph_schema_version") or "graph@1"
                ),
                circuit_id=str(item.get("circuit_id") or "n/a"),
                schema=str(item.get("schema") or CHAIN_ENTRY_SCHEMA),
            )
        return entry_from_seal(item)
    raise CompactionError("unsupported seal chain entry type")


def _coerce_units(
    units: Sequence[RequiredUnitEvidence | Mapping[str, Any]],
) -> tuple[RequiredUnitEvidence, ...]:
    parsed: list[RequiredUnitEvidence] = []
    for item in units:
        if isinstance(item, RequiredUnitEvidence):
            parsed.append(item)
            continue
        if not isinstance(item, Mapping):
            raise CompactionError("units must be RequiredUnitEvidence or mappings")
        parsed.append(
            RequiredUnitEvidence(
                unit_id=str(item.get("unit_id") or ""),
                proof_object_cid=str(item.get("proof_object_cid") or ""),
                category=str(item.get("category") or "unit_test"),
                terminal_status=str(
                    item.get("terminal_status")
                    or "integrity_verified"
                ),
                proof_mode=str(item.get("proof_mode") or "integrity_only"),
                required_for_seal=bool(item.get("required_for_seal", True)),
                freshly_verified=bool(item.get("freshly_verified", True)),
                cache_reused_without_fresh_verification=bool(
                    item.get("cache_reused_without_fresh_verification", False)
                ),
                circuit_id=str(item.get("circuit_id") or "n/a"),
                verification_key_id=str(item.get("verification_key_id") or "n/a"),
            )
        )
    return tuple(parsed)


def _repository_state_from_current(
    current: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
    *,
    repository_state: RepositoryStateView | Mapping[str, Any] | None,
) -> RepositoryStateView:
    if isinstance(repository_state, RepositoryStateView):
        return repository_state
    if isinstance(repository_state, Mapping):
        return RepositoryStateView(
            repository_id=str(
                repository_state.get("repository_id")
                or _mapping_get(current, "repository_id")
                or ""
            ),
            revision=str(
                repository_state.get("revision")
                or _mapping_get(current, "revision")
                or ""
            ),
            source_root_cid=str(
                repository_state.get("source_root_cid")
                or _mapping_get(current, "source_root_cid")
                or ""
            ),
            repository_state_cid=str(
                repository_state.get("repository_state_cid")
                or _mapping_get(current, "repository_state_cid")
                or ""
            ),
            environment_cid=str(
                repository_state.get("environment_cid")
                or _mapping_get(current, "environment_cid")
                or ""
            ),
            parent_revision_ids=tuple(
                repository_state.get("parent_revision_ids")
                or _mapping_get(current, "parent_revision_ids")
                or ()
            ),
        )
    return RepositoryStateView(
        repository_id=str(_mapping_get(current, "repository_id") or ""),
        revision=str(_mapping_get(current, "revision") or ""),
        source_root_cid=str(_mapping_get(current, "source_root_cid") or ""),
        repository_state_cid=str(
            _mapping_get(current, "repository_state_cid") or ""
        ),
        environment_cid=str(_mapping_get(current, "environment_cid") or ""),
        parent_revision_ids=tuple(
            _mapping_get(current, "parent_revision_ids") or ()
        ),
    )


def verify_seal_chain(
    chain: Sequence[SealChainEntry | FullCheckpointSeal | DeltaSeal | Mapping[str, Any]],
    *,
    current_seal_cid: str,
) -> tuple[bool, tuple[str, ...], str, Mapping[str, Any]]:
    """Verify parent linkage and acceptance for a complete ordered chain.

    Returns ``(ok, ordered_seal_cids, message, details)``.
    """

    if not chain:
        return False, (), "seal chain is empty", {"reason": "empty_chain"}

    entries = [_coerce_chain_entry(item) for item in chain]
    cids = [entry.seal_cid for entry in entries]
    if len(cids) != len(set(cids)):
        return (
            False,
            tuple(cids),
            "seal chain contains duplicate seal CIDs",
            {"seal_cids": cids},
        )

    # First entry must be genesis-rooted or an accepted full checkpoint root.
    first = entries[0]
    if first.parent_seal_cid not in {GENESIS_PARENT_SEAL, "", "n/a"}:
        # Allow a mid-chain full checkpoint only when it is sealed_full.
        if first.seal_status != SealStatus.SEALED_FULL.value:
            return (
                False,
                tuple(cids),
                "chain root parent is not genesis and first seal is not sealed_full",
                {
                    "first_seal_cid": first.seal_cid,
                    "parent_seal_cid": first.parent_seal_cid,
                },
            )

    for index, entry in enumerate(entries):
        if not entry.accepted or entry.seal_status not in _ACCEPTED_STATUSES:
            return (
                False,
                tuple(cids),
                f"chain seal at index {index} is not accepted",
                {
                    "index": index,
                    "seal_cid": entry.seal_cid,
                    "seal_status": entry.seal_status,
                    "accepted": entry.accepted,
                },
            )
        try:
            parse_seal_status(entry.seal_status)
        except Exception:
            return (
                False,
                tuple(cids),
                f"chain seal at index {index} has unknown status",
                {"seal_cid": entry.seal_cid, "seal_status": entry.seal_status},
            )
        if index == 0:
            continue
        previous = entries[index - 1]
        if entry.parent_seal_cid != previous.seal_cid:
            return (
                False,
                tuple(cids),
                (
                    f"broken chain at index {index}: parent "
                    f"{entry.parent_seal_cid!r} != previous "
                    f"{previous.seal_cid!r}"
                ),
                {
                    "index": index,
                    "seal_cid": entry.seal_cid,
                    "parent_seal_cid": entry.parent_seal_cid,
                    "expected_parent_seal_cid": previous.seal_cid,
                },
            )

    tip = entries[-1]
    if tip.seal_cid != current_seal_cid:
        return (
            False,
            tuple(cids),
            "current seal is not the tip of the provided chain",
            {
                "current_seal_cid": current_seal_cid,
                "tip_seal_cid": tip.seal_cid,
            },
        )

    return True, tuple(cids), "seal chain verified", {"length": len(entries)}


def _collect_retained_evidence(
    *,
    retention: RetentionPolicy,
    chain_entries: Sequence[SealChainEntry],
    units: Sequence[RequiredUnitEvidence],
    available_evidence_cids: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (retained_seals, retained_evidence, missing_required)."""

    retained_seals: set[str] = set()
    retained_evidence: set[str] = set()

    if retention.retain_historical_seal_references:
        if retention.retain_entire_verified_chain:
            retained_seals.update(entry.seal_cid for entry in chain_entries)
        retained_seals.update(retention.required_historical_seal_cids)

    if retention.retain_unit_proof_references:
        for unit in units:
            if unit.proof_object_cid:
                retained_evidence.add(unit.proof_object_cid)
        for entry in chain_entries:
            retained_evidence.update(entry.unit_proof_cids.values())

    if retention.retain_manifest_and_forest_roots:
        for entry in chain_entries:
            for value in (
                entry.manifest_root_cid,
                entry.forest_root_cid,
                entry.aggregation_root,
            ):
                if value:
                    retained_evidence.add(value)

    retained_evidence.update(retention.required_evidence_cids)

    missing: list[str] = []
    for seal_cid in retention.required_historical_seal_cids:
        if seal_cid not in retained_seals and seal_cid not in available_evidence_cids:
            missing.append(seal_cid)
        elif seal_cid not in {entry.seal_cid for entry in chain_entries} and (
            seal_cid not in available_evidence_cids
        ):
            missing.append(seal_cid)
    for evidence_cid in retention.required_evidence_cids:
        if evidence_cid not in available_evidence_cids:
            missing.append(evidence_cid)

    # Required seals must also appear in the verified chain or explicit index.
    chain_cids = {entry.seal_cid for entry in chain_entries}
    for seal_cid in retention.required_historical_seal_cids:
        if seal_cid not in chain_cids and seal_cid not in available_evidence_cids:
            if seal_cid not in missing:
                missing.append(seal_cid)

    return (
        tuple(sorted(retained_seals)),
        tuple(sorted(cid for cid in retained_evidence if cid)),
        tuple(sorted(set(missing))),
    )


def compact_seal_chain(
    current_seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
    retention_policy: RetentionPolicy | Mapping[str, Any] | None,
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    *,
    seal_chain: Sequence[
        SealChainEntry | FullCheckpointSeal | DeltaSeal | Mapping[str, Any]
    ] = (),
    units: Sequence[RequiredUnitEvidence | Mapping[str, Any]] = (),
    expected_unit_ids: Sequence[str] | None = None,
    repository_state: RepositoryStateView | Mapping[str, Any] | None = None,
    available_evidence_cids: Sequence[str] = (),
    trusted_keys: Any = None,
    unit_proofs: Sequence[UnitProofView | Mapping[str, Any]] = (),
    verify_current_cryptographically: bool = True,
) -> CompactionOutcome:
    """Compact a verified seal chain into a new complete full checkpoint.

    Verifies rather than trusts the chain.  Rejects on broken linkage, missing
    required retention references, incomplete current manifest, failed unit
    verification, or forest verification failure.  Never rewrites history or
    silently deletes evidence.
    """

    try:
        retention = _coerce_retention(retention_policy)
        policy = _coerce_verification_policy(verification_policy)
    except CompactionError as exc:
        return _reject(
            reason=CompactionReason.MALFORMED_INPUT,
            message=str(exc),
        )

    try:
        current_cid = _seal_cid_of(current_seal)
        current_status = _seal_status_value(current_seal)
    except CompactionError as exc:
        return _reject(
            reason=CompactionReason.MALFORMED_INPUT,
            message=str(exc),
        )

    if current_status not in _ACCEPTED_STATUSES:
        return _reject(
            reason=CompactionReason.CURRENT_SEAL_NOT_ACCEPTED,
            current_seal_cid=current_cid,
            message=(
                f"current seal status {current_status!r} is not an accepted "
                "sealed status"
            ),
            details={"seal_status": current_status},
        )
    if isinstance(current_seal, (FullCheckpointSeal, DeltaSeal)) and not current_seal.sealed:
        return _reject(
            reason=CompactionReason.CURRENT_SEAL_NOT_ACCEPTED,
            current_seal_cid=current_cid,
            message="current seal is not sealed",
        )

    # Build the working chain: explicit history plus current tip when needed.
    try:
        working: list[
            SealChainEntry | FullCheckpointSeal | DeltaSeal | Mapping[str, Any]
        ] = list(seal_chain)
        if not working:
            working = [current_seal]
        else:
            # Compare tip via coerced entry so SealChainEntry and seal objects
            # share one CID extraction path.
            tip_cid = _coerce_chain_entry(working[-1]).seal_cid
            if tip_cid != current_cid:
                # Append current when the caller supplied ancestors only.
                working = list(working) + [current_seal]

        chain_ok, chain_cids, chain_message, chain_details = verify_seal_chain(
            working, current_seal_cid=current_cid
        )
    except CompactionError as exc:
        return _reject(
            reason=CompactionReason.MALFORMED_INPUT,
            current_seal_cid=current_cid,
            message=str(exc),
        )
    if not chain_ok:
        reason = (
            CompactionReason.EMPTY_CHAIN
            if chain_details.get("reason") == "empty_chain"
            else CompactionReason.BROKEN_CHAIN
        )
        if "incomplete" in chain_message or "not the tip" in chain_message:
            reason = CompactionReason.INCOMPLETE_HISTORY
        return _reject(
            reason=reason,
            current_seal_cid=current_cid,
            message=chain_message,
            details=dict(chain_details),
            verified_chain_seal_cids=chain_cids,
        )

    try:
        chain_entries = [_coerce_chain_entry(item) for item in working]
    except CompactionError as exc:
        return _reject(
            reason=CompactionReason.MALFORMED_INPUT,
            current_seal_cid=current_cid,
            chain_verified=True,
            verified_chain_seal_cids=chain_cids,
            message=str(exc),
        )

    # Optional cryptographic/policy verification of the current tip.
    if verify_current_cryptographically and isinstance(
        current_seal, (FullCheckpointSeal, DeltaSeal)
    ):
        proofs: list[UnitProofView] = []
        for item in unit_proofs:
            if isinstance(item, UnitProofView):
                proofs.append(item)
            elif isinstance(item, Mapping):
                proofs.append(
                    UnitProofView(
                        unit_id=str(item.get("unit_id") or ""),
                        proof_object_cid=str(item.get("proof_object_cid") or ""),
                        proof_bytes=item.get("proof_bytes"),  # type: ignore[arg-type]
                        public_input_cid=str(item.get("public_input_cid") or ""),
                        observed_public_input_cid=str(
                            item.get("observed_public_input_cid") or ""
                        ),
                        proof_system_id=str(
                            item.get("proof_system_id") or "integrity"
                        ),
                        signature=str(item.get("signature") or ""),
                        signer_id=str(item.get("signer_id") or ""),
                        verification_key_id=str(
                            item.get("verification_key_id") or ""
                        ),
                        expected_proof_digest=str(
                            item.get("expected_proof_digest") or ""
                        ),
                        freshly_verified=bool(item.get("freshly_verified", True)),
                    )
                )
        parent_chain = tuple(chain_cids[:-1])
        parent_for_verify: Any = None
        if len(working) >= 2:
            parent_for_verify = working[-2]
            if isinstance(parent_for_verify, SealChainEntry):
                parent_for_verify = parent_for_verify.to_canonical()
        keys = trusted_keys
        if keys is None:
            keys = (policy.verification_key_id, "n/a")
        verification = verify_seal(
            current_seal,
            keys,
            policy,
            parent_seal=parent_for_verify,
            parent_chain=parent_chain,
            unit_proofs=tuple(proofs),
            require_complete_history=isinstance(current_seal, DeltaSeal),
            require_cryptographic_check=bool(proofs),
        )
        if not verification.accepted:
            reason = CompactionReason.CURRENT_SEAL_REJECTED
            if verification.reason is SealVerificationReason.INCOMPLETE_HISTORY:
                reason = CompactionReason.INCOMPLETE_HISTORY
            return _reject(
                reason=reason,
                current_seal_cid=current_cid,
                chain_verified=True,
                verified_chain_seal_cids=chain_cids,
                message=(
                    "current seal failed verification under trusted keys and "
                    f"policy: {verification.message}"
                ),
                details={
                    "verification_reason": verification.reason.value,
                    "failed_stage": verification.failed_stage,
                },
            )

    try:
        parsed_units = _coerce_units(units)
    except CompactionError as exc:
        return _reject(
            reason=CompactionReason.MALFORMED_INPUT,
            current_seal_cid=current_cid,
            chain_verified=True,
            verified_chain_seal_cids=chain_cids,
            message=str(exc),
        )

    if not parsed_units:
        return _reject(
            reason=CompactionReason.MANIFEST_INCOMPLETE,
            current_seal_cid=current_cid,
            chain_verified=True,
            verified_chain_seal_cids=chain_cids,
            message="current required unit set is empty; cannot compact",
        )

    required_ids = tuple(
        sorted({unit.unit_id for unit in parsed_units if unit.required_for_seal})
    )
    if expected_unit_ids is not None:
        expected = tuple(sorted({str(item) for item in expected_unit_ids}))
        if set(required_ids) != set(expected):
            return _reject(
                reason=CompactionReason.MANIFEST_INCOMPLETE,
                current_seal_cid=current_cid,
                chain_verified=True,
                verified_chain_seal_cids=chain_cids,
                message="current required manifest does not match expected unit set",
                details={
                    "required_unit_ids": list(required_ids),
                    "expected_unit_ids": list(expected),
                },
            )

    # Every current required unit must be freshly verified under current policy.
    for unit in parsed_units:
        if not unit.required_for_seal:
            continue
        if unit.cache_reused_without_fresh_verification or not unit.freshly_verified:
            return _reject(
                reason=CompactionReason.UNIT_VERIFICATION_FAILED,
                current_seal_cid=current_cid,
                chain_verified=True,
                manifest_verified=True,
                verified_chain_seal_cids=chain_cids,
                message=(
                    f"required unit {unit.unit_id!r} is not freshly verified "
                    "for compaction"
                ),
                details={"unit_id": unit.unit_id},
            )

    available = frozenset(str(item) for item in available_evidence_cids if str(item))
    # Evidence present on the current units and chain is considered available.
    auto_available = set(available)
    for unit in parsed_units:
        if unit.proof_object_cid:
            auto_available.add(unit.proof_object_cid)
    for entry in chain_entries:
        auto_available.add(entry.seal_cid)
        auto_available.update(entry.unit_proof_cids.values())
        for value in (
            entry.manifest_root_cid,
            entry.forest_root_cid,
            entry.aggregation_root,
        ):
            if value:
                auto_available.add(value)

    retained_seals, retained_evidence, missing = _collect_retained_evidence(
        retention=retention,
        chain_entries=chain_entries,
        units=parsed_units,
        available_evidence_cids=frozenset(auto_available),
    )
    if missing:
        return _reject(
            reason=CompactionReason.REQUIRED_EVIDENCE_LOST
            if any(item in retention.required_evidence_cids for item in missing)
            else CompactionReason.RETENTION_REFERENCE_MISSING,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            missing_required_references=missing,
            message=(
                "required historical references or evidence are missing; "
                "refusing to compact"
            ),
            details={"missing": list(missing)},
        )

    try:
        state = _repository_state_from_current(
            current_seal, repository_state=repository_state
        )
        # New checkpoint is parent-bound to the current tip (historical relation).
        new_seal = create_full_checkpoint(
            state,
            policy,
            units=parsed_units,
            expected_unit_ids=required_ids,
            parent_seal_cid=current_cid,
            fallback_reasons=(
                "chain_compaction",
                "release_qualification",
            ),
        )
    except Exception as exc:  # noqa: BLE001 — fail closed on any build error
        return _reject(
            reason=CompactionReason.NEW_CHECKPOINT_FAILED,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            message=f"new full checkpoint construction failed: {exc}",
            details={"error_type": type(exc).__name__},
        )
    if not new_seal.sealed:
        return _reject(
            reason=CompactionReason.NEW_CHECKPOINT_FAILED,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            message=(
                "new full checkpoint failed to seal after chain verification: "
                f"{new_seal.reason.value}"
            ),
            details={
                "checkpoint_reason": new_seal.reason.value,
                "rejected_unit_ids": list(new_seal.rejected_unit_ids),
            },
            seal=new_seal,
        )

    # Forest completeness: every category root present, repository root set.
    if set(new_seal.category_roots) != set(FOREST_CATEGORIES):
        return _reject(
            reason=CompactionReason.FOREST_VERIFICATION_FAILED,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            message="compacted forest is missing one or more category roots",
            seal=new_seal,
        )
    if not new_seal.repository_proof_root or not new_seal.manifest_root_cid:
        return _reject(
            reason=CompactionReason.FOREST_VERIFICATION_FAILED,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            message="compacted forest or manifest root is empty",
            seal=new_seal,
        )

    # Re-verify the newly built checkpoint under the same policy.
    new_verification = verify_seal(
        new_seal,
        trusted_keys=trusted_keys
        or (
            policy.verification_key_id,
            "n/a",
        ),
        verification_policy=policy,
        parent_chain=chain_cids,
        require_complete_history=False,
        require_cryptographic_check=False,
    )
    if not new_verification.accepted:
        return _reject(
            reason=CompactionReason.FOREST_VERIFICATION_FAILED,
            current_seal_cid=current_cid,
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            verified_chain_seal_cids=chain_cids,
            retained_historical_seal_cids=retained_seals,
            retained_evidence_cids=retained_evidence,
            message=(
                "new compacted checkpoint failed verification: "
                f"{new_verification.message}"
            ),
            details={
                "verification_reason": new_verification.reason.value,
                "failed_stage": new_verification.failed_stage,
            },
            seal=new_seal,
        )

    # Retention always includes the verified chain and the new seal reference.
    final_retained_seals = tuple(
        sorted(set(retained_seals) | set(chain_cids) | {new_seal.seal_cid()})
    )
    final_retained_evidence = tuple(
        sorted(
            set(retained_evidence)
            | {unit.proof_object_cid for unit in parsed_units if unit.proof_object_cid}
            | {
                new_seal.manifest_root_cid,
                new_seal.repository_proof_root,
                new_seal.aggregation_root,
            }
        )
    )

    return _accept(
        seal=new_seal,
        current_seal_cid=current_cid,
        verified_chain_seal_cids=chain_cids,
        retained_historical_seal_cids=final_retained_seals,
        retained_evidence_cids=final_retained_evidence,
        details={
            "retention_policy_cid": retention.policy_cid(),
            "verification_policy_cid": policy.policy_cid,
            "required_unit_ids": list(required_ids),
            "category_roots": {
                cat: new_seal.category_roots[cat] for cat in FOREST_CATEGORIES
            },
            "history_rewritten": False,
            "evidence_silently_deleted": False,
        },
    )


__all__ = (
    "CHAIN_ENTRY_SCHEMA",
    "EVIDENCE_SUBSET",
    "GOAL_EVIDENCE_SUBSET",
    "OUTCOME_SCHEMA",
    "RETENTION_SCHEMA",
    "CompactionError",
    "CompactionOutcome",
    "CompactionReason",
    "RetentionPolicy",
    "SealChainEntry",
    "compact_seal_chain",
    "entry_from_seal",
    "verify_seal_chain",
)
