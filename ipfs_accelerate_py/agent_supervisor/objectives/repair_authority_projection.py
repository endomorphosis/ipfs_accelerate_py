"""DCR-083 canonical non-executing repair authority/status projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..autonomous_repair.contracts import (
    AuthorityStage,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
)
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from ..todo_daemon.deterministic_repair_composition import (
    DCR080_COMPOSITION_SCHEMA,
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionResult,
)

DCR083_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-083-authority-projection@1"
)
DCR083_ACTIVATION: Final = "integration_pending_live_dcr080"


class RepairAuthorityStatus(str, Enum):  # noqa: UP042 - Python 3.8
    READY = "ready"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"
    REOPENED = "reopened"
    COMPLETED = "completed"


class RepairAuthorityProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class Dcr010EvidenceBinding:
    evidence_cid: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evidence_cid, str)
            or not self.evidence_cid.strip()
            or "synthetic" in self.evidence_cid
        ):
            raise RepairAuthorityProjectionError("DCR-010 evidence identity must be exact")


@dataclass(frozen=True)
class RepairAuthorityProjection:
    task_id: str
    goal_id: str
    authority_roots: RepairAuthorityRoots
    lifecycle_stage: AuthorityStage
    status: RepairAuthorityStatus
    reason_codes: tuple[str, ...]
    baseline_cid: str
    readiness_cid: str
    dcr010_evidence_cid: str
    dcr080_transition_cid: str
    dependency_cids: tuple[str, ...]
    envelope_cids: tuple[str, ...]

    @property
    def projection_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR083_PROJECTION_SCHEMA,
            "authoritative": False,
            "activation_status": DCR083_ACTIVATION,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "authority_roots": self.authority_roots.to_dict(),
            "lifecycle_stage": self.lifecycle_stage.value,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "baseline_cid": self.baseline_cid,
            "readiness_cid": self.readiness_cid,
            "dcr010_evidence_cid": self.dcr010_evidence_cid,
            "dcr080_transition_cid": self.dcr080_transition_cid,
            "dependency_cids": list(self.dependency_cids),
            "envelope_cids": list(self.envelope_cids),
            "execution_authorized": False,
            "completion_authoritative": False,
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }


def project_repair_authority(
    *,
    task_id: str,
    goal_id: str,
    chain: tuple[RepairEvidenceEnvelope, ...],
    current_roots: RepairAuthorityRoots,
    dependencies: tuple[RepairAuthorityProjection, ...],
    dcr010: Dcr010EvidenceBinding,
    dcr080: DeterministicRepairCompositionResult,
) -> RepairAuthorityProjection:
    """Derive one status from typed evidence, never caller claims or booleans."""
    if not isinstance(task_id, str) or not task_id or not isinstance(goal_id, str) or not goal_id:
        raise RepairAuthorityProjectionError("task and goal identifiers are required")
    dcr080_body = (
        dcr080.to_dict() if isinstance(dcr080, DeterministicRepairCompositionResult) else {}
    )
    if not isinstance(current_roots, RepairAuthorityRoots) or not isinstance(
        dcr010, Dcr010EvidenceBinding
    ):
        raise RepairAuthorityProjectionError("typed roots/DCR-010/DCR-080 bindings required")
    if (
        not isinstance(dcr080, DeterministicRepairCompositionResult)
        or dcr080.task_id != task_id
        or dcr080.disposition
        not in {
            DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
            DeterministicRepairCompositionDisposition.REJECTED,
        }
        or dcr080.receipt_cid != content_identity(dcr080_body)
        or dcr080_body.get("schema") != DCR080_COMPOSITION_SCHEMA
        or any(
            dcr080_body.get(name) != 0
            for name in ("model_call_count", "provider_call_count", "network_call_count")
        )
    ):
        raise RepairAuthorityProjectionError(
            "DCR-080 receipt is forged, wrong-task, or not defer/reject"
        )
    if not chain or any(not isinstance(item, RepairEvidenceEnvelope) for item in chain):
        raise RepairAuthorityProjectionError("typed DCR-002 chain is required")
    reasons: list[str] = []
    for index, item in enumerate(chain):
        if item.authority_roots != current_roots:
            reasons.append("stale_authority_roots")
        if index and item.previous_envelope_cid != chain[index - 1].content_id:
            reasons.append("forged_or_non_immediate_envelope_chain")
        if index:
            try:
                item.require_advances(chain[index - 1])
            except Exception:
                reasons.append("forged_or_invalid_envelope_transition")
    terminal = chain[-1]
    for dependency in dependencies:
        if not isinstance(dependency, RepairAuthorityProjection):
            reasons.append("dependency_projection_not_typed")
            continue
        if dependency.authority_roots != current_roots:
            reasons.append("dependency_roots_stale")
        if dependency.status is not RepairAuthorityStatus.COMPLETED:
            reasons.append("dependency_open_or_inconclusive")
    if terminal.authority_stage is not AuthorityStage.PUBLISHED:
        reasons.append("unpublished_repair_handoff")
    if terminal.authority_stage is AuthorityStage.PUBLISHED:
        try:
            terminal.require_typed_authority(require_completion=True)
        except Exception:
            reasons.append("published_envelope_lacks_current_typed_authority")
    # The only supported DCR-080 binding is explicitly pending; it prevents a
    # status projection from fabricating completion even for a complete chain.
    reasons.append("live_dcr080_completion_transition_pending")
    status = (
        RepairAuthorityStatus.REOPENED
        if any("forged" in item or "stale" in item for item in reasons)
        else RepairAuthorityStatus.BLOCKED
        if any("dependency" in item or "unpublished" in item for item in reasons)
        else RepairAuthorityStatus.INCONCLUSIVE
    )
    return RepairAuthorityProjection(
        task_id,
        goal_id,
        current_roots,
        terminal.authority_stage,
        status,
        tuple(sorted(set(reasons))),
        chain[0].content_id,
        terminal.content_id,
        dcr010.evidence_cid,
        dcr080.receipt_cid,
        tuple(sorted(item.projection_cid for item in dependencies)),
        tuple(item.content_id for item in chain),
    )


def canonical_repair_authority_projection_bytes(value: RepairAuthorityProjection) -> bytes:
    if not isinstance(value, RepairAuthorityProjection):
        raise RepairAuthorityProjectionError("projection must be typed")
    return canonical_json_bytes(value.to_dict())


__all__ = [
    "DCR083_ACTIVATION",
    "DCR083_PROJECTION_SCHEMA",
    "Dcr010EvidenceBinding",
    "RepairAuthorityProjection",
    "RepairAuthorityProjectionError",
    "RepairAuthorityStatus",
    "canonical_repair_authority_projection_bytes",
    "project_repair_authority",
]
