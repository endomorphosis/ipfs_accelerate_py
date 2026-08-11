"""Final, side-effect-free admission boundary for contract-repair providers.

This module deliberately does not dispatch a provider, read a worktree, or
load target source.  Its only input about the repository is the already-built
``RepositorySnapshot`` ledger.  Consequently a successful result is a narrow
receipt authorizing *one* packet's existing paths; it is never an authority to
select a target or enlarge a packet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    RepairStrategy,
    RepairTargetDecision,
)
from ..analysis.contract_repair_reranker import CandidateEligibilityDisposition
from ..analysis.repository_snapshot import (
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
)
from ..integrations.contract_repair_capabilities import (
    ContractRepairCapabilityReport,
    ContractRepairCapabilityStatus,
)
from ..planning.repair_target_admission import AdmissionResult
from ..proof.contract_repair_edit_packet import ContractRepairEditPacket
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity


CONTRACT_REPAIR_PRE_PROVIDER_GATE_INTERFACE: Final[str] = "ContractRepairPreProviderGate@1"
PRE_PROVIDER_GATE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-pre-provider-gate-receipt@1"
)
MAX_GATE_RECEIPT_BYTES: Final[int] = 65_536
MAX_GATE_PATHS: Final[int] = 256
DEFAULT_REQUIRED_CAPABILITIES: Final[tuple[str, ...]] = ("datasets.hammer",)


class ContractRepairPreProviderGateError(ValueError):
    """The proposed provider hand-off is not current and fully proved."""


class PreProviderGateReason(str, Enum):
    MALFORMED_INPUT = "malformed_input"
    PACKET_DECISION_MISMATCH = "packet_decision_mismatch"
    AMBIGUOUS_OR_ABSTAINED = "ambiguous_or_abstained"
    ROOT_DRIFT = "root_drift"
    TREE_OR_OVERLAY_CHANGED = "tree_or_overlay_changed"
    TARGET_MISSING_OR_MOVED = "target_missing_or_moved"
    TARGET_HASH_DRIFT = "target_hash_drift"
    READ_ONLY_OR_ESCAPED_PATH = "read_only_or_escaped_path"
    EXPIRED_PROOF = "expired_proof"
    PROOF_DOWNGRADED = "proof_downgraded"
    INCOMPLETE_CAPABILITY = "incomplete_capability"
    UNSUPPORTED_CONTRACT_CLAUSE = "unsupported_contract_clause"


def _paths(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairPreProviderGateError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ContractRepairPreProviderGateError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ContractRepairPreProviderGateError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if not result or len(result) > MAX_GATE_PATHS:
        raise ContractRepairPreProviderGateError(f"{name} is empty or exceeds its bound")
    return tuple(sorted(result))


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairPreProviderGateError(f"{name} must be an identifier sequence")
    result = tuple(sorted({value.strip() for value in values if isinstance(value, str) and value.strip()}))
    if not result or len(result) > MAX_GATE_PATHS:
        raise ContractRepairPreProviderGateError(f"{name} is empty or exceeds its bound")
    return result


@dataclass(frozen=True)
class PreProviderGateReceipt:
    """A bounded proof that a packet was safe to expose at one instant."""

    packet_id: str
    decision_id: str
    admission_audit_id: str
    snapshot_id: str
    roots: AuthorityRoots
    target_path: str
    target_artifact_id: str
    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    capability_report_id: str
    required_capability_ids: tuple[str, ...]
    checked_at: int
    expires_at: int

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairPreProviderGateError("receipt roots must be AuthorityRoots")
        for name in ("packet_id", "decision_id", "admission_audit_id", "snapshot_id", "target_artifact_id", "capability_report_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
                raise ContractRepairPreProviderGateError(f"receipt {name} must be a compact identifier")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "target_path", _paths((self.target_path,), "target_path")[0])
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(self, "required_capability_ids", _ids(self.required_capability_ids, "required_capability_ids"))
        if self.target_path not in self.read_paths or self.target_path not in self.write_paths:
            raise ContractRepairPreProviderGateError("receipt target is outside its packet authority")
        for name in ("checked_at", "expires_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractRepairPreProviderGateError(f"receipt {name} must be a non-negative integer")
        if self.expires_at <= self.checked_at:
            raise ContractRepairPreProviderGateError("receipt must expire after it is checked")
        if len(canonical_json_bytes(self.to_dict())) > MAX_GATE_RECEIPT_BYTES:
            raise ContractRepairPreProviderGateError("receipt exceeds its serialized byte bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRE_PROVIDER_GATE_RECEIPT_SCHEMA,
            "interface": CONTRACT_REPAIR_PRE_PROVIDER_GATE_INTERFACE,
            "packet_id": self.packet_id,
            "decision_id": self.decision_id,
            "admission_audit_id": self.admission_audit_id,
            "snapshot_id": self.snapshot_id,
            "roots": self.roots.to_dict(),
            "target_path": self.target_path,
            "target_artifact_id": self.target_artifact_id,
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "capability_report_id": self.capability_report_id,
            "required_capability_ids": list(self.required_capability_ids),
            "checked_at": self.checked_at,
            "expires_at": self.expires_at,
            "provider_invoked": False,
            "authorized_paths": list(self.write_paths),
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def content_id(self) -> str:
        """Compatibility spelling for consumers of canonical contracts."""

        return self.receipt_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreProviderGateReceipt":
        fields = {
            "schema", "interface", "receipt_id", "packet_id", "decision_id", "admission_audit_id", "snapshot_id", "roots",
            "target_path", "target_artifact_id", "read_paths", "write_paths", "capability_report_id", "required_capability_ids",
            "checked_at", "expires_at", "provider_invoked", "authorized_paths",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(fields):
            raise ContractRepairPreProviderGateError("receipt contains unsupported fields")
        if payload.get("schema") != PRE_PROVIDER_GATE_RECEIPT_SCHEMA or payload.get("interface") != CONTRACT_REPAIR_PRE_PROVIDER_GATE_INTERFACE:
            raise ContractRepairPreProviderGateError("receipt has an unsupported schema or interface")
        if payload.get("provider_invoked", False) is not False:
            raise ContractRepairPreProviderGateError("a pre-provider receipt cannot claim provider invocation")
        try:
            receipt = cls(
                packet_id=payload["packet_id"], decision_id=payload["decision_id"], admission_audit_id=payload["admission_audit_id"],
                snapshot_id=payload["snapshot_id"], roots=AuthorityRoots.from_dict(payload["roots"]), target_path=payload["target_path"],
                target_artifact_id=payload["target_artifact_id"], read_paths=tuple(payload["read_paths"]), write_paths=tuple(payload["write_paths"]),
                capability_report_id=payload["capability_report_id"], required_capability_ids=tuple(payload["required_capability_ids"]),
                checked_at=payload["checked_at"], expires_at=payload["expires_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractRepairPreProviderGateError("receipt is malformed") from exc
        if tuple(payload.get("authorized_paths", receipt.write_paths)) != receipt.write_paths:
            raise ContractRepairPreProviderGateError("receipt cannot broaden authorized paths")
        if payload.get("receipt_id") not in (None, "", receipt.receipt_id):
            raise ContractRepairPreProviderGateError("receipt identity is forged")
        return receipt


class ContractRepairPreProviderGate:
    """Replay current packet, proof, capability, and snapshot bindings.

    ``validate`` is intentionally pure and returns closed reason codes.  The
    caller must call ``require_valid`` and obtain its receipt before invoking a
    provider; this class itself has no callback parameter and cannot execute
    untrusted source.
    """

    def validate(
        self,
        packet: ContractRepairEditPacket,
        decision: RepairTargetDecision,
        admission: AdmissionResult,
        snapshot: RepositorySnapshot,
        *,
        current_roots: AuthorityRoots,
        capability_report: ContractRepairCapabilityReport,
        now: int,
        required_capability_ids: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
        read_only_paths: Sequence[str] = (),
    ) -> tuple[PreProviderGateReason, ...]:
        invalid: set[PreProviderGateReason] = set()
        typed = (
            isinstance(packet, ContractRepairEditPacket) and isinstance(decision, RepairTargetDecision)
            and isinstance(admission, AdmissionResult) and isinstance(snapshot, RepositorySnapshot)
            and isinstance(current_roots, AuthorityRoots) and isinstance(capability_report, ContractRepairCapabilityReport)
            and isinstance(now, int) and not isinstance(now, bool)
        )
        if not typed:
            return (PreProviderGateReason.MALFORMED_INPUT,)
        try:
            required = _ids(required_capability_ids, "required_capability_ids")
            blocked_paths = _paths(read_only_paths, "read_only_paths") if read_only_paths else ()
        except ContractRepairPreProviderGateError:
            return (PreProviderGateReason.MALFORMED_INPUT,)
        if current_roots != packet.roots or decision.roots != packet.roots or admission.audit.roots != packet.roots:
            invalid.add(PreProviderGateReason.ROOT_DRIFT)
        if (
            admission.decision != decision or packet.decision_id != decision.content_id or packet.candidate_set_id != decision.candidate_set_id
            or packet.strategy != decision.strategy or packet.read_paths != decision.permitted_read_paths
            or packet.write_paths != decision.permitted_write_paths or packet.proof_refs != decision.proof_refs
            or packet.selection_rationale_refs != decision.evidence_refs or packet.invalidation_refs != decision.invalidation_refs
        ):
            invalid.add(PreProviderGateReason.PACKET_DECISION_MISMATCH)
        selected = next((candidate for candidate in decision.candidates if candidate.content_id == decision.selected_candidate_id), None)
        if (
            decision.disposition is not DecisionDisposition.ADMITTED or decision.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}
            or selected is None or packet.trace_id != getattr(selected, "trace_id", "") or packet.target_span != getattr(selected, "target_span", None)
            or packet.unsupported_clause_ids
        ):
            invalid.add(PreProviderGateReason.AMBIGUOUS_OR_ABSTAINED)
        if packet.unsupported_clause_ids:
            invalid.add(PreProviderGateReason.UNSUPPORTED_CONTRACT_CLAUSE)
        if not admission.expiry.valid_at(now):
            invalid.add(PreProviderGateReason.EXPIRED_PROOF)
        selected_rank = next((rank for rank in admission.audit.ranks if selected is not None and rank.candidate_id == selected.content_id), None)
        proof_artifacts = {ref.artifact_id for ref in decision.proof_refs if ref.kind == "proof_receipt"}
        if (
            admission.audit.decision_id != decision.content_id or admission.audit.candidate_set_id != decision.candidate_set_id
            or selected_rank is None or selected_rank.disposition is not CandidateEligibilityDisposition.ELIGIBLE
            or not decision.proof_refs or not set(getattr(selected_rank, "proof_receipt_ids", ())).issubset(proof_artifacts)
        ):
            invalid.add(PreProviderGateReason.PROOF_DOWNGRADED)
        if snapshot.head_tree_id != current_roots.tree_id or snapshot.index_tree_id != snapshot.head_tree_id or not snapshot.is_clean or snapshot.stats.overlay_path_count:
            invalid.add(PreProviderGateReason.TREE_OR_OVERLAY_CHANGED)
        try:
            snapshot.assert_exhaustive_tracked_coverage()
            target = snapshot.disposition_for_path(packet.target_span.path)
        except Exception:  # ledger corruption is never a reason to inspect source
            target = None
        if (
            target is None or not target.tracked or target.overlay or target.git_status is not GitStatus.CLEAN
            or target.entry_kind is not EntryKind.REGULAR or target.kind in {CoverageKind.EXCLUDED, CoverageKind.UNSUPPORTED, CoverageKind.BINARY_OR_GENERATED}
            or packet.target_span.path in blocked_paths
        ):
            invalid.add(PreProviderGateReason.TARGET_MISSING_OR_MOVED)
        elif packet.target_span.artifact_id not in {target.content_digest, target.git_object_id}:
            invalid.add(PreProviderGateReason.TARGET_HASH_DRIFT)
        if (
            packet.target_span.path not in packet.read_paths or packet.target_span.path not in packet.write_paths
            or packet.read_paths != (packet.target_span.path,) or packet.write_paths != (packet.target_span.path,)
            or packet.target_span.path in blocked_paths
        ):
            invalid.add(PreProviderGateReason.READ_ONLY_OR_ESCAPED_PATH)
        capability_map = capability_report.capability_map
        for capability_id in required:
            capability = capability_map.get(capability_id)
            if (
                capability is None or capability.status is not ContractRepairCapabilityStatus.AVAILABLE
                or not capability.reconstruction_compatible
            ):
                invalid.add(PreProviderGateReason.INCOMPLETE_CAPABILITY)
        return tuple(sorted(invalid, key=lambda item: item.value))

    def require_valid(
        self,
        packet: ContractRepairEditPacket,
        decision: RepairTargetDecision,
        admission: AdmissionResult,
        snapshot: RepositorySnapshot,
        *,
        current_roots: AuthorityRoots,
        capability_report: ContractRepairCapabilityReport,
        now: int,
        required_capability_ids: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
        read_only_paths: Sequence[str] = (),
    ) -> PreProviderGateReceipt:
        invalid = self.validate(
            packet, decision, admission, snapshot, current_roots=current_roots,
            capability_report=capability_report, now=now,
            required_capability_ids=required_capability_ids, read_only_paths=read_only_paths,
        )
        if invalid:
            raise ContractRepairPreProviderGateError(
                "contract repair pre-provider gate rejected: " + ", ".join(item.value for item in invalid)
            )
        required = _ids(required_capability_ids, "required_capability_ids")
        return PreProviderGateReceipt(
            packet_id=packet.packet_id, decision_id=decision.content_id, admission_audit_id=admission.audit.content_id,
            snapshot_id=snapshot.snapshot_id, roots=packet.roots, target_path=packet.target_span.path,
            target_artifact_id=packet.target_span.artifact_id, read_paths=packet.read_paths, write_paths=packet.write_paths,
            capability_report_id=content_identity(capability_report_to_dict(capability_report)),
            required_capability_ids=required, checked_at=now, expires_at=admission.expiry.expires_at,
        )

    check = require_valid
    admit = require_valid

    def is_valid(self, *args: Any, **kwargs: Any) -> bool:
        return not self.validate(*args, **kwargs)


def capability_report_to_dict(report: ContractRepairCapabilityReport) -> dict[str, Any]:
    """Return the body-free, float-free portion relevant to admission.

    Probe timing is observability data, not capability identity.  Excluding it
    also keeps this receipt compatible with the canonical proof encoder, which
    intentionally rejects floating point values.
    """

    return {
        "schema_version": report.schema_version,
        "report_version": report.report_version,
        "accelerator_module_paths": list(report.accelerator_module_paths),
        "datasets_module_paths": list(report.datasets_module_paths),
        "datasets_gitlink_revision": report.datasets_gitlink_revision,
        "capabilities": [
            {
                "capability_id": item.capability_id,
                "status": item.status.value,
                "module_paths": list(item.module_paths),
                "interface_version": item.interface_version,
                "schema_version": item.schema_version,
                "supported_semantics": list(item.supported_semantics),
                "reconstruction_compatible": item.reconstruction_compatible,
            }
            for item in sorted(report.capabilities, key=lambda item: item.capability_id)
        ],
    }


__all__ = [
    "CONTRACT_REPAIR_PRE_PROVIDER_GATE_INTERFACE", "DEFAULT_REQUIRED_CAPABILITIES",
    "MAX_GATE_RECEIPT_BYTES", "ContractRepairPreProviderGate", "ContractRepairPreProviderGateError",
    "PRE_PROVIDER_GATE_RECEIPT_SCHEMA", "PreProviderGateReason", "PreProviderGateReceipt",
]
