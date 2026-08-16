"""SCA formal-first RPR admission: fail-closed LLM implement gating.

Proof-gated contract repair (RPR) for the SCA program. LLM implementation is
``proposal_only`` and is rejected unless an admitted target packet binds a
current snapshot, counterexample, and reproof command.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from .autonomous_repair.contracts import RepairAuthorityRoots, repair_evidence_cid


SCA_RPR_ADMISSION_INTERFACE: Final[str] = "ScaRprAdmission@1"
ADMITTED_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sca-rpr-admitted-packet@1"
)
REJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sca-rpr-rejection@1"
)
READY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sca-rpr-admission-ready@1"
)
DCR_REPAIR_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-packet@1"
)
DCR_REPAIR_PACKET_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-evidence@1"
)
DCR_REPAIR_PACKET_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-admission@1"
)


class RprAdmissionError(ValueError):
    """Fail-closed RPR admission error."""


@dataclass(frozen=True)
class AdmittedTargetPacket:
    """Admitted implement target (proposal_only LLM path)."""

    schema: str
    task_id: str
    snapshot_id: str
    counterexample_id: str
    reproof_command: str
    finding_id: str = ""
    contract_id: str = ""
    write_paths: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    doctor_disposition: str = ""
    llm_output: str = "proposal_only"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["write_paths"] = list(self.write_paths)
        data["validation_commands"] = list(self.validation_commands)
        return data


@dataclass(frozen=True)
class AdmissionRejection:
    schema: str
    reason_codes: tuple[str, ...]
    detail: str
    model_write_authority: str = "reject"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "model_write_authority": self.model_write_authority,
        }


def _nonempty(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RprAdmissionError(f"{field_name} is required")
    return value.strip()


def admit_implement_task(
    task: Mapping[str, Any],
    *,
    current_snapshot_id: str,
) -> AdmittedTargetPacket | AdmissionRejection:
    """Admit or reject an LLM implement task under RPR policy.

    Required bindings:
    - snapshot_id matches current authoritative snapshot
    - counterexample_id (or counterexample content id)
    - reproof_command
    """

    if not isinstance(task, Mapping):
        return AdmissionRejection(
            schema=REJECTION_SCHEMA,
            reason_codes=("malformed_task",),
            detail="implement task must be a mapping",
        )

    reasons: list[str] = []
    task_id = str(task.get("task_id") or task.get("id") or "").strip()
    snapshot_id = str(
        task.get("snapshot_id")
        or task.get("current_snapshot_id")
        or ""
    ).strip()
    counterexample_id = str(
        task.get("counterexample_id") or task.get("counterexample_ref") or ""
    ).strip()
    if not counterexample_id and isinstance(task.get("counterexample"), Mapping):
        counterexample_id = str(
            task["counterexample"].get("id")
            or task["counterexample"].get("counterexample_id")
            or ""
        ).strip()
    reproof = str(
        task.get("reproof_command")
        or (
            (task.get("reproof_commands") or [""])[0]
            if isinstance(task.get("reproof_commands"), (list, tuple))
            and task.get("reproof_commands")
            else ""
        )
    ).strip()

    if not task_id:
        reasons.append("missing_task_id")
    if not snapshot_id:
        reasons.append("missing_snapshot_id")
    elif str(current_snapshot_id or "").strip() and snapshot_id != str(
        current_snapshot_id
    ).strip():
        reasons.append("snapshot_mismatch")
    if not counterexample_id:
        reasons.append("missing_counterexample")
    if not reproof:
        reasons.append("missing_reproof_command")

    if reasons:
        return AdmissionRejection(
            schema=REJECTION_SCHEMA,
            reason_codes=tuple(reasons),
            detail="unbound implement rejected: " + ",".join(reasons),
        )

    write_paths = tuple(
        str(p)
        for p in (task.get("write_paths") or task.get("predicted_files") or ())
        if str(p).strip()
    )
    validation = tuple(
        str(c)
        for c in (task.get("validation_commands") or task.get("validation") or ())
        if str(c).strip()
    )
    if isinstance(task.get("validation"), str) and task.get("validation").strip():
        validation = (task["validation"].strip(),)

    return AdmittedTargetPacket(
        schema=ADMITTED_PACKET_SCHEMA,
        task_id=task_id,
        snapshot_id=snapshot_id,
        counterexample_id=counterexample_id,
        reproof_command=reproof,
        finding_id=str(task.get("finding_id") or ""),
        contract_id=str(task.get("contract_id") or ""),
        write_paths=write_paths,
        validation_commands=validation,
        doctor_disposition=str(task.get("doctor_disposition") or ""),
        llm_output="proposal_only",
        notes="admitted under RPR; LLM output remains proposal_only",
    )


def assert_llm_implement_allowed(packet: Mapping[str, Any] | AdmittedTargetPacket) -> None:
    """Raise if packet is not an admitted RPR target."""

    if isinstance(packet, AdmittedTargetPacket):
        data = packet.to_dict()
    elif isinstance(packet, Mapping):
        data = packet
    else:
        raise RprAdmissionError("packet must be a mapping or AdmittedTargetPacket")

    if str(data.get("schema") or "") != ADMITTED_PACKET_SCHEMA:
        raise RprAdmissionError("packet schema is not an admitted RPR target")
    for field_name in ("task_id", "snapshot_id", "counterexample_id", "reproof_command"):
        if not str(data.get(field_name) or "").strip():
            raise RprAdmissionError(f"admitted packet missing {field_name}")
    if str(data.get("llm_output") or "proposal_only") != "proposal_only":
        raise RprAdmissionError("llm_output must remain proposal_only")


def write_readiness_receipt(
    path: str | Path,
    *,
    doctor_bridge_ok: bool = True,
    ready: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``rpr_admission_ready.json`` receipt."""

    receipt = {
        "schema": READY_RECEIPT_SCHEMA,
        "interface": SCA_RPR_ADMISSION_INTERFACE,
        "ready": bool(ready),
        "doctor_bridge_ok": bool(doctor_bridge_ok),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "llm_output": "proposal_only",
            "require_counterexample": True,
            "require_reproof_command": True,
            "require_snapshot_binding": True,
            "unbound_implement": "reject",
        },
        "extra": dict(extra or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# DCR-070 deterministic repair-packet admission
# ---------------------------------------------------------------------------
#
# This is deliberately separate from the legacy SCA proposal packet above.
# An ``AdmittedTargetPacket`` remains an LLM proposal-only object and is never
# convertible to this packet or accepted by this resolver-backed boundary.


class RepairPacketEvidenceKind(str, Enum):
    EPOCH = "epoch"
    FOREST = "forest"
    GRAPH = "graph"
    FINDING = "finding"
    DOCTOR = "dcr050_doctor"
    PLANNER = "dcr060_planner"
    REGISTRY = "dcr040_registry"
    DESCRIPTOR = "dcr040_descriptor"
    OWNER = "owner_binding"
    SOURCE = "source_binding"
    PROOF = "dcr033_proof_or_counterexample"
    LOGIC = "dcr035_logic_gate"
    IMPACT = "impact_noninterference"
    VALIDATION = "structured_validation"
    INVERSE = "inverse"
    LEASE = "lease_fence"


_DCR070_REQUIRED_EVIDENCE: Final[frozenset[RepairPacketEvidenceKind]] = frozenset(
    RepairPacketEvidenceKind
)
_DCR070_EVIDENCE_FIELDS: Final[Mapping[RepairPacketEvidenceKind, frozenset[str]]] = {
    RepairPacketEvidenceKind.EPOCH: frozenset({"epoch_cid"}),
    RepairPacketEvidenceKind.FOREST: frozenset({"forest_cid"}),
    RepairPacketEvidenceKind.GRAPH: frozenset({"graph_cid"}),
    RepairPacketEvidenceKind.FINDING: frozenset({"finding_cid"}),
    RepairPacketEvidenceKind.DOCTOR: frozenset(
        {"doctor_receipt_cid", "service_identity", "roots_cid"}
    ),
    RepairPacketEvidenceKind.PLANNER: frozenset(
        {"planner_dag_cid", "candidate_cid", "schedule_cid", "roots_cid"}
    ),
    RepairPacketEvidenceKind.REGISTRY: frozenset({"registry_cid"}),
    RepairPacketEvidenceKind.DESCRIPTOR: frozenset({"descriptor_cid", "registry_cid"}),
    RepairPacketEvidenceKind.OWNER: frozenset(
        {"owner_root", "git_head", "git_tree", "overlay_cid"}
    ),
    RepairPacketEvidenceKind.SOURCE: frozenset({"source_path", "source_span", "old_digest"}),
    RepairPacketEvidenceKind.PROOF: frozenset({"proof_or_counterexample_cid"}),
    RepairPacketEvidenceKind.LOGIC: frozenset({"stage_gate_cid"}),
    RepairPacketEvidenceKind.IMPACT: frozenset({"impact_cid", "noninterference_cid"}),
    RepairPacketEvidenceKind.VALIDATION: frozenset({"validation_cid"}),
    RepairPacketEvidenceKind.INVERSE: frozenset({"inverse_cid"}),
    RepairPacketEvidenceKind.LEASE: frozenset({"lease_cid", "fence_cid"}),
}


class RepairPacketAdmissionDisposition(str, Enum):
    INTEGRATION_PENDING = "integration_pending"
    REJECTED = "rejected"


def _dcr070_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RprAdmissionError(f"{field_name} must be non-empty exact text")
    if "synthetic" in value.lower() or "stub" in value.lower():
        raise RprAdmissionError(f"{field_name} may not use synthetic or stub identity")
    return value


@dataclass(frozen=True)
class RepairPacketEvidence:
    """Closed resolver value with a locally recomputable identity.

    This is an intentionally small carrier for typed local integration
    fixtures.  Raw mappings, booleans and prose cannot stand in for it.
    """

    kind: RepairPacketEvidenceKind
    authority_roots: RepairAuthorityRoots
    body: Mapping[str, Any]
    status: str = "passing"
    model_call_count: int = 0
    llm_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepairPacketEvidenceKind):
            raise RprAdmissionError("repair packet evidence kind must be closed")
        if not isinstance(self.authority_roots, RepairAuthorityRoots):
            raise RprAdmissionError("repair packet evidence needs typed authority roots")
        if (
            not isinstance(self.body, Mapping)
            or set(self.body) != _DCR070_EVIDENCE_FIELDS[self.kind]
        ):
            raise RprAdmissionError("repair packet evidence body must use its exact closed shape")
        normalized_body = {
            field_name: _dcr070_text(value, f"{self.kind.value}.{field_name}")
            for field_name, value in self.body.items()
        }
        object.__setattr__(self, "body", normalized_body)
        if self.kind is RepairPacketEvidenceKind.PROOF:
            allowed_statuses = {"reconstructed", "replayed"}
        else:
            allowed_statuses = {"passing"}
        if self.status not in allowed_statuses:
            raise RprAdmissionError("repair packet evidence status is not admitted")
        for field_name in (
            "model_call_count",
            "llm_call_count",
            "provider_call_count",
            "network_call_count",
        ):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) != 0:
                raise RprAdmissionError(f"{field_name} must be exactly zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_REPAIR_PACKET_EVIDENCE_SCHEMA,
            "kind": self.kind.value,
            "authority_roots": self.authority_roots.to_dict(),
            "body": dict(self.body),
            "status": self.status,
            "model_call_count": 0,
            "llm_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


class RepairPacketResolver:
    """Closed in-memory resolver; it never invokes a service or network."""

    def __init__(self, evidence: Sequence[RepairPacketEvidence]) -> None:
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise RprAdmissionError("repair packet resolver needs a sequence of typed evidence")
        values = tuple(evidence)
        if any(not isinstance(item, RepairPacketEvidence) for item in values):
            raise RprAdmissionError("repair packet resolver refuses raw evidence")
        self._values = {item.content_id: item for item in values}
        if len(self._values) != len(values):
            raise RprAdmissionError("repair packet resolver evidence identities must be unique")

    def resolve(self, cid: str) -> RepairPacketEvidence | None:
        return self._values.get(cid)


@dataclass(frozen=True)
class ProofCarryingRepairPacket:
    """DCR-070 closed, non-executing repair admission request."""

    repair_id: str
    authority_roots: RepairAuthorityRoots
    predecessor_evidence_cid: str
    derivation_cid: str
    evidence_cids: Mapping[RepairPacketEvidenceKind, str]
    source_path: str
    source_span: str
    old_digest: str
    owner_root: str
    git_head: str
    git_tree: str
    overlay_cid: str
    write_paths: tuple[str, ...]
    inverse_cid: str
    lease_cid: str
    fence_cid: str

    def __post_init__(self) -> None:
        for field_name in (
            "repair_id", "predecessor_evidence_cid", "derivation_cid", "source_path",
            "source_span", "old_digest", "owner_root", "git_head", "git_tree",
            "overlay_cid", "inverse_cid", "lease_cid", "fence_cid",
        ):
            object.__setattr__(
                self, field_name, _dcr070_text(getattr(self, field_name), field_name)
            )
        if not isinstance(self.authority_roots, RepairAuthorityRoots):
            raise RprAdmissionError("repair packet requires typed authority roots")
        if not self.write_paths or any(
            not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/")
            for path in self.write_paths
        ):
            raise RprAdmissionError("repair packet write_paths must be non-empty relative paths")
        normalized: dict[RepairPacketEvidenceKind, str] = {}
        if not isinstance(self.evidence_cids, Mapping):
            raise RprAdmissionError("repair packet evidence_cids must be a mapping")
        for kind, cid in self.evidence_cids.items():
            if not isinstance(kind, RepairPacketEvidenceKind):
                raise RprAdmissionError("repair packet evidence kinds must be closed enums")
            normalized[kind] = _dcr070_text(cid, f"evidence_cids.{kind.value}")
        if set(normalized) != _DCR070_REQUIRED_EVIDENCE:
            raise RprAdmissionError(
                "repair packet must bind every required evidence kind exactly once"
            )
        object.__setattr__(self, "evidence_cids", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_REPAIR_PACKET_SCHEMA,
            "repair_id": self.repair_id,
            "authority_roots": self.authority_roots.to_dict(),
            "predecessor_evidence_cid": self.predecessor_evidence_cid,
            "derivation_cid": self.derivation_cid,
            "evidence_cids": {
                key.value: value
                for key, value in sorted(
                    self.evidence_cids.items(), key=lambda item: item[0].value
                )
            },
            "source_path": self.source_path,
            "source_span": self.source_span,
            "old_digest": self.old_digest,
            "owner_root": self.owner_root,
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "overlay_cid": self.overlay_cid,
            "write_paths": list(self.write_paths),
            "inverse_cid": self.inverse_cid,
            "lease_cid": self.lease_cid,
            "fence_cid": self.fence_cid,
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class RepairPacketAdmission:
    disposition: RepairPacketAdmissionDisposition
    reason_codes: tuple[str, ...]
    packet_cid: str = ""
    admission_receipt_cid: str = ""
    envelope_cid: str = ""
    worktree_created: bool = False
    execution_authorized: bool = False
    completion_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_REPAIR_PACKET_ADMISSION_SCHEMA,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "packet_cid": self.packet_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "envelope_cid": self.envelope_cid,
            "worktree_created": False,
            "execution_authorized": False,
            "completion_authorized": False,
        }


def admit_proof_carrying_repair_packet(
    packet: Any,
    *,
    resolver: RepairPacketResolver,
    current_roots: RepairAuthorityRoots,
) -> RepairPacketAdmission:
    """Resolve and verify a DCR-070 packet without creating a worktree.

    Current production DCR-050/060 receipts remain integration-pending, so a
    structurally sound packet is returned as pending rather than admitted for
    mutation.  The result cannot be used by the legacy LLM implementation API.
    """

    reasons: list[str] = []
    if isinstance(packet, AdmittedTargetPacket):
        reasons.append("legacy_admitted_target_packet_cannot_authorize_deterministic_runtime")
    elif not isinstance(packet, ProofCarryingRepairPacket):
        reasons.append("typed_proof_carrying_repair_packet_required")
    if not isinstance(resolver, RepairPacketResolver):
        reasons.append("typed_resolver_required")
    if not isinstance(current_roots, RepairAuthorityRoots):
        reasons.append("typed_current_authority_roots_required")
    if reasons:
        return RepairPacketAdmission(
            RepairPacketAdmissionDisposition.REJECTED, tuple(sorted(reasons))
        )
    assert isinstance(packet, ProofCarryingRepairPacket)
    assert isinstance(current_roots, RepairAuthorityRoots)
    if packet.authority_roots != current_roots:
        reasons.append("stale_or_drifted_authority_roots")
    if packet.git_tree != current_roots.git_tree_id:
        reasons.append("packet_git_tree_does_not_match_current_roots")
    if any(
        not path.startswith(packet.owner_root.rstrip("/") + "/")
        and path != packet.owner_root
        for path in packet.write_paths
    ):
        reasons.append("cross_root_write_set_rejected")
    for kind, cid in packet.evidence_cids.items():
        value = resolver.resolve(cid)
        if not isinstance(value, RepairPacketEvidence):
            reasons.append(f"unresolvable_or_untyped_{kind.value}")
            continue
        if value.content_id != cid or value.kind is not kind:
            reasons.append(f"forged_or_kind_mismatched_{kind.value}")
        if value.authority_roots != current_roots:
            reasons.append(f"stale_root_{kind.value}")
        if value.status not in {"passing", "reconstructed", "replayed"}:
            reasons.append(f"non_passing_{kind.value}")
        if kind is RepairPacketEvidenceKind.OWNER and dict(value.body) != {
            "owner_root": packet.owner_root,
            "git_head": packet.git_head,
            "git_tree": packet.git_tree,
            "overlay_cid": packet.overlay_cid,
        }:
            reasons.append("owner_head_tree_overlay_binding_mismatch")
        if kind is RepairPacketEvidenceKind.SOURCE and dict(value.body) != {
            "source_path": packet.source_path,
            "source_span": packet.source_span,
            "old_digest": packet.old_digest,
        }:
            reasons.append("source_path_span_or_old_digest_mismatch")
        if (
            kind is RepairPacketEvidenceKind.INVERSE
            and packet.inverse_cid != value.body["inverse_cid"]
        ):
            reasons.append("nonempty_inverse_binding_mismatch")
        if kind is RepairPacketEvidenceKind.LEASE and (
            packet.lease_cid != value.body["lease_cid"]
            or packet.fence_cid != value.body["fence_cid"]
            or packet.lease_cid == packet.fence_cid
        ):
            reasons.append("lease_or_fence_binding_mismatch")
    if reasons:
        return RepairPacketAdmission(
            RepairPacketAdmissionDisposition.REJECTED,
            tuple(sorted(set(reasons))),
            packet_cid=packet.content_id,
        )
    # No current typed DCR-050/060 live receipt contract is available in this
    # composition root.  Do not mint a DCR-002 ADMITTED envelope: that stage
    # could be misread as mutation authority by a downstream consumer.
    return RepairPacketAdmission(
        RepairPacketAdmissionDisposition.INTEGRATION_PENDING,
        ("integration_pending_dcr050_dcr060_live_receipts",),
        packet_cid=packet.content_id,
    )


__all__ = [
    "ADMITTED_PACKET_SCHEMA",
    "AdmissionRejection",
    "AdmittedTargetPacket",
    "READY_RECEIPT_SCHEMA",
    "REJECTION_SCHEMA",
    "RprAdmissionError",
    "SCA_RPR_ADMISSION_INTERFACE",
    "DCR_REPAIR_PACKET_ADMISSION_SCHEMA",
    "DCR_REPAIR_PACKET_EVIDENCE_SCHEMA",
    "DCR_REPAIR_PACKET_SCHEMA",
    "ProofCarryingRepairPacket",
    "RepairPacketAdmission",
    "RepairPacketAdmissionDisposition",
    "RepairPacketEvidence",
    "RepairPacketEvidenceKind",
    "RepairPacketResolver",
    "admit_implement_task",
    "admit_proof_carrying_repair_packet",
    "assert_llm_implement_allowed",
    "write_readiness_receipt",
]
