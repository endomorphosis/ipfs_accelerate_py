"""Closed fault matrix and restart evidence for integrated Q1-Q4 security.

Every listed Q rejection is injectable.  Partial checkpoints are
quarantined rather than resumed.  Immutable evidence is retained.
Accepted work is compare-and-swap on the input root so a retry cannot
create a second accepted result.

This module does not mutate production promotion pointers.  Tests run in
fault-scoped leases and keep published evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..control.control_contracts import EventCursor
from ..proof.formal_verification_contracts import content_identity
from ..runtime.event_log import append_jsonl_event, latest_event_cursor
from ..validation.integrated_security import (
    ALL_Q_REJECTIONS,
    INTEGRATED_SECURITY_REQUIREMENT_ID,
    MATERIAL_STAGES,
    Q1_REJECTIONS,
    Q2_REJECTIONS,
    Q3_REJECTIONS,
    Q4_REJECTIONS,
    IntegratedSecurityReceipt,
    SecurityDecision,
    SecurityReason,
    SecurityStage,
    admitted_fixture,
    evaluate_integrated_security,
    hostile_fixture,
)
from .learning_recovery import LearningCheckpointAdapter
from .supervisor_recovery import (
    RecoveryCheckpoint,
    RecoveryDisposition,
    RecoveryFault,
    RepairReceipt,
    SupervisorRecovery,
)


SECURITY_FAULT_MATRIX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/security-fault-matrix@1"
)
SECURITY_FAULT_CASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/security-fault-case@1"
)
SECURITY_RECOVERY_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/security-recovery-evidence@1"
)
ACCEPTED_WORK_LEDGER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/security-accepted-work-ledger@1"
)
STAGE_RECOVERY_FAULTS: Final[Mapping[SecurityStage, RecoveryFault]] = {
    SecurityStage.DATASET_INTAKE: RecoveryFault.CORRUPT_CACHE,
    SecurityStage.PROOF_AUTHORITY: RecoveryFault.PROVIDER_LOSS,
    SecurityStage.TRAINING_STATE: RecoveryFault.PROCESS_CRASH,
    SecurityStage.LEASE: RecoveryFault.STALE_LEASE,
    SecurityStage.CHECKPOINT: RecoveryFault.PARTIAL_CHECKPOINT_WRITE,
    SecurityStage.PROMOTION: RecoveryFault.INTERRUPTED_MERGE,
    SecurityStage.UPLOAD: RecoveryFault.DISK_FULL,
}


class SecurityFaultMatrixError(RuntimeError):
    """Unsafe or malformed fault-matrix operation."""


class AcceptedWorkConflict(SecurityFaultMatrixError):
    """A second accepted result for the same input root was refused."""


class SecurityFaultKind(str, Enum):
    Q_REJECTION = "q_rejection"
    PARTIAL_CHECKPOINT = "partial_checkpoint"
    PROCESS_CRASH = "process_crash"
    RESTART = "restart"
    CONCURRENCY = "concurrency"
    STALE_LEASE = "stale_lease"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SecurityFaultCase:
    """One injected rejection or crash, plus the gate/recovery outcome."""

    case_id: str
    kind: SecurityFaultKind
    stage: SecurityStage
    reason: str
    admitted: bool
    evidence_ids: tuple[str, ...]
    receipt_id: str = ""
    disposition: str = ""
    schema: str = SECURITY_FAULT_CASE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "case_id": self.case_id,
            "disposition": self.disposition,
            "evidence_ids": list(self.evidence_ids),
            "kind": self.kind.value,
            "reason": self.reason,
            "receipt_id": self.receipt_id,
            "schema": self.schema,
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class SecurityRecoveryEvidence:
    """Crash/restart evidence for one material stage."""

    stage: SecurityStage
    fault: RecoveryFault
    disposition: RecoveryDisposition
    preserved_evidence_ids: tuple[str, ...]
    restart_count: int
    duplicate_accepted_work: int
    receipt_id: str
    schema: str = SECURITY_RECOVERY_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "duplicate_accepted_work": self.duplicate_accepted_work,
            "fault": self.fault.value,
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "receipt_id": self.receipt_id,
            "restart_count": self.restart_count,
            "schema": self.schema,
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class SecurityFaultMatrixReceipt:
    """Closed-population evidence that every listed Q failure was injected."""

    cases: tuple[SecurityFaultCase, ...]
    recovery: tuple[SecurityRecoveryEvidence, ...]
    missing_rejections: tuple[str, ...]
    duplicate_accepted_work: int
    requirement_id: str = INTEGRATED_SECURITY_REQUIREMENT_ID
    schema: str = SECURITY_FAULT_MATRIX_SCHEMA

    @property
    def closed(self) -> bool:
        return not self.missing_rejections and self.duplicate_accepted_work == 0

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [item.to_dict() for item in self.cases],
            "closed": self.closed,
            "duplicate_accepted_work": self.duplicate_accepted_work,
            "missing_rejections": list(self.missing_rejections),
            "recovery": [item.to_dict() for item in self.recovery],
            "requirement_id": self.requirement_id,
            "schema": self.schema,
        }


class AcceptedWorkLedger:
    """Compare-and-swap ledger: one accepted result per input root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "accepted-work.json"
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"schema": ACCEPTED_WORK_LEDGER_SCHEMA, "entries": {}}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecurityFaultMatrixError("accepted-work ledger is not valid JSON") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("entries"), Mapping):
            raise SecurityFaultMatrixError("accepted-work ledger is malformed")
        return dict(payload)

    def accept(self, *, input_root: str, result_id: str) -> str:
        """Admit one result for ``input_root``.  A different result fails closed."""

        key = str(input_root or "").strip()
        value = str(result_id or "").strip()
        if not key or not value:
            raise SecurityFaultMatrixError("accepted work requires input_root and result_id")
        with self._lock:
            payload = self._load()
            entries = dict(payload.get("entries") or {})
            existing = str(entries.get(key) or "")
            if existing and existing != value:
                raise AcceptedWorkConflict(
                    f"input root {key} already accepted {existing}; refused {value}"
                )
            if existing == value:
                return existing
            entries[key] = value
            payload["schema"] = ACCEPTED_WORK_LEDGER_SCHEMA
            payload["entries"] = entries
            _atomic_write(self._path, _canonical_bytes(payload))
            return value

    def get(self, input_root: str) -> str:
        with self._lock:
            return str((self._load().get("entries") or {}).get(input_root) or "")

    def count(self) -> int:
        with self._lock:
            return len(self._load().get("entries") or {})


@dataclass
class SecurityFaultMatrix:
    """Inject every listed Q rejection and prove safe restart per stage."""

    root: Path
    recovery: SupervisorRecovery = field(init=False)
    ledger: AcceptedWorkLedger = field(init=False)
    adapter: LearningCheckpointAdapter = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recovery = SupervisorRecovery(self.root / "recovery")
        self.ledger = AcceptedWorkLedger(self.root / "ledger")
        self.adapter = LearningCheckpointAdapter(self.recovery)

    def evaluate_payload(
        self,
        payload: Mapping[str, Any],
        *,
        evidence_ids: Sequence[str] = (),
        test_mode: bool = True,
    ) -> IntegratedSecurityReceipt:
        return evaluate_integrated_security(
            {
                "stage": payload.get("stage", SecurityStage.DATASET_INTAKE.value),
                "payload": payload,
                "actor_role": payload.get("actor_role", "operator"),
                "test_mode": test_mode,
                "evidence_ids": tuple(evidence_ids),
            }
        )

    def inject_rejection(self, reason: str) -> SecurityFaultCase:
        payload = hostile_fixture(reason)
        receipt = self.evaluate_payload(payload)
        if receipt.admitted or reason not in receipt.reasons:
            raise SecurityFaultMatrixError(
                f"{reason} was not rejected; reasons={list(receipt.reasons)}"
            )
        return SecurityFaultCase(
            case_id=f"reject:{reason}",
            kind=SecurityFaultKind.Q_REJECTION,
            stage=receipt.stage,
            reason=reason,
            admitted=False,
            evidence_ids=receipt.evidence_ids,
            receipt_id=receipt.content_id,
            disposition=receipt.decision.value,
        )

    def inject_all_rejections(self) -> tuple[SecurityFaultCase, ...]:
        return tuple(self.inject_rejection(reason) for reason in ALL_Q_REJECTIONS)

    def _stage_paths(
        self,
        stage: SecurityStage,
        *,
        incident_id: str = "",
    ) -> tuple[Path, SupervisorRecovery, LearningCheckpointAdapter]:
        stage_root = self.root / "stages" / stage.value
        if incident_id:
            safe = "".join(
                character if character.isalnum() or character in "-._" else "-"
                for character in incident_id
            )
            stage_root = stage_root / safe
        recovery = SupervisorRecovery(stage_root / "recovery")
        return stage_root / "events.jsonl", recovery, LearningCheckpointAdapter(recovery)

    def _checkpoint_admitted(
        self,
        adapter: LearningCheckpointAdapter,
        event_log: Path,
        *,
        generation: int,
        evidence: Sequence[str],
        fence: int,
    ) -> RecoveryCheckpoint:
        append_jsonl_event(event_log, "stage_sealed", {"generation": generation})
        cursor = latest_event_cursor(event_log)
        binding = dict(admitted_fixture(SecurityStage.CHECKPOINT)["binding"])
        binding["cursor_id"] = f"cursor:{generation}"
        binding["weights_id"] = f"weights:{generation}"
        binding["random_id"] = f"rng:{generation}"
        binding["cursor_step"] = generation
        return adapter.save(
            binding,
            repository_id="repository:security",
            tree_id="tree:security",
            generation=generation,
            cursor=cursor,
            fence=fence,
            accepted_merged_tree_evidence=evidence,
        )

    def recover_stage(
        self,
        stage: SecurityStage | str,
        *,
        incident_id: str,
        evidence_ids: Sequence[str] = ("stage-evidence",),
        fault: RecoveryFault | None = None,
    ) -> SecurityRecoveryEvidence:
        """Inject a stage crash, recover once, and refuse a second accept."""

        selected = (
            stage if isinstance(stage, SecurityStage) else SecurityStage(str(stage))
        )
        selected_fault = fault or STAGE_RECOVERY_FAULTS[selected]
        evidence = tuple(str(item) for item in evidence_ids)
        event_log, recovery, adapter = self._stage_paths(
            selected, incident_id=incident_id
        )
        sealed = self._checkpoint_admitted(
            adapter,
            event_log,
            generation=1,
            evidence=evidence,
            fence=1,
        )
        latest = self._checkpoint_admitted(
            adapter,
            event_log,
            generation=2,
            evidence=evidence,
            fence=2,
        )
        if selected is SecurityStage.CHECKPOINT:
            latest_path = recovery.checkpoints._checkpoint_path(latest)
            latest_path.write_bytes(b'{"partial":')
            gate = self.evaluate_payload(
                hostile_fixture(SecurityReason.PARTIAL_CHECKPOINT.value)
            )
            if gate.admitted:
                raise SecurityFaultMatrixError("partial checkpoint was admitted")
        if selected_fault is RecoveryFault.PARTIAL_EVENT_WRITE:
            with event_log.open("ab") as stream:
                stream.write(b'{"type":"partial"')
        first = recovery.recover(
            incident_id=incident_id,
            fault=selected_fault,
            repository_id="repository:security",
            tree_id="tree:security",
            event_log_path=event_log,
            current_fencing_token=3 if selected is SecurityStage.LEASE else None,
            observed_fencing_token=2 if selected is SecurityStage.LEASE else None,
            verify=lambda restored: restored.generation >= sealed.generation,
        )
        second = recovery.recover(
            incident_id=incident_id,
            fault=selected_fault,
            repository_id="repository:security",
            tree_id="tree:security",
            event_log_path=event_log,
        )
        if first.receipt_id != second.receipt_id:
            raise SecurityFaultMatrixError("restart issued a second repair receipt")
        result_id = first.receipt_id
        input_root = f"{selected.value}:{incident_id}"
        accepted = self.ledger.accept(input_root=input_root, result_id=result_id)
        replayed = self.ledger.accept(input_root=input_root, result_id=result_id)
        if accepted != replayed:
            raise SecurityFaultMatrixError("accepted-work CAS drifted")
        try:
            self.ledger.accept(input_root=input_root, result_id=result_id + ":dup")
            raise SecurityFaultMatrixError("duplicate accepted work was stored")
        except AcceptedWorkConflict:
            pass
        preserved = tuple(first.preserved_evidence_ids or evidence)
        if any(item not in preserved for item in evidence):
            raise SecurityFaultMatrixError("immutable evidence was not preserved")
        restart_count = 1 if first.disposition is RecoveryDisposition.RECOVERED else 0
        if second.disposition is RecoveryDisposition.RECOVERED:
            restart_count = 1
        return SecurityRecoveryEvidence(
            stage=selected,
            fault=selected_fault,
            disposition=first.disposition,
            preserved_evidence_ids=preserved,
            restart_count=restart_count,
            duplicate_accepted_work=0,
            receipt_id=first.receipt_id,
        )

    def recover_all_stages(self) -> tuple[SecurityRecoveryEvidence, ...]:
        return tuple(
            self.recover_stage(
                stage,
                incident_id=f"stage-{stage.value}",
                evidence_ids=(f"{stage.value}-evidence",),
            )
            for stage in MATERIAL_STAGES
        )

    def run(self) -> SecurityFaultMatrixReceipt:
        """Inject every listed Q rejection and recover every material stage."""

        cases = self.inject_all_rejections()
        recovery = self.recover_all_stages()
        observed = {item.reason for item in cases if not item.admitted}
        missing = tuple(reason for reason in ALL_Q_REJECTIONS if reason not in observed)
        duplicates = sum(item.duplicate_accepted_work for item in recovery)
        return SecurityFaultMatrixReceipt(
            cases=cases,
            recovery=recovery,
            missing_rejections=missing,
            duplicate_accepted_work=duplicates,
        )


def listed_q_rejections() -> dict[str, tuple[str, ...]]:
    return {
        "q1_dataset_intake": Q1_REJECTIONS,
        "q2_proof_authority": Q2_REJECTIONS,
        "q3_training_state": Q3_REJECTIONS,
        "q4_promotion_upload": Q4_REJECTIONS,
    }


__all__ = (
    "ACCEPTED_WORK_LEDGER_SCHEMA",
    "SECURITY_FAULT_CASE_SCHEMA",
    "SECURITY_FAULT_MATRIX_SCHEMA",
    "SECURITY_RECOVERY_EVIDENCE_SCHEMA",
    "STAGE_RECOVERY_FAULTS",
    "AcceptedWorkConflict",
    "AcceptedWorkLedger",
    "SecurityFaultCase",
    "SecurityFaultKind",
    "SecurityFaultMatrix",
    "SecurityFaultMatrixError",
    "SecurityFaultMatrixReceipt",
    "SecurityRecoveryEvidence",
    "listed_q_rejections",
)
