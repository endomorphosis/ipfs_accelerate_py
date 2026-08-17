"""Learning checkpoint persist/resume over the existing recovery store.

The adapter writes :class:`LearningCheckpointBinding` snapshots through
``SupervisorRecovery`` / ``RecoveryCheckpointStore``.  It does not create a
second store, does not overwrite without the current fence, and never treats a
restored snapshot as promotion authority.

Crash recovery is incident-idempotent: the first successful recover of an
incident may restart the run once; a later call returns the same receipt
without a second restart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..control.control_contracts import EventCursor
from ..merge.campaign_leases import CampaignLease, CampaignLeaseCoordinator
from ..runtime.learning_checkpoint import (
    CAMPAIGN_DURABILITY_REQUIREMENT_ID,
    IncompatibleResumeError,
    LearningCheckpointBinding,
    LearningCheckpointError,
    PromotionMutationError,
    StaleFenceError,
    assert_compatible_resume,
    binding_from_checkpoint_state,
    checkpoint_state_payload,
    resume_decision,
    semantic_roots_for,
)
from .supervisor_recovery import (
    RecoveryCheckpoint,
    RecoveryDisposition,
    RecoveryFault,
    RepairReceipt,
    SupervisorRecovery,
)


LEARNING_RESUME_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/learning-resume-receipt@1"
)


class LearningRecoveryError(RuntimeError):
    """Unsafe persist, resume, or crash-recovery of a learning checkpoint."""


@dataclass(frozen=True)
class LearningResumeReceipt:
    """Compatible-resume or crash-recovery outcome for one training run."""

    binding: LearningCheckpointBinding
    checkpoint: RecoveryCheckpoint
    compatible: bool
    restart_performed: bool
    restart_count: int
    promotion_authority: bool = False
    repair: RepairReceipt | None = None
    reason_code: str = "resumed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEARNING_RESUME_RECEIPT_SCHEMA,
            "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
            "binding": self.binding.to_dict(),
            "checkpoint_id": self.checkpoint.checkpoint_id,
            "generation": self.checkpoint.generation,
            "fencing_epoch": self.checkpoint.fencing_epoch,
            "compatible": self.compatible,
            "restart_performed": self.restart_performed,
            "restart_count": self.restart_count,
            "promotion_authority": False,
            "reason_code": self.reason_code,
            "repair_receipt_id": "" if self.repair is None else self.repair.receipt_id,
        }


class LearningCheckpointAdapter:
    """Persist and resume learning bindings on the existing recovery store."""

    def __init__(
        self,
        recovery: SupervisorRecovery | Path | str,
        *,
        leases: CampaignLeaseCoordinator | None = None,
    ) -> None:
        if isinstance(recovery, SupervisorRecovery):
            self.recovery = recovery
        else:
            self.recovery = SupervisorRecovery(recovery)
        self.leases = leases
        self._restarted_incidents: dict[str, str] = {}

    def save(
        self,
        binding: LearningCheckpointBinding | Mapping[str, Any],
        *,
        repository_id: str,
        tree_id: str,
        generation: int,
        cursor: EventCursor,
        fence: int,
        lease: CampaignLease | None = None,
        extra: Mapping[str, Any] | None = None,
        accepted_merged_tree_evidence: Sequence[str] = (),
    ) -> RecoveryCheckpoint:
        """Write one snapshot.  A stale fence or promotion payload fails closed."""

        selected = (
            binding
            if isinstance(binding, LearningCheckpointBinding)
            else LearningCheckpointBinding.from_dict(binding)
        )
        extras = dict(extra or {})
        if extras.get("promotion_authority") is True:
            raise PromotionMutationError("learning checkpoint cannot grant promotion authority")
        if lease is not None:
            if self.leases is None:
                raise LearningRecoveryError("checkpoint write requires the lease coordinator")
            current = self.leases.assert_write_fence(lease, fence)
            if current.resource.value != "checkpoint" and current.resource.value != "run":
                raise LearningRecoveryError(
                    "learning checkpoint writes require the checkpoint or run lease"
                )
        elif fence < 1:
            raise StaleFenceError("checkpoint write requires a positive fence")
        current = self.recovery.checkpoints.load_last_valid()
        if current is not None:
            if current.fencing_epoch and fence < current.fencing_epoch:
                raise StaleFenceError(
                    f"checkpoint write used stale fence {fence}; current is {current.fencing_epoch}"
                )
            if (
                current.fencing_epoch
                and fence == current.fencing_epoch
                and generation != current.generation
            ):
                raise StaleFenceError(
                    f"checkpoint write used stale fence {fence}; current is {current.fencing_epoch}"
                )
            if current.generation > generation:
                raise LearningCheckpointError("checkpoint generation moved backwards")
            try:
                previous = binding_from_checkpoint_state(current.state)
            except LearningCheckpointError:
                previous = None
            if previous is not None:
                assert_compatible_resume(previous, selected)
        return self.recovery.checkpoint(
            repository_id=repository_id,
            tree_id=tree_id,
            generation=generation,
            state=checkpoint_state_payload(selected, extra=extras),
            cursor=cursor,
            accepted_merged_tree_evidence=accepted_merged_tree_evidence,
            semantic_roots=semantic_roots_for(selected),
            fencing_epoch=fence,
        )

    def resume(
        self,
        requested: LearningCheckpointBinding | Mapping[str, Any],
        *,
        repository_id: str,
        tree_id: str,
        fence: int | None = None,
        lease: CampaignLease | None = None,
    ) -> LearningResumeReceipt:
        """Load the last valid snapshot and reject an incompatible lineage."""

        if lease is not None:
            if self.leases is None:
                raise LearningRecoveryError("checkpoint resume requires the lease coordinator")
            if fence is None:
                raise StaleFenceError("resume requires the current fence")
            self.leases.assert_write_fence(lease, fence)
        expected = (
            requested
            if isinstance(requested, LearningCheckpointBinding)
            else LearningCheckpointBinding.from_dict(requested)
        )
        checkpoint = self.recovery.checkpoints.load_last_valid()
        if checkpoint is None:
            raise LearningRecoveryError("no_valid_checkpoint")
        if checkpoint.repository_id != repository_id or checkpoint.tree_id != tree_id:
            raise IncompatibleResumeError("checkpoint binding does not match repository/tree")
        if fence is not None and checkpoint.fencing_epoch and fence < checkpoint.fencing_epoch:
            raise StaleFenceError(
                f"resume used stale fence {fence}; current is {checkpoint.fencing_epoch}"
            )
        stored = binding_from_checkpoint_state(checkpoint.state)
        decision = resume_decision(stored, expected)
        return LearningResumeReceipt(
            binding=stored,
            checkpoint=checkpoint,
            compatible=bool(decision["compatible"]),
            restart_performed=False,
            restart_count=0,
            reason_code="compatible_resume",
        )

    def recover_crash(
        self,
        *,
        incident_id: str,
        fault: RecoveryFault | str,
        repository_id: str,
        tree_id: str,
        requested: LearningCheckpointBinding | Mapping[str, Any] | None = None,
        current_fencing_token: int | None = None,
        observed_fencing_token: int | None = None,
        event_log_path: Path | str | None = None,
        repair: Callable[[RecoveryCheckpoint, int], bool | None] | None = None,
        verify: Callable[[RecoveryCheckpoint], bool] | None = None,
    ) -> LearningResumeReceipt:
        """Recover one crash.  A second call of the same incident does not restart."""

        expected = None
        if requested is not None:
            expected = (
                requested
                if isinstance(requested, LearningCheckpointBinding)
                else LearningCheckpointBinding.from_dict(requested)
            )

        def _verify(checkpoint: RecoveryCheckpoint) -> bool:
            stored = binding_from_checkpoint_state(checkpoint.state)
            if expected is not None:
                assert_compatible_resume(stored, expected)
            if verify is not None and verify(checkpoint) is not True:
                return False
            return True

        already = self.recovery.receipt(incident_id)
        receipt = self.recovery.recover(
            incident_id=incident_id,
            fault=fault,
            repository_id=repository_id,
            tree_id=tree_id,
            event_log_path=event_log_path,
            current_fencing_token=current_fencing_token,
            observed_fencing_token=observed_fencing_token,
            repair=repair,
            verify=_verify if expected is not None or verify is not None else None,
        )
        checkpoint = self.recovery.checkpoints.load_last_valid()
        if checkpoint is None:
            raise LearningRecoveryError(receipt.reason_code or "no_valid_checkpoint")
        stored = binding_from_checkpoint_state(checkpoint.state)
        recovered = receipt.disposition in {
            RecoveryDisposition.RECOVERED,
            RecoveryDisposition.NOOP,
        }
        if not recovered and receipt.reason_code == "recovery_verification_failed":
            raise IncompatibleResumeError(receipt.reason_code)
        if already is None and recovered:
            self._restarted_incidents[incident_id] = receipt.receipt_id
            restart_performed = True
        else:
            if already is not None:
                self._restarted_incidents.setdefault(incident_id, already.receipt_id)
            restart_performed = False
        return LearningResumeReceipt(
            binding=stored,
            checkpoint=checkpoint,
            compatible=recovered,
            restart_performed=restart_performed,
            restart_count=1 if incident_id in self._restarted_incidents else 0,
            repair=receipt,
            reason_code=receipt.reason_code,
        )


__all__ = (
    "LEARNING_RESUME_RECEIPT_SCHEMA",
    "LearningCheckpointAdapter",
    "LearningRecoveryError",
    "LearningResumeReceipt",
)
