"""Daemon-facing campaign resume, lease adoption, and one-shot restart.

The coordinator composes the existing worktree/lease/CAS/recovery/refill
surfaces.  It may restart a crashed run exactly once per incident.  It does
not mutate a promotion pointer and will not start refill after a no-progress
or trigger bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..control.control_contracts import EventCursor
from ..merge.campaign_leases import (
    CampaignLease,
    CampaignLeaseCoordinator,
    DuplicateWriterError,
)
from ..rescue.learning_recovery import LearningCheckpointAdapter, LearningResumeReceipt
from ..rescue.supervisor_recovery import RecoveryFault, SupervisorRecovery
from ..runtime.learning_checkpoint import (
    CAMPAIGN_DURABILITY_REQUIREMENT_ID,
    L3ResourceKind,
    LearningCheckpointBinding,
    exclusive_lease_key,
)
from ..self_improvement.campaign_refill_policy import (
    CampaignRefillCandidate,
    CampaignRefillController,
    CampaignRefillDecision,
    CampaignRefillHistory,
    CampaignRefillPolicy,
    RefillDisposition,
)


CAMPAIGN_RESUME_COORDINATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-resume-coordination@1"
)


@dataclass(frozen=True)
class CampaignResumeReport:
    """One daemon resume/recovery decision."""

    run_lease: CampaignLease | None
    checkpoint_lease: CampaignLease | None
    resume: LearningResumeReceipt | None
    refill: CampaignRefillDecision | None
    restart_performed: bool
    promotion_authority: bool = False
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_RESUME_COORDINATION_SCHEMA,
            "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
            "run_lease_id": "" if self.run_lease is None else self.run_lease.lease_id,
            "checkpoint_lease_id": (
                "" if self.checkpoint_lease is None else self.checkpoint_lease.lease_id
            ),
            "run_lease_key": exclusive_lease_key(L3ResourceKind.RUN),
            "checkpoint_lease_key": exclusive_lease_key(L3ResourceKind.CHECKPOINT),
            "resume": None if self.resume is None else self.resume.to_dict(),
            "refill": None if self.refill is None else self.refill.to_dict(),
            "restart_performed": self.restart_performed,
            "promotion_authority": False,
            "reason_code": self.reason_code,
        }


class CampaignResumeCoordinator:
    """Adopt L3 leases, resume a compatible checkpoint, and bound refill."""

    def __init__(
        self,
        root: Path | str,
        *,
        owner_id: str,
        leases: CampaignLeaseCoordinator | None = None,
        adapter: LearningCheckpointAdapter | None = None,
        refill: CampaignRefillController | None = None,
        clock=None,
    ) -> None:
        self.root = Path(root)
        self.owner_id = owner_id
        self.leases = leases or CampaignLeaseCoordinator(self.root / "leases", clock=clock)
        recovery = SupervisorRecovery(self.root / "recovery")
        self.adapter = adapter or LearningCheckpointAdapter(recovery, leases=self.leases)
        self.refill = refill or CampaignRefillController(CampaignRefillPolicy())

    def acquire_resource(
        self,
        kind: L3ResourceKind | str,
        *,
        resource_id: str = "",
    ) -> CampaignLease:
        return self.leases.acquire(kind, owner_id=self.owner_id, resource_id=resource_id)

    def persist(
        self,
        binding: LearningCheckpointBinding | Mapping[str, Any],
        *,
        repository_id: str,
        tree_id: str,
        generation: int,
        cursor: EventCursor,
        extra: Mapping[str, Any] | None = None,
    ) -> LearningResumeReceipt:
        run_lease = self.acquire_resource(L3ResourceKind.RUN)
        checkpoint_lease = self.acquire_resource(L3ResourceKind.CHECKPOINT)
        checkpoint = self.adapter.save(
            binding,
            repository_id=repository_id,
            tree_id=tree_id,
            generation=generation,
            cursor=cursor,
            fence=checkpoint_lease.fence,
            lease=checkpoint_lease,
            extra=extra,
        )
        stored = (
            binding
            if isinstance(binding, LearningCheckpointBinding)
            else LearningCheckpointBinding.from_dict(binding)
        )
        return LearningResumeReceipt(
            binding=stored,
            checkpoint=checkpoint,
            compatible=True,
            restart_performed=False,
            restart_count=0,
            reason_code="checkpoint_persisted",
        )

    def recover_and_resume(
        self,
        *,
        incident_id: str,
        repository_id: str,
        tree_id: str,
        requested: LearningCheckpointBinding | Mapping[str, Any],
        fault: RecoveryFault | str = RecoveryFault.PROCESS_CRASH,
        event_log_path: Path | str | None = None,
        refill_candidates: Sequence[CampaignRefillCandidate | Mapping[str, Any]] = (),
        refill_history: CampaignRefillHistory | None = None,
        cursor_advanced: bool = True,
        current_fencing_token: int | None = None,
        observed_fencing_token: int | None = None,
    ) -> CampaignResumeReport:
        """Adopt run/checkpoint leases, recover once, then bound refill."""

        try:
            run_lease = self.acquire_resource(L3ResourceKind.RUN)
            checkpoint_lease = self.acquire_resource(L3ResourceKind.CHECKPOINT)
        except DuplicateWriterError as exc:
            return CampaignResumeReport(
                run_lease=None,
                checkpoint_lease=None,
                resume=None,
                refill=None,
                restart_performed=False,
                reason_code=str(exc),
            )
        fence = current_fencing_token if current_fencing_token is not None else checkpoint_lease.fence
        observed = (
            observed_fencing_token if observed_fencing_token is not None else max(1, fence - 1)
        )
        resume = self.adapter.recover_crash(
            incident_id=incident_id,
            fault=fault,
            repository_id=repository_id,
            tree_id=tree_id,
            requested=requested,
            current_fencing_token=fence,
            observed_fencing_token=max(1, observed),
            event_log_path=event_log_path,
        )
        refill = None
        if refill_candidates:
            refill = self.refill.decide(
                refill_candidates,
                history=refill_history,
                cursor_advanced=cursor_advanced,
                progress_identity=resume.binding.progress_id,
            )
            if refill.disposition is not RefillDisposition.ADMITTED:
                return CampaignResumeReport(
                    run_lease=run_lease,
                    checkpoint_lease=checkpoint_lease,
                    resume=resume,
                    refill=refill,
                    restart_performed=resume.restart_performed,
                    reason_code=refill.reason_code,
                )
        return CampaignResumeReport(
            run_lease=run_lease,
            checkpoint_lease=checkpoint_lease,
            resume=resume,
            refill=refill,
            restart_performed=resume.restart_performed,
            reason_code=resume.reason_code,
        )


__all__ = (
    "CAMPAIGN_RESUME_COORDINATION_SCHEMA",
    "CampaignResumeCoordinator",
    "CampaignResumeReport",
)
