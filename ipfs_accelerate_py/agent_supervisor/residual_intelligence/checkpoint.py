"""Narrow residual adapter over canonical LearningCheckpointBinding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.runtime.learning_checkpoint import (
    LearningCheckpointBinding,
)

from .contracts import ResidualIntelligenceError, TrainingAvailability, required_text
from .rights import TrainingCorpusAdmission

ADAPTER_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-checkpoint-adapter@1"
REASON_INCOMPATIBLE_RESUME: Final = "incompatible_resume"
REASON_CORRUPT: Final = "checkpoint_corrupt"
REASON_WITHDRAWN: Final = "corpus_withdrawn"
REASON_NO_PROMOTION: Final = "resume_without_promotion"


@dataclass(frozen=True)
class ExpertCheckpointLineage:
    binding: LearningCheckpointBinding
    admission: TrainingCorpusAdmission
    corrupt: bool = False
    withdrawn: bool = False
    schema: str = ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.binding, LearningCheckpointBinding):
            raise ResidualIntelligenceError("lineage requires LearningCheckpointBinding")
        if not isinstance(self.admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("lineage requires TrainingCorpusAdmission")
        if self.binding.corpus_id != self.admission.corpus_root:
            raise ResidualIntelligenceError("checkpoint corpus_id must match admission")
        if self.binding.split_id != self.admission.split_root:
            raise ResidualIntelligenceError("checkpoint split_id must match admission")


@dataclass(frozen=True)
class ResidualCheckpointAdapter:
    def validate_residual_resume(
        self,
        current: ExpertCheckpointLineage,
        incoming: ExpertCheckpointLineage,
    ) -> dict[str, Any]:
        if current.corrupt or incoming.corrupt:
            return {"ok": False, "reason": REASON_CORRUPT, "promoted": False}
        if current.withdrawn or incoming.withdrawn:
            return {"ok": False, "reason": REASON_WITHDRAWN, "promoted": False}
        if incoming.admission.admission_decision is not TrainingAvailability.ADMITTED:
            return {"ok": False, "reason": REASON_WITHDRAWN, "promoted": False}
        if current.binding.lineage_id != incoming.binding.lineage_id:
            return {"ok": False, "reason": REASON_INCOMPATIBLE_RESUME, "promoted": False}
        return {
            "ok": True,
            "reason": REASON_NO_PROMOTION,
            "promoted": False,
            "lineage_id": current.binding.lineage_id,
        }


def validate_residual_resume(
    current: ExpertCheckpointLineage,
    incoming: ExpertCheckpointLineage,
) -> dict[str, Any]:
    return ResidualCheckpointAdapter().validate_residual_resume(current, incoming)
