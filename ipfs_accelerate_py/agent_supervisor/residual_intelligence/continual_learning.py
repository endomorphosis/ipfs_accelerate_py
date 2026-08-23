"""Bounded offline continual-learning epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, TrainingAvailability, required_text
from .training_plan import TrainingPlan

EPOCH_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-continual-learning-epoch@1"
LIMITS_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-training-epoch-limits@1"


@dataclass(frozen=True)
class TrainingEpochLimits:
    examples: int = 50_000
    steps: int = 10_000
    wall_seconds: int = 14_400
    gpu_seconds: int = 10_800
    spend_usd: int = 25
    candidates: int = 3
    checkpoints: int = 4
    human_reviews: int = 50
    schema: str = LIMITS_SCHEMA

    def __post_init__(self) -> None:
        bounds = {
            "examples": 50_000,
            "steps": 10_000,
            "wall_seconds": 14_400,
            "gpu_seconds": 10_800,
            "spend_usd": 25,
            "candidates": 3,
            "checkpoints": 4,
            "human_reviews": 50,
        }
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0 or value > maximum:
                raise ResidualIntelligenceError(f"{name} exceeds epoch bound {maximum}")


@dataclass(frozen=True)
class ResumeCompatibility:
    compatible: bool
    reason: str = ""


@dataclass(frozen=True)
class ContinualLearningEpoch:
    plan: TrainingPlan
    limits: TrainingEpochLimits
    offline: bool = True
    status: str = "planned"
    schema: str = EPOCH_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.plan, TrainingPlan):
            raise ResidualIntelligenceError("epoch requires TrainingPlan")
        if not isinstance(self.limits, TrainingEpochLimits):
            raise ResidualIntelligenceError("epoch requires TrainingEpochLimits")
        if self.offline is not True:
            raise ResidualIntelligenceError("epochs are offline-only")
        if self.status not in {"planned", "running", "cancelled", "complete"}:
            raise ResidualIntelligenceError("unknown epoch status")
        if self.status in {"planned", "running"}:
            if self.plan.admission.admission_decision is not TrainingAvailability.ADMITTED:
                raise ResidualIntelligenceError("training_unavailable")

    def resume_compatibility(self, other: ContinualLearningEpoch) -> ResumeCompatibility:
        fields = (
            "parent_model_id",
            "architecture_id",
            "tokenizer_id",
            "corpus_root",
            "split_root",
            "code_id",
            "compiler_id",
            "environment_id",
        )
        for name in fields:
            if getattr(self.plan, name) != getattr(other.plan, name):
                return ResumeCompatibility(False, f"incompatible:{name}")
        if other.limits != self.limits:
            return ResumeCompatibility(False, "incompatible:limits")
        return ResumeCompatibility(True, "exact-lineage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "offline": True,
            "promoted": False,
            "parent_model_id": self.plan.parent_model_id,
            "corpus_root": self.plan.corpus_root,
        }
