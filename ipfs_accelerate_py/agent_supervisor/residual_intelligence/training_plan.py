"""Offline training-plan bindings. Promotion is never implied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, TrainingAvailability, required_text
from .rights import TrainingCorpusAdmission

TRAINING_PLAN_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-training-plan@1"


@dataclass(frozen=True)
class TrainingPlan:
    parent_model_id: str
    architecture_id: str
    tokenizer_id: str
    corpus_root: str
    split_root: str
    curriculum_id: str
    loss_id: str
    optimizer_id: str
    scheduler_id: str
    seed_id: str
    environment_id: str
    code_id: str
    compiler_id: str
    evaluation_id: str
    admission: TrainingCorpusAdmission
    schema: str = TRAINING_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINING_PLAN_SCHEMA:
            raise ResidualIntelligenceError("unsupported training plan schema")
        for name in (
            "parent_model_id",
            "architecture_id",
            "tokenizer_id",
            "corpus_root",
            "split_root",
            "curriculum_id",
            "loss_id",
            "optimizer_id",
            "scheduler_id",
            "seed_id",
            "environment_id",
            "code_id",
            "compiler_id",
            "evaluation_id",
        ):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        if not isinstance(self.admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("training plan requires TrainingCorpusAdmission")
        if self.admission.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError("training_unavailable")
        if self.admission.corpus_root != self.corpus_root or self.admission.split_root != self.split_root:
            raise ResidualIntelligenceError("epoch corpus/split roots must match admission")
