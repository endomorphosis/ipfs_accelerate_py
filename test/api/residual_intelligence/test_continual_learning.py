from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.continual_learning import (
    ContinualLearningEpoch,
    ResumeCompatibility,
    TrainingEpochLimits,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    TrainingAvailability,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.training_plan import TrainingPlan
from .helpers import admission


def plan(*, admitted: bool = True) -> TrainingPlan:
    record, _ = admission(admitted=admitted)
    return TrainingPlan(
        parent_model_id="model:parent",
        architecture_id="arch:1",
        tokenizer_id=record.tokenizer_identity,
        corpus_root=record.corpus_root,
        split_root=record.split_root,
        curriculum_id="curr:1",
        loss_id="loss:1",
        optimizer_id="opt:1",
        scheduler_id="sched:1",
        seed_id="seed:1",
        environment_id=record.environment,
        code_id="code:1",
        compiler_id=record.compiler_identity,
        evaluation_id="eval:1",
        admission=record,
    )


def test_offline_epoch_binds_admission_and_forbids_promotion() -> None:
    epoch = ContinualLearningEpoch(plan=plan(), limits=TrainingEpochLimits())
    assert epoch.offline is True
    assert epoch.to_dict()["promoted"] is False
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        plan(admitted=False)


def test_resume_requires_exact_lineage_and_cannot_widen_bounds() -> None:
    first = ContinualLearningEpoch(plan=plan(), limits=TrainingEpochLimits())
    same = ContinualLearningEpoch(plan=plan(), limits=TrainingEpochLimits())
    assert first.resume_compatibility(same).compatible is True
    with pytest.raises(ResidualIntelligenceError):
        TrainingEpochLimits(examples=50_001)
