from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.checkpoint import (
    REASON_CORRUPT,
    REASON_NO_PROMOTION,
    REASON_WITHDRAWN,
    ExpertCheckpointLineage,
    validate_residual_resume,
)
from ipfs_accelerate_py.agent_supervisor.runtime.learning_checkpoint import LearningCheckpointBinding
from .helpers import admission


def binding(record) -> LearningCheckpointBinding:
    return LearningCheckpointBinding(
        architecture_id="arch:1",
        weights_id="weights:1",
        optimizer_id="opt:1",
        scheduler_id="sched:1",
        tokenizer_id=record.tokenizer_identity,
        vocab_id="vocab:1",
        cursor_id="cursor:1",
        corpus_id=record.corpus_root,
        split_id=record.split_root,
        curriculum_id="curr:1",
        loss_id="loss:1",
        random_id="seed:1",
        env_id=record.environment,
        code_id="code:1",
        compiler_id=record.compiler_identity,
    )


def test_compatible_resume_is_not_promotion() -> None:
    record, _ = admission()
    lineage = ExpertCheckpointLineage(binding=binding(record), admission=record)
    result = validate_residual_resume(lineage, lineage)
    assert result["ok"] is True
    assert result["promoted"] is False
    assert result["reason"] == REASON_NO_PROMOTION


def test_corrupt_and_withdrawn_checkpoints_reject() -> None:
    record, _ = admission()
    live = ExpertCheckpointLineage(binding=binding(record), admission=record)
    corrupt = ExpertCheckpointLineage(binding=binding(record), admission=record, corrupt=True)
    withdrawn = ExpertCheckpointLineage(binding=binding(record), admission=record, withdrawn=True)
    assert validate_residual_resume(live, corrupt)["reason"] == REASON_CORRUPT
    assert validate_residual_resume(live, withdrawn)["reason"] == REASON_WITHDRAWN
