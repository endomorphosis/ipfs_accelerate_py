from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    TrainingAvailability,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.corpus import (
    ResidualDistillationCorpus,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.rights import (
    SourceRight,
    TrainingCorpusAdmission,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.splits import (
    SplitPartition,
    assert_training_view_excludes_hidden,
)

from .helpers import SOURCE_ID, admission, split_fixture


def test_corpus_admission_and_rights_round_trip() -> None:
    record, examples = admission()
    assert record.can_train is True
    assert TrainingCorpusAdmission.from_dict(record.to_dict()) == record
    corpus = ResidualDistillationCorpus(admission=record, examples=examples)
    assert corpus.summary()["example_count"] == 4
    corpus.require_training_admitted()


def test_rights_rejection_is_training_unavailable() -> None:
    record, examples = admission(admitted=False)
    corpus = ResidualDistillationCorpus(admission=record, examples=examples)
    assert corpus.training_availability == "training_unavailable"
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        corpus.require_training_admitted()


def test_denied_rights_cannot_be_marked_admitted() -> None:
    record, _examples = admission()
    payload = record.to_dict(include_id=False)
    payload["source_rights"] = {SOURCE_ID: SourceRight.DENIED.value}
    payload["admission_decision"] = TrainingAvailability.ADMITTED.value
    with pytest.raises(ResidualIntelligenceError, match="training rights"):
        TrainingCorpusAdmission.from_dict(payload)


def test_credentials_never_enter_training() -> None:
    record, _examples = admission()
    payload = record.to_dict(include_id=False)
    payload["privacy_classification"] = PrivacyClass.CREDENTIAL.value
    payload["admission_decision"] = TrainingAvailability.ADMITTED.value
    with pytest.raises(ResidualIntelligenceError, match="never"):
        TrainingCorpusAdmission.from_dict(payload)


def test_lineage_safe_grouped_split_and_holdout_protection() -> None:
    examples, manifest = split_fixture()
    partition_by_group = {item.split_group: item.partition for item in manifest.assignments}
    assert partition_by_group["dev-a"] is SplitPartition.DEVELOPMENT
    assert partition_by_group["holdout-a"] is SplitPartition.HELD_OUT
    assert partition_by_group["adversarial-a"] is SplitPartition.ADVERSARIAL
    assert manifest.leakage_audit().passed is True

    training_ids = [
        item.example_identity
        for item in manifest.assignments
        if item.partition is SplitPartition.TRAIN
    ]
    assert_training_view_excludes_hidden(manifest, training_ids)
    hidden_id = next(
        item.example_identity for item in manifest.assignments if item.hidden_from_training
    )
    with pytest.raises(ResidualIntelligenceError, match="hidden"):
        assert_training_view_excludes_hidden(manifest, [hidden_id])
    assert len(examples) == len(manifest.assignments)


def test_holdout_leakage_cannot_be_hidden_by_average() -> None:
    _examples, manifest = split_fixture()
    assignment = next(item for item in manifest.assignments if item.hidden_from_training)
    with pytest.raises(ResidualIntelligenceError, match="hidden"):
        assert_training_view_excludes_hidden(manifest, [assignment.example_identity])
