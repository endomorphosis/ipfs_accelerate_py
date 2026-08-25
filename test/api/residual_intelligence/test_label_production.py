from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.corpus import LabelDisposition
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.labels import (
    IndependentLabelProducer,
    LabelEvidencePolicy,
    produce_proof_grounded_label,
)
from .helpers import admission


TREE = "tree:final"


def producers() -> tuple[IndependentLabelProducer, ...]:
    return (
        IndependentLabelProducer(
            producer_id="producer:test", kind="test", current_tree_cid=TREE
        ),
        IndependentLabelProducer(
            producer_id="producer:proof", kind="proof", current_tree_cid=TREE
        ),
    )


def test_positive_and_negative_current_tree_evidence() -> None:
    record, _examples = admission()
    label = produce_proof_grounded_label(
        question_id="question:label-1",
        family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        tree_cid=TREE,
        producers=producers(),
        positive_evidence_ids=("test:pass",),
        negative_evidence_ids=("counter:1",),
        policy=LabelEvidencePolicy(),
        admission=record,
    )
    assert label.disposition is LabelDisposition.ACCEPTED
    assert label.admission_id == record.admission_id
    assert label.tree_cid == TREE


def test_missing_negative_or_stale_tree_does_not_force_clean_labels() -> None:
    ambiguous = produce_proof_grounded_label(
        question_id="question:label-2",
        family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        tree_cid=TREE,
        producers=producers(),
        positive_evidence_ids=("test:pass",),
        negative_evidence_ids=(),
        policy=LabelEvidencePolicy(),
    )
    assert ambiguous.disposition is LabelDisposition.AMBIGUOUS
    stale = produce_proof_grounded_label(
        question_id="question:label-3",
        family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        tree_cid=TREE,
        producers=(
            IndependentLabelProducer(
                producer_id="producer:old", kind="test", current_tree_cid="tree:old"
            ),
        ),
        positive_evidence_ids=("test:pass",),
        negative_evidence_ids=("counter:1",),
        policy=LabelEvidencePolicy(),
    )
    assert stale.disposition is LabelDisposition.INCONCLUSIVE
