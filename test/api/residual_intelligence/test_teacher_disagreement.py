from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.corpus import LabelDisposition
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.labels import (
    TeacherCandidate,
    resolve_teacher_disagreement,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.local_experts import (
    IndependentValidationReceipt,
)


def teacher(index: int, output: str, *, kind: str = "static", confidence=None) -> TeacherCandidate:
    return TeacherCandidate(
        teacher_id=f"teacher:{index}",
        producer_kind=kind,
        output={"label": output},
        provenance_id=f"prov:{index}",
        confidence_ppm=confidence,
    )


def test_all_teachers_preserved_and_confidence_cannot_elect_winner() -> None:
    record = resolve_teacher_disagreement(
        "question:1",
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        (
            teacher(1, "missing_edge", confidence=990_000),
            teacher(2, "timeout", confidence=10_000),
        ),
        (
            IndependentValidationReceipt(
                validator_identity="validator:a@1", accepted=True
            ),
            IndependentValidationReceipt(
                validator_identity="validator:b@1", accepted=False
            ),
        ),
        counterexample_ids=("counter:1",),
    )
    assert len(record.teachers) == 2
    assert record.disposition in {
        LabelDisposition.AMBIGUOUS,
        LabelDisposition.INCONCLUSIVE,
        LabelDisposition.HUMAN_REVIEW_REQUIRED,
    }
    assert record.disposition is not LabelDisposition.ACCEPTED
    assert record.counterexample_ids == ("counter:1",)
    assert record.validators[1].accepted is False


def test_human_review_required_when_a_human_teacher_is_present() -> None:
    record = resolve_teacher_disagreement(
        "question:2",
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        (teacher(1, "a"), teacher(2, "b", kind="human")),
        (),
    )
    assert record.disposition is LabelDisposition.HUMAN_REVIEW_REQUIRED
