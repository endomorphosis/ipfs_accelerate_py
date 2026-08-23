"""Teacher disagreement and proof-grounded label production.

Validators and human-review authority remain external.  Confidence scores and
teacher order never decide ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    canonical_id,
    reject_candidate_authority,
    required_text,
    text_tuple,
)
from .corpus import LabelDisposition
from .local_experts import IndependentValidationReceipt
from .rights import TrainingCorpusAdmission

TEACHER_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-teacher-candidate@1"
)
TEACHER_DISAGREEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-teacher-disagreement@1"
)
LABEL_EVIDENCE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-label-evidence-policy@1"
)
PROOF_GROUNDED_LABEL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-proof-grounded-label@1"
)
INDEPENDENT_LABEL_PRODUCER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-independent-label-producer@1"
)
ADMITTED_PRODUCER_KINDS: Final[frozenset[str]] = frozenset(
    {
        "type",
        "static",
        "test",
        "proof",
        "policy",
        "authority",
        "effect",
        "merge",
        "human",
        "current_tree",
    }
)
REASON_NO_CONFIDENCE_GROUND_TRUTH: Final = "no_confidence_ground_truth"
REASON_UNRESOLVED: Final = "unresolved_teacher_disagreement"


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class TeacherCandidate:
    teacher_id: str
    producer_kind: str
    output: Mapping[str, Any]
    provenance_id: str
    confidence_ppm: int | None = None
    schema: str = TEACHER_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_CANDIDATE_SCHEMA:
            raise ResidualIntelligenceError("unsupported teacher candidate schema")
        object.__setattr__(self, "teacher_id", required_text(self.teacher_id, "teacher_id"))
        kind = required_text(self.producer_kind, "producer_kind")
        if kind not in ADMITTED_PRODUCER_KINDS:
            raise ResidualIntelligenceError(f"unknown independent producer: {kind}")
        object.__setattr__(self, "producer_kind", kind)
        if not isinstance(self.output, Mapping):
            raise ResidualIntelligenceError("teacher output must be a mapping")
        object.__setattr__(self, "output", dict(self.output))
        object.__setattr__(self, "provenance_id", required_text(self.provenance_id, "provenance_id"))
        if self.confidence_ppm is not None:
            if type(self.confidence_ppm) is not int or not 0 <= self.confidence_ppm <= 1_000_000:
                raise ResidualIntelligenceError("confidence_ppm must be in [0, 1000000]")
        reject_candidate_authority(self.output)

    @property
    def candidate_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "teacher_id": self.teacher_id,
            "producer_kind": self.producer_kind,
            "output": dict(self.output),
            "provenance_id": self.provenance_id,
            "confidence_ppm": self.confidence_ppm,
        }
        if include_id:
            payload["candidate_id"] = self.candidate_id
        return payload


@dataclass(frozen=True)
class TeacherDisagreement:
    question_id: str
    family: ResidualTaskFamily
    teachers: tuple[TeacherCandidate, ...]
    validators: tuple[IndependentValidationReceipt, ...]
    counterexample_ids: tuple[str, ...]
    disposition: LabelDisposition
    schema: str = TEACHER_DISAGREEMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_DISAGREEMENT_SCHEMA:
            raise ResidualIntelligenceError("unsupported teacher disagreement schema")
        object.__setattr__(self, "question_id", required_text(self.question_id, "question_id"))
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        teachers = tuple(self.teachers)
        if len(teachers) < 2:
            raise ResidualIntelligenceError("disagreement requires at least two teachers")
        object.__setattr__(self, "teachers", teachers)
        object.__setattr__(self, "validators", tuple(self.validators))
        object.__setattr__(
            self,
            "counterexample_ids",
            text_tuple(self.counterexample_ids, "counterexample_ids", max_items=64),
        )
        object.__setattr__(self, "disposition", LabelDisposition(self.disposition))
        if self.disposition is LabelDisposition.ACCEPTED:
            raise ResidualIntelligenceError(REASON_NO_CONFIDENCE_GROUND_TRUTH)

    @property
    def disagreement_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "question_id": self.question_id,
            "family": self.family.value,
            "teachers": tuple(item.to_dict() for item in self.teachers),
            "validators": tuple(item.to_dict() for item in self.validators),
            "counterexample_ids": self.counterexample_ids,
            "disposition": self.disposition.value,
        }
        if include_id:
            payload["disagreement_id"] = self.disagreement_id
        return payload


def resolve_teacher_disagreement(
    question_id: str,
    family: ResidualTaskFamily,
    teachers: Sequence[TeacherCandidate],
    validators: Sequence[IndependentValidationReceipt],
    *,
    counterexample_ids: Sequence[str] = (),
) -> TeacherDisagreement:
    if any(item.confidence_ppm is not None for item in teachers):
        # Confidence may be recorded but cannot elect a winner.
        pass
    outputs = [tuple(sorted(item.output.items())) for item in teachers]
    agreed = len(set(outputs)) == 1
    accepted_validators = [item for item in validators if item.accepted is True]
    rejected_validators = [item for item in validators if item.accepted is not True]
    if agreed and accepted_validators and not rejected_validators and not counterexample_ids:
        disposition = LabelDisposition.INCONCLUSIVE
        # Agreement is not ground truth; keep inconclusive until proof-grounded
        # production admits an independent current-tree producer.
    elif counterexample_ids and rejected_validators:
        disposition = LabelDisposition.AMBIGUOUS
    elif any(item.producer_kind == "human" for item in teachers):
        disposition = LabelDisposition.HUMAN_REVIEW_REQUIRED
    else:
        disposition = LabelDisposition.INCONCLUSIVE
    return TeacherDisagreement(
        question_id=question_id,
        family=family,
        teachers=tuple(teachers),
        validators=tuple(validators),
        counterexample_ids=tuple(counterexample_ids),
        disposition=disposition,
    )


@dataclass(frozen=True)
class IndependentLabelProducer:
    producer_id: str
    kind: str
    current_tree_cid: str
    schema: str = INDEPENDENT_LABEL_PRODUCER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer_id", required_text(self.producer_id, "producer_id"))
        kind = required_text(self.kind, "kind")
        if kind not in ADMITTED_PRODUCER_KINDS:
            raise ResidualIntelligenceError(f"unknown independent producer: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "current_tree_cid", required_text(self.current_tree_cid, "current_tree_cid")
        )


@dataclass(frozen=True)
class LabelEvidencePolicy:
    require_positive_and_negative: bool = True
    forbid_model_agreement_reward: bool = True
    exact_tree_required: bool = True
    schema: str = LABEL_EVIDENCE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "require_positive_and_negative",
            _require_bool(self.require_positive_and_negative, "require_positive_and_negative"),
        )
        object.__setattr__(
            self,
            "forbid_model_agreement_reward",
            _require_bool(self.forbid_model_agreement_reward, "forbid_model_agreement_reward"),
        )
        object.__setattr__(
            self,
            "exact_tree_required",
            _require_bool(self.exact_tree_required, "exact_tree_required"),
        )


@dataclass(frozen=True)
class ProofGroundedLabel:
    question_id: str
    family: ResidualTaskFamily
    tree_cid: str
    producers: tuple[IndependentLabelProducer, ...]
    positive_evidence_ids: tuple[str, ...]
    negative_evidence_ids: tuple[str, ...]
    disposition: LabelDisposition
    admission_id: str = ""
    schema: str = PROOF_GROUNDED_LABEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROOF_GROUNDED_LABEL_SCHEMA:
            raise ResidualIntelligenceError("unsupported proof-grounded label schema")
        object.__setattr__(self, "question_id", required_text(self.question_id, "question_id"))
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "tree_cid", required_text(self.tree_cid, "tree_cid"))
        producers = tuple(self.producers)
        if not producers:
            raise ResidualIntelligenceError("label requires independent producers")
        object.__setattr__(self, "producers", producers)
        object.__setattr__(
            self,
            "positive_evidence_ids",
            text_tuple(self.positive_evidence_ids, "positive_evidence_ids", max_items=64),
        )
        object.__setattr__(
            self,
            "negative_evidence_ids",
            text_tuple(self.negative_evidence_ids, "negative_evidence_ids", max_items=64),
        )
        object.__setattr__(self, "disposition", LabelDisposition(self.disposition))
        object.__setattr__(
            self,
            "admission_id",
            "" if self.admission_id in (None, "") else required_text(self.admission_id, "admission_id"),
        )

    @property
    def label_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "question_id": self.question_id,
            "family": self.family.value,
            "tree_cid": self.tree_cid,
            "producers": tuple(
                {
                    "producer_id": item.producer_id,
                    "kind": item.kind,
                    "current_tree_cid": item.current_tree_cid,
                }
                for item in self.producers
            ),
            "positive_evidence_ids": self.positive_evidence_ids,
            "negative_evidence_ids": self.negative_evidence_ids,
            "disposition": self.disposition.value,
            "admission_id": self.admission_id,
        }
        if include_id:
            payload["label_id"] = self.label_id
        return payload


def produce_proof_grounded_label(
    *,
    question_id: str,
    family: ResidualTaskFamily,
    tree_cid: str,
    producers: Sequence[IndependentLabelProducer],
    positive_evidence_ids: Sequence[str],
    negative_evidence_ids: Sequence[str],
    policy: LabelEvidencePolicy,
    admission: TrainingCorpusAdmission | None = None,
    disagreement: TeacherDisagreement | None = None,
) -> ProofGroundedLabel:
    if policy.exact_tree_required and any(
        item.current_tree_cid != tree_cid for item in producers
    ):
        disposition = LabelDisposition.INCONCLUSIVE
    elif policy.require_positive_and_negative and (
        not positive_evidence_ids or not negative_evidence_ids
    ):
        disposition = LabelDisposition.AMBIGUOUS
    elif disagreement is not None:
        disposition = disagreement.disposition
    elif not producers:
        disposition = LabelDisposition.INCONCLUSIVE
    else:
        disposition = LabelDisposition.ACCEPTED
    if policy.forbid_model_agreement_reward and disagreement is not None:
        if disagreement.disposition is not LabelDisposition.ACCEPTED:
            disposition = disagreement.disposition
    admission_id = ""
    if admission is not None:
        admission_id = admission.admission_id
    return ProofGroundedLabel(
        question_id=question_id,
        family=family,
        tree_cid=tree_cid,
        producers=tuple(producers),
        positive_evidence_ids=tuple(positive_evidence_ids),
        negative_evidence_ids=tuple(negative_evidence_ids),
        disposition=disposition,
        admission_id=admission_id,
    )
