"""Proof-grounded residual distillation corpus contracts and builders."""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    bounded_json_mapping,
    canonical_id,
    reject_candidate_authority,
    reject_secret_material,
    required_text,
    strict_fields,
    text_tuple,
)
from .rights import TrainingCorpusAdmission

DISTILLATION_EXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-distillation-example@1"
)
DISTILLATION_CORPUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-distillation-corpus@1"
)
MAX_ALTERNATIVES: Final = 64


class LabelDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    INCONCLUSIVE = "inconclusive"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class CorpusSourceKind(str, Enum):
    FIRST_PARTY_TRAJECTORY = "first_party_trajectory"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    ADVERSARIAL_MUTANT = "adversarial_mutant"
    MECHANICAL_COUNTEREXAMPLE = "mechanical_counterexample"
    AUTHORIZED_PRIVATE = "authorized_private"
    HUMAN_REVIEWED = "human_reviewed"
    LICENSED_PUBLIC = "licensed_public"


_FORBIDDEN_TRAINING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "private_chain_of_thought",
        "hidden_test_body",
        "raw_secret",
        "credential_value",
    }
)


def _strict_artifact_mapping(value: Any, name: str) -> dict[str, Any]:
    result = bounded_json_mapping(value, name)
    reject_secret_material(result, noun=name)
    for key in result:
        if str(key).strip().casefold() in _FORBIDDEN_TRAINING_FIELDS:
            raise ResidualIntelligenceError(f"{name} contains forbidden field {key!r}")
    return result


def _mapping_tuple(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ResidualIntelligenceError(f"{name} must be a sequence of objects")
    if len(value) > MAX_ALTERNATIVES:
        raise ResidualIntelligenceError(f"{name} exceeds {MAX_ALTERNATIVES} items")
    return tuple(
        _strict_artifact_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class ResidualDistillationExample:
    """One lineage-bound, independently labelled residual example."""

    task_family: ResidualTaskFamily
    input_feature_identity: str
    context_identity: str
    source_identity: str
    source_kind: CorpusSourceKind
    teacher_or_source_producer: str
    teacher_output: Mapping[str, Any]
    independent_validation: tuple[str, ...]
    label_disposition: LabelDisposition
    accepted_output: Mapping[str, Any]
    rejected_alternatives: tuple[Mapping[str, Any], ...]
    counterexamples: tuple[str, ...]
    proof_test_evidence: tuple[str, ...]
    repository_family: str
    language_framework: str
    rights_reference: str
    privacy_class: PrivacyClass
    split_group: str
    semantic_lineage: tuple[str, ...]
    adversarial: bool = False
    boundary_case: bool = False
    hidden_test_derived: bool = False
    schema: str = DISTILLATION_EXAMPLE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "example_identity",
            "task_family",
            "input_feature_identity",
            "context_identity",
            "source_identity",
            "source_kind",
            "teacher_or_source_producer",
            "teacher_output",
            "independent_validation",
            "label_disposition",
            "accepted_output",
            "rejected_alternatives",
            "counterexamples",
            "proof_test_evidence",
            "repository_family",
            "language_framework",
            "rights_reference",
            "privacy_class",
            "split_group",
            "semantic_lineage",
            "adversarial",
            "boundary_case",
            "hidden_test_derived",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != DISTILLATION_EXAMPLE_SCHEMA:
            raise ResidualIntelligenceError("unsupported distillation example schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "source_kind", CorpusSourceKind(self.source_kind))
        object.__setattr__(self, "label_disposition", LabelDisposition(self.label_disposition))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        for field in (
            "input_feature_identity",
            "context_identity",
            "source_identity",
            "teacher_or_source_producer",
            "repository_family",
            "language_framework",
            "rights_reference",
            "split_group",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        teacher = _strict_artifact_mapping(self.teacher_output, "teacher_output")
        accepted = _strict_artifact_mapping(self.accepted_output, "accepted_output")
        reject_candidate_authority(teacher)
        reject_candidate_authority(accepted)
        object.__setattr__(self, "teacher_output", teacher)
        object.__setattr__(self, "accepted_output", accepted)
        object.__setattr__(
            self,
            "rejected_alternatives",
            _mapping_tuple(self.rejected_alternatives, "rejected_alternatives"),
        )
        for alternative in self.rejected_alternatives:
            reject_candidate_authority(alternative)
        for field in (
            "independent_validation",
            "counterexamples",
            "proof_test_evidence",
            "semantic_lineage",
        ):
            object.__setattr__(self, field, text_tuple(getattr(self, field), field))
        if not self.semantic_lineage:
            raise ResidualIntelligenceError("semantic_lineage must not be empty")
        for field in ("adversarial", "boundary_case", "hidden_test_derived"):
            if type(getattr(self, field)) is not bool:
                raise ResidualIntelligenceError(f"{field} must be boolean")
        if self.hidden_test_derived:
            raise ResidualIntelligenceError("hidden-test-derived examples are forbidden")
        if self.privacy_class in {PrivacyClass.CREDENTIAL, PrivacyClass.PROOF_WITNESS}:
            raise ResidualIntelligenceError(
                f"{self.privacy_class.value} examples cannot enter a training corpus"
            )
        if self.label_disposition is LabelDisposition.ACCEPTED:
            if not self.independent_validation or not self.accepted_output:
                raise ResidualIntelligenceError(
                    "positive examples require independent validation and accepted output"
                )
            if not self.proof_test_evidence:
                raise ResidualIntelligenceError(
                    "positive examples require current proof or test evidence"
                )
        if self.label_disposition is LabelDisposition.REJECTED and not (
            self.rejected_alternatives or self.counterexamples
        ):
            raise ResidualIntelligenceError(
                "negative examples require a rejected alternative or counterexample"
            )

    @property
    def example_identity(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "input_feature_identity": self.input_feature_identity,
            "context_identity": self.context_identity,
            "source_identity": self.source_identity,
            "source_kind": self.source_kind.value,
            "teacher_or_source_producer": self.teacher_or_source_producer,
            "teacher_output": dict(self.teacher_output),
            "independent_validation": list(self.independent_validation),
            "label_disposition": self.label_disposition.value,
            "accepted_output": dict(self.accepted_output),
            "rejected_alternatives": [dict(item) for item in self.rejected_alternatives],
            "counterexamples": list(self.counterexamples),
            "proof_test_evidence": list(self.proof_test_evidence),
            "repository_family": self.repository_family,
            "language_framework": self.language_framework,
            "rights_reference": self.rights_reference,
            "privacy_class": self.privacy_class.value,
            "split_group": self.split_group,
            "semantic_lineage": list(self.semantic_lineage),
            "adversarial": self.adversarial,
            "boundary_case": self.boundary_case,
            "hidden_test_derived": False,
        }
        if include_id:
            result["example_identity"] = self.example_identity
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualDistillationExample:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"example_identity"},
            noun="residual distillation example",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            input_feature_identity=str(payload.get("input_feature_identity") or ""),
            context_identity=str(payload.get("context_identity") or ""),
            source_identity=str(payload.get("source_identity") or ""),
            source_kind=CorpusSourceKind(str(payload.get("source_kind") or "")),
            teacher_or_source_producer=str(payload.get("teacher_or_source_producer") or ""),
            teacher_output=payload.get("teacher_output") or {},
            independent_validation=tuple(payload.get("independent_validation") or ()),
            label_disposition=LabelDisposition(str(payload.get("label_disposition") or "")),
            accepted_output=payload.get("accepted_output") or {},
            rejected_alternatives=tuple(payload.get("rejected_alternatives") or ()),
            counterexamples=tuple(payload.get("counterexamples") or ()),
            proof_test_evidence=tuple(payload.get("proof_test_evidence") or ()),
            repository_family=str(payload.get("repository_family") or ""),
            language_framework=str(payload.get("language_framework") or ""),
            rights_reference=str(payload.get("rights_reference") or ""),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            split_group=str(payload.get("split_group") or ""),
            semantic_lineage=tuple(payload.get("semantic_lineage") or ()),
            adversarial=payload.get("adversarial"),
            boundary_case=payload.get("boundary_case"),
            hidden_test_derived=payload.get("hidden_test_derived"),
        )
        claimed = str(payload.get("example_identity") or "")
        if claimed and claimed != result.example_identity:
            raise ResidualIntelligenceError("distillation example identity mismatch")
        return result


@dataclass(frozen=True)
class ResidualDistillationCorpus:
    """Bounded examples plus their exact admission decision."""

    admission: TrainingCorpusAdmission
    examples: tuple[ResidualDistillationExample, ...]
    schema: str = DISTILLATION_CORPUS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DISTILLATION_CORPUS_SCHEMA:
            raise ResidualIntelligenceError("unsupported distillation corpus schema")
        if not isinstance(self.admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("corpus requires TrainingCorpusAdmission")
        examples = tuple(self.examples)
        if len(examples) > 1_000_000:
            raise ResidualIntelligenceError("corpus exceeds bounded example population")
        if any(not isinstance(item, ResidualDistillationExample) for item in examples):
            raise ResidualIntelligenceError("corpus examples must be typed records")
        ids = [item.example_identity for item in examples]
        if len(set(ids)) != len(ids):
            raise ResidualIntelligenceError("corpus contains duplicate example identities")
        sources = set(self.admission.source_identities)
        if any(item.source_identity not in sources for item in examples):
            raise ResidualIntelligenceError("corpus example has no source admission binding")
        if any(item.rights_reference != self.admission.source_rights_root for item in examples):
            raise ResidualIntelligenceError("corpus example rights reference mismatch")
        object.__setattr__(self, "examples", examples)

    @property
    def corpus_identity(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def training_availability(self) -> str:
        return self.admission.admission_decision.value

    def require_training_admitted(self) -> None:
        self.admission.require_training_admitted()

    def summary(self) -> dict[str, Any]:
        return {
            "example_count": len(self.examples),
            "task_family_counts": dict(
                sorted(Counter(item.task_family.value for item in self.examples).items())
            ),
            "label_counts": dict(
                sorted(Counter(item.label_disposition.value for item in self.examples).items())
            ),
            "adversarial_count": sum(1 for item in self.examples if item.adversarial),
            "boundary_count": sum(1 for item in self.examples if item.boundary_case),
            "training_availability": self.training_availability,
        }

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "admission": self.admission.to_dict(),
            "examples": [item.to_dict() for item in self.examples],
            "summary": self.summary(),
        }
        if include_id:
            result["corpus_identity"] = self.corpus_identity
        return result


def build_first_party_trajectory_corpus(
    *,
    admission: TrainingCorpusAdmission,
    rows: Sequence[Mapping[str, Any]],
) -> ResidualDistillationCorpus:
    """Compile explicit first-party rows without inferring rights from existence."""

    examples = tuple(ResidualDistillationExample.from_dict(row) for row in rows)
    if any(item.source_kind is not CorpusSourceKind.FIRST_PARTY_TRAJECTORY for item in examples):
        raise ResidualIntelligenceError("first-party builder received a non-first-party row")
    return ResidualDistillationCorpus(admission=admission, examples=examples)


def build_synthetic_adversarial_corpus(
    *,
    admission: TrainingCorpusAdmission,
    rows: Sequence[Mapping[str, Any]],
) -> ResidualDistillationCorpus:
    """Compile only synthetic, mutant, or mechanical counterexample rows."""

    examples = tuple(ResidualDistillationExample.from_dict(row) for row in rows)
    allowed = {
        CorpusSourceKind.SYNTHETIC_FIXTURE,
        CorpusSourceKind.ADVERSARIAL_MUTANT,
        CorpusSourceKind.MECHANICAL_COUNTEREXAMPLE,
    }
    if any(item.source_kind not in allowed for item in examples):
        raise ResidualIntelligenceError("synthetic builder received an unauthorized source kind")
    return ResidualDistillationCorpus(admission=admission, examples=examples)


__all__ = (
    "CorpusSourceKind",
    "DISTILLATION_CORPUS_SCHEMA",
    "DISTILLATION_EXAMPLE_SCHEMA",
    "LabelDisposition",
    "ResidualDistillationCorpus",
    "ResidualDistillationExample",
    "build_first_party_trajectory_corpus",
    "build_synthetic_adversarial_corpus",
)
