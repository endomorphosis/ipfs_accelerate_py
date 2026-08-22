"""Mandatory corpus-rights and training-admission gate."""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    TrainingAvailability,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)

TRAINING_CORPUS_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/training-corpus-admission@1"
)
LEAKAGE_AUDIT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-corpus-leakage-audit@1"


class SourceRight(str, Enum):
    FIRST_PARTY_OWNED = "first_party_owned"
    SYNTHETIC_GENERATED = "synthetic_generated"
    LICENSED_FOR_TRAINING = "licensed_for_training"
    EXPLICIT_PRIVATE_AUTHORIZATION = "explicit_private_authorization"
    HUMAN_REVIEWED_WITH_RIGHTS = "human_reviewed_with_rights"
    UNKNOWN = "unknown"
    DENIED = "denied"


class TransformationRight(str, Enum):
    TRAINING_AND_DERIVATIVES_PERMITTED = "training_and_derivatives_permitted"
    INTERNAL_TRAINING_ONLY = "internal_training_only"
    SCOPED_PRIVATE_TRAINING_ONLY = "scoped_private_training_only"
    EVALUATION_ONLY = "evaluation_only"
    UNKNOWN = "unknown"
    DENIED = "denied"


_ADMITTED_SOURCE_RIGHTS: Final[frozenset[SourceRight]] = frozenset(
    {
        SourceRight.FIRST_PARTY_OWNED,
        SourceRight.SYNTHETIC_GENERATED,
        SourceRight.LICENSED_FOR_TRAINING,
        SourceRight.EXPLICIT_PRIVATE_AUTHORIZATION,
        SourceRight.HUMAN_REVIEWED_WITH_RIGHTS,
    }
)
_ADMITTED_TRANSFORMATION_RIGHTS: Final[frozenset[TransformationRight]] = frozenset(
    {
        TransformationRight.TRAINING_AND_DERIVATIVES_PERMITTED,
        TransformationRight.INTERNAL_TRAINING_ONLY,
        TransformationRight.SCOPED_PRIVATE_TRAINING_ONLY,
    }
)
_NEVER_TRAIN_PRIVACY: Final[frozenset[PrivacyClass]] = frozenset(
    {PrivacyClass.CREDENTIAL, PrivacyClass.PROOF_WITNESS}
)


@dataclass(frozen=True)
class LeakageAudit:
    """Grouped-split audit that never exposes hidden-test bodies."""

    split_root: str
    grouping_policy_id: str
    train_group_count: int
    development_group_count: int
    holdout_group_count: int
    adversarial_group_count: int
    cross_partition_group_count: int
    duplicate_example_count: int
    hidden_test_commitment: str
    hidden_test_bodies_accessed: bool = False
    passed: bool = False
    schema: str = LEAKAGE_AUDIT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "audit_id",
            "split_root",
            "grouping_policy_id",
            "train_group_count",
            "development_group_count",
            "holdout_group_count",
            "adversarial_group_count",
            "cross_partition_group_count",
            "duplicate_example_count",
            "hidden_test_commitment",
            "hidden_test_bodies_accessed",
            "passed",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LEAKAGE_AUDIT_SCHEMA:
            raise ResidualIntelligenceError("unsupported leakage audit schema")
        for field in ("split_root", "grouping_policy_id", "hidden_test_commitment"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        for field in (
            "train_group_count",
            "development_group_count",
            "holdout_group_count",
            "adversarial_group_count",
            "cross_partition_group_count",
            "duplicate_example_count",
        ):
            object.__setattr__(
                self,
                field,
                bounded_int(getattr(self, field), field, minimum=0, maximum=100_000_000),
            )
        if type(self.hidden_test_bodies_accessed) is not bool or type(self.passed) is not bool:
            raise ResidualIntelligenceError("leakage audit flags must be boolean")
        derived_pass = (
            self.cross_partition_group_count == 0
            and self.duplicate_example_count == 0
            and not self.hidden_test_bodies_accessed
            and self.holdout_group_count > 0
            and self.adversarial_group_count > 0
        )
        if self.passed != derived_pass:
            raise ResidualIntelligenceError(
                "leakage audit passed flag does not match non-compensable gates"
            )

    @property
    def audit_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "split_root": self.split_root,
            "grouping_policy_id": self.grouping_policy_id,
            "train_group_count": self.train_group_count,
            "development_group_count": self.development_group_count,
            "holdout_group_count": self.holdout_group_count,
            "adversarial_group_count": self.adversarial_group_count,
            "cross_partition_group_count": self.cross_partition_group_count,
            "duplicate_example_count": self.duplicate_example_count,
            "hidden_test_commitment": self.hidden_test_commitment,
            "hidden_test_bodies_accessed": self.hidden_test_bodies_accessed,
            "passed": self.passed,
        }
        if include_id:
            result["audit_id"] = self.audit_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LeakageAudit:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"audit_id"},
            noun="leakage audit",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            split_root=str(payload.get("split_root") or ""),
            grouping_policy_id=str(payload.get("grouping_policy_id") or ""),
            train_group_count=payload.get("train_group_count"),
            development_group_count=payload.get("development_group_count"),
            holdout_group_count=payload.get("holdout_group_count"),
            adversarial_group_count=payload.get("adversarial_group_count"),
            cross_partition_group_count=payload.get("cross_partition_group_count"),
            duplicate_example_count=payload.get("duplicate_example_count"),
            hidden_test_commitment=str(payload.get("hidden_test_commitment") or ""),
            hidden_test_bodies_accessed=payload.get("hidden_test_bodies_accessed"),
            passed=payload.get("passed"),
        )
        claimed = str(payload.get("audit_id") or "")
        if claimed and claimed != result.audit_id:
            raise ResidualIntelligenceError("leakage audit identity mismatch")
        return result


def _rights_map(
    value: Mapping[str, Any],
    *,
    sources: tuple[str, ...],
    enum_type: type[Enum],
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be an object")
    if set(value) != set(sources):
        raise ResidualIntelligenceError(f"{name} must bind every exact source identity")
    result: dict[str, str] = {}
    for source in sources:
        try:
            result[source] = str(enum_type(str(value[source])).value)
        except ValueError as exc:
            raise ResidualIntelligenceError(
                f"{name} contains unsupported disposition for {source}"
            ) from exc
    return result


@dataclass(frozen=True)
class TrainingCorpusAdmission:
    """The only contract that may make a corpus eligible for offline training."""

    source_identities: tuple[str, ...]
    source_rights: Mapping[str, str]
    transformation_rights: Mapping[str, str]
    privacy_classification: PrivacyClass
    tenant_scope: str
    data_retention_policy: str
    corpus_root: str
    split_root: str
    holdout_roots: tuple[str, ...]
    deduplication_policy: str
    leakage_audit: LeakageAudit
    tokenizer_identity: str
    compiler_identity: str
    label_producers: tuple[str, ...]
    negative_example_policy: str
    adversarial_partition: str
    environment: str
    admission_decision: TrainingAvailability
    reason_codes: tuple[str, ...]
    schema: str = TRAINING_CORPUS_ADMISSION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "admission_id",
            "source_rights_root",
            "source_identities",
            "source_rights",
            "transformation_rights",
            "privacy_classification",
            "tenant_scope",
            "data_retention_policy",
            "corpus_root",
            "split_root",
            "holdout_roots",
            "deduplication_policy",
            "leakage_audit",
            "tokenizer_identity",
            "compiler_identity",
            "label_producers",
            "negative_example_policy",
            "adversarial_partition",
            "environment",
            "admission_decision",
            "reason_codes",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != TRAINING_CORPUS_ADMISSION_SCHEMA:
            raise ResidualIntelligenceError("unsupported training corpus admission schema")
        sources = text_tuple(
            self.source_identities,
            "source_identities",
            allow_empty=False,
        )
        object.__setattr__(self, "source_identities", sources)
        object.__setattr__(
            self,
            "source_rights",
            _rights_map(
                self.source_rights,
                sources=sources,
                enum_type=SourceRight,
                name="source_rights",
            ),
        )
        object.__setattr__(
            self,
            "transformation_rights",
            _rights_map(
                self.transformation_rights,
                sources=sources,
                enum_type=TransformationRight,
                name="transformation_rights",
            ),
        )
        object.__setattr__(
            self, "privacy_classification", PrivacyClass(self.privacy_classification)
        )
        for field in (
            "tenant_scope",
            "data_retention_policy",
            "corpus_root",
            "split_root",
            "deduplication_policy",
            "tokenizer_identity",
            "compiler_identity",
            "negative_example_policy",
            "adversarial_partition",
            "environment",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        object.__setattr__(
            self,
            "holdout_roots",
            text_tuple(self.holdout_roots, "holdout_roots", allow_empty=False),
        )
        object.__setattr__(
            self,
            "label_producers",
            text_tuple(self.label_producers, "label_producers", allow_empty=False),
        )
        object.__setattr__(self, "reason_codes", text_tuple(self.reason_codes, "reason_codes"))
        object.__setattr__(
            self, "admission_decision", TrainingAvailability(self.admission_decision)
        )
        if not isinstance(self.leakage_audit, LeakageAudit):
            raise ResidualIntelligenceError("leakage_audit must be a typed LeakageAudit")
        if self.leakage_audit.split_root != self.split_root:
            raise ResidualIntelligenceError("leakage audit split root mismatch")
        if self.admission_decision is TrainingAvailability.ADMITTED:
            self._validate_admission()
        elif not self.reason_codes:
            raise ResidualIntelligenceError("training_unavailable requires reason codes")

    def _validate_admission(self) -> None:
        if self.privacy_classification in _NEVER_TRAIN_PRIVACY:
            raise ResidualIntelligenceError(
                f"{self.privacy_classification.value} data can never be training-admitted"
            )
        if self.privacy_classification in {
            PrivacyClass.TENANT_PRIVATE,
            PrivacyClass.MATTER_CONFIDENTIAL,
            PrivacyClass.PERSONAL_DATA,
            PrivacyClass.HEALTH_DATA,
            PrivacyClass.LEGAL_PRIVILEGED,
        } and self.tenant_scope in {"global", "public", "shared"}:
            raise ResidualIntelligenceError(
                "private or regulated data requires a non-global tenant scope"
            )
        if any(
            SourceRight(value) not in _ADMITTED_SOURCE_RIGHTS
            for value in self.source_rights.values()
        ):
            raise ResidualIntelligenceError("all sources require admitted training rights")
        if any(
            TransformationRight(value) not in _ADMITTED_TRANSFORMATION_RIGHTS
            for value in self.transformation_rights.values()
        ):
            raise ResidualIntelligenceError("all sources require admitted transformation rights")
        if not self.leakage_audit.passed:
            raise ResidualIntelligenceError("training admission requires a passing leakage audit")

    @property
    def can_train(self) -> bool:
        return self.admission_decision is TrainingAvailability.ADMITTED

    @property
    def source_rights_root(self) -> str:
        """Non-circular identity examples can bind before split admission."""

        return canonical_id(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/source-rights-root@1",
                "source_identities": list(self.source_identities),
                "source_rights": dict(self.source_rights),
                "transformation_rights": dict(self.transformation_rights),
                "privacy_classification": self.privacy_classification.value,
                "tenant_scope": self.tenant_scope,
                "data_retention_policy": self.data_retention_policy,
            }
        )

    @property
    def admission_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def require_training_admitted(self) -> None:
        if not self.can_train:
            raise ResidualIntelligenceError("training_unavailable")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "source_rights_root": self.source_rights_root,
            "source_identities": list(self.source_identities),
            "source_rights": dict(self.source_rights),
            "transformation_rights": dict(self.transformation_rights),
            "privacy_classification": self.privacy_classification.value,
            "tenant_scope": self.tenant_scope,
            "data_retention_policy": self.data_retention_policy,
            "corpus_root": self.corpus_root,
            "split_root": self.split_root,
            "holdout_roots": list(self.holdout_roots),
            "deduplication_policy": self.deduplication_policy,
            "leakage_audit": self.leakage_audit.to_dict(),
            "tokenizer_identity": self.tokenizer_identity,
            "compiler_identity": self.compiler_identity,
            "label_producers": list(self.label_producers),
            "negative_example_policy": self.negative_example_policy,
            "adversarial_partition": self.adversarial_partition,
            "environment": self.environment,
            "admission_decision": self.admission_decision.value,
            "reason_codes": list(self.reason_codes),
        }
        if include_id:
            result["admission_id"] = self.admission_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingCorpusAdmission:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"admission_id", "source_rights_root"},
            noun="training corpus admission",
        )
        leakage = payload.get("leakage_audit")
        if not isinstance(leakage, Mapping):
            raise ResidualIntelligenceError("leakage_audit must be an object")
        result = cls(
            schema=str(payload.get("schema") or ""),
            source_identities=tuple(payload.get("source_identities") or ()),
            source_rights=payload.get("source_rights") or {},
            transformation_rights=payload.get("transformation_rights") or {},
            privacy_classification=PrivacyClass(str(payload.get("privacy_classification") or "")),
            tenant_scope=str(payload.get("tenant_scope") or ""),
            data_retention_policy=str(payload.get("data_retention_policy") or ""),
            corpus_root=str(payload.get("corpus_root") or ""),
            split_root=str(payload.get("split_root") or ""),
            holdout_roots=tuple(payload.get("holdout_roots") or ()),
            deduplication_policy=str(payload.get("deduplication_policy") or ""),
            leakage_audit=LeakageAudit.from_dict(leakage),
            tokenizer_identity=str(payload.get("tokenizer_identity") or ""),
            compiler_identity=str(payload.get("compiler_identity") or ""),
            label_producers=tuple(payload.get("label_producers") or ()),
            negative_example_policy=str(payload.get("negative_example_policy") or ""),
            adversarial_partition=str(payload.get("adversarial_partition") or ""),
            environment=str(payload.get("environment") or ""),
            admission_decision=TrainingAvailability(str(payload.get("admission_decision") or "")),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("admission_id") or "")
        if claimed and claimed != result.admission_id:
            raise ResidualIntelligenceError("training corpus admission identity mismatch")
        claimed_rights = str(payload.get("source_rights_root") or "")
        if claimed_rights and claimed_rights != result.source_rights_root:
            raise ResidualIntelligenceError("source rights root identity mismatch")
        return result


def source_rights_identity(
    *,
    source_identities: tuple[str, ...],
    source_rights: Mapping[str, str],
    transformation_rights: Mapping[str, str],
    privacy_classification: PrivacyClass | str,
    tenant_scope: str,
    data_retention_policy: str,
) -> str:
    """Build the exact source-rights identity without a split/admission cycle."""

    sources = text_tuple(source_identities, "source_identities", allow_empty=False)
    rights = _rights_map(
        source_rights,
        sources=sources,
        enum_type=SourceRight,
        name="source_rights",
    )
    transforms = _rights_map(
        transformation_rights,
        sources=sources,
        enum_type=TransformationRight,
        name="transformation_rights",
    )
    return canonical_id(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/source-rights-root@1",
            "source_identities": list(sources),
            "source_rights": rights,
            "transformation_rights": transforms,
            "privacy_classification": PrivacyClass(privacy_classification).value,
            "tenant_scope": required_text(tenant_scope, "tenant_scope"),
            "data_retention_policy": required_text(data_retention_policy, "data_retention_policy"),
        }
    )


__all__ = (
    "LEAKAGE_AUDIT_SCHEMA",
    "TRAINING_CORPUS_ADMISSION_SCHEMA",
    "LeakageAudit",
    "SourceRight",
    "TrainingCorpusAdmission",
    "TransformationRight",
    "source_rights_identity",
)
