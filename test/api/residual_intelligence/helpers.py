from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PrivacyClass,
    ResidualTaskFamily,
    TrainingAvailability,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.corpus import (
    CorpusSourceKind,
    LabelDisposition,
    ResidualDistillationExample,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.rights import (
    SourceRight,
    TrainingCorpusAdmission,
    TransformationRight,
    source_rights_identity,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.splits import (
    SemanticSplitManifest,
    SemanticSplitPolicy,
    semantic_lineage_split,
)

SOURCE_ID = "source:synthetic:vrif-fixtures-v1"
SOURCE_RIGHTS = {SOURCE_ID: SourceRight.SYNTHETIC_GENERATED.value}
TRANSFORM_RIGHTS = {SOURCE_ID: TransformationRight.TRAINING_AND_DERIVATIVES_PERMITTED.value}


def rights_root() -> str:
    return source_rights_identity(
        source_identities=(SOURCE_ID,),
        source_rights=SOURCE_RIGHTS,
        transformation_rights=TRANSFORM_RIGHTS,
        privacy_classification=PrivacyClass.PUBLIC,
        tenant_scope="public-fixture",
        data_retention_policy="retain-versioned-fixtures",
    )


def example(
    index: int,
    *,
    group: str,
    adversarial: bool = False,
    accepted: bool = True,
) -> ResidualDistillationExample:
    label = LabelDisposition.ACCEPTED if accepted else LabelDisposition.REJECTED
    return ResidualDistillationExample(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        input_feature_identity=f"features:{index}",
        context_identity=f"context:{index}",
        source_identity=SOURCE_ID,
        source_kind=(
            CorpusSourceKind.ADVERSARIAL_MUTANT
            if adversarial
            else CorpusSourceKind.SYNTHETIC_FIXTURE
        ),
        teacher_or_source_producer="mechanical-fixture-generator@1",
        teacher_output={
            "failure_class": "missing_dependency_edge",
            "recommended_action": "expand_context_reference",
        },
        independent_validation=("validator:dependency-graph@1",) if accepted else (),
        label_disposition=label,
        accepted_output=(
            {
                "failure_class": "missing_dependency_edge",
                "recommended_action": "expand_context_reference",
            }
            if accepted
            else {}
        ),
        rejected_alternatives=(({"failure_class": "provider_timeout"},) if not accepted else ()),
        counterexamples=(f"counterexample:{index}",) if not accepted else (),
        proof_test_evidence=("test:dependency-graph:pass",) if accepted else (),
        repository_family="ipfs_accelerate_py",
        language_framework="python-pytest",
        rights_reference=rights_root(),
        privacy_class=PrivacyClass.PUBLIC,
        split_group=group,
        semantic_lineage=(group, f"failure-family:{group}"),
        adversarial=adversarial,
        boundary_case=adversarial,
        hidden_test_derived=False,
    )


def split_fixture() -> tuple[tuple[ResidualDistillationExample, ...], SemanticSplitManifest]:
    examples = (
        example(1, group="train-a"),
        example(2, group="dev-a"),
        example(3, group="holdout-a"),
        example(4, group="adversarial-a", adversarial=True, accepted=False),
    )
    policy = SemanticSplitPolicy(
        policy_id="semantic-lineage-fixture@1",
        seed_identity="seed:vrif-fixture-v1",
        forced_development_groups=("dev-a",),
        forced_held_out_groups=("holdout-a",),
        forced_adversarial_groups=("adversarial-a",),
    )
    manifest = semantic_lineage_split(
        examples,
        policy=policy,
        hidden_test_commitment="sha256:hidden-test-fixture-commitment",
    )
    return examples, manifest


def admission(*, admitted: bool = True) -> tuple[TrainingCorpusAdmission, tuple[Any, ...]]:
    examples, manifest = split_fixture()
    audit = manifest.leakage_audit()
    record = TrainingCorpusAdmission(
        source_identities=(SOURCE_ID,),
        source_rights=SOURCE_RIGHTS,
        transformation_rights=TRANSFORM_RIGHTS,
        privacy_classification=PrivacyClass.PUBLIC,
        tenant_scope="public-fixture",
        data_retention_policy="retain-versioned-fixtures",
        corpus_root="corpus:synthetic:vrif-fixtures-v1",
        split_root=manifest.split_root,
        holdout_roots=("holdout:fixture",),
        deduplication_policy="canonical-example-identity",
        leakage_audit=audit,
        tokenizer_identity="tokenizer:fixture-v1",
        compiler_identity="compiler:structured-ir-v1",
        label_producers=("validator:dependency-graph@1",),
        negative_example_policy="retain-mechanical-and-adversarial-negatives",
        adversarial_partition="adversarial:fixture-v1",
        environment="environment:hermetic-pytest",
        admission_decision=(
            TrainingAvailability.ADMITTED if admitted else TrainingAvailability.TRAINING_UNAVAILABLE
        ),
        reason_codes=() if admitted else ("operator_hold",),
    )
    return record, examples
