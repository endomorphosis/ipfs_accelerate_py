from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    CONTRACT_VERSION,
    MAX_PAYLOAD_DEPTH,
    MAX_RECEIPT_BYTES,
    MAX_REFILL_GOALS,
    MAX_REFILL_TASKS,
    NON_COMPENSABLE_GATES,
    ArtifactBounds,
    AuthorityClass,
    AuthorityClassError,
    ContractBoundsError,
    DetachedReferenceError,
    DisagreementRecord,
    DisagreementResolution,
    EvidenceFreshness,
    EvidenceReference,
    FailureCode,
    ForgedSummaryError,
    OperationCapability,
    PathEscapeError,
    PromotionDecision,
    PromotionGateError,
    PromotionVector,
    RefillEpoch,
    RefillEpochStatus,
    ResultBinding,
    RetryDisposition,
    SemanticDependencyIdentity,
    StageEvent,
    StageEventKind,
    StageReceipt,
    SupervisorV2ContractError,
    SupervisorV2Policy,
    TargetDescriptor,
    TargetKind,
    TypedFailure,
    UncertaintyDisposition,
    UncertaintyRecord,
    UnknownFieldError,
    V2_CONTRACT_INTEGRITY_REQUIREMENT_ID,
    canonical_v2_json_bytes,
    semantic_dependency_set_id,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _dependency(
    key: str = "source-tree", revision: str = "tree:abc"
) -> SemanticDependencyIdentity:
    return SemanticDependencyIdentity(
        namespace="repository",
        key=key,
        revision=revision,
        digest="sha256:" + "a" * 64,
    )


def _binding(
    *,
    tree_id: str = "tree:abc",
    producer_id: str = "producer:local",
    capability_id: str = "capability:analysis",
) -> ResultBinding:
    return ResultBinding(
        repository_id="repository:example",
        tree_id=tree_id,
        objective_id="ASI-G200",
        objective_revision="objective:2",
        task_id="ASI-093",
        task_revision="task:1",
        policy_id="policy:v2",
        policy_revision="policy:v2@1",
        producer_id=producer_id,
        producer_revision=f"{producer_id}@1",
        capability_id=capability_id,
        capability_revision=f"{capability_id}@1",
        environment_id="environment:fixture",
        environment_revision="environment:fixture@1",
        semantic_dependencies=(_dependency(revision=tree_id),),
    )


def _evidence(
    *,
    binding: ResultBinding | None = None,
    authority: AuthorityClass = AuthorityClass.VALIDATION,
    summary: str = "The current-tree validation passed.",
    suffix: str = "validation",
) -> EvidenceReference:
    return EvidenceReference(
        binding=binding or _binding(),
        kind="pytest_receipt",
        authority=authority,
        artifact_uri=f"data/evidence/{suffix}.json",
        artifact_content_id=f"bafy-{suffix}",
        sha256="sha256:" + "b" * 64,
        byte_count=2_048,
        media_type="application/json",
        summary=summary,
        freshness=EvidenceFreshness.FRESH,
    )


def _target(tree_id: str = "tree:abc") -> TargetDescriptor:
    return TargetDescriptor(
        repository_id="repository:example",
        tree_id=tree_id,
        state_revision="state:7",
        kind=TargetKind.REPOSITORY_AND_STATE,
        repository_root="/srv/work/repository",
        state_root="/srv/work/state",
        relative_paths=("test/api/test_agent_supervisor_v2_contracts.py",),
    )


def _passing_vector(binding: ResultBinding | None = None) -> PromotionVector:
    return PromotionVector(
        binding=binding or _binding(),
        safety_gates={name: True for name in NON_COMPENSABLE_GATES},
        metrics_millionths={
            "token_reduction": 400_000,
            "warm_cache_reuse": 800_000,
        },
        decision=PromotionDecision.PROVISIONAL,
        composite_score_millionths=700_000,
    )


def test_all_contracts_are_versioned_immutable_and_round_trip() -> None:
    assert V2_CONTRACT_INTEGRITY_REQUIREMENT_ID.isdecimal()
    binding = _binding()
    evidence = _evidence(binding=binding)
    completed = StageEvent(
        binding=binding,
        stage="validation",
        attempt=1,
        sequence=0,
        kind=StageEventKind.COMPLETED,
        authority=AuthorityClass.VALIDATION,
        occurred_at=NOW,
        evidence_references=(evidence,),
    )
    receipt = StageReceipt(
        binding=binding,
        stage="validation",
        attempt=1,
        authority=AuthorityClass.VALIDATION,
        events=(completed,),
        summary="Validation completed against the current tree.",
    )
    policy = SupervisorV2Policy("policy:v2", "policy:v2@1")
    capability = OperationCapability(
        operation="status",
        capability_id="capability:status",
        capability_revision="capability:status@2",
        authority=AuthorityClass.DIAGNOSTIC,
        request_schema="schema:status-request@2",
        result_schema="schema:status-result@2",
        target_kinds=(TargetKind.STATE,),
    )
    uncertainty = UncertaintyRecord(
        binding=binding,
        subject="cache speedup",
        statement="The exact speedup remains fixture-dependent.",
        disposition=UncertaintyDisposition.OPEN,
        probability_lower_millionths=200_000,
        probability_upper_millionths=800_000,
        evidence_references=(evidence,),
    )
    failure = TypedFailure(
        binding=binding,
        code=FailureCode.VALIDATION_FAILED,
        authority=AuthorityClass.VALIDATION,
        retry=RetryDisposition.NEW_BINDING,
        reason_code="test_failed",
        public_message="A bounded validation command failed.",
        occurred_at=NOW,
        evidence_references=(evidence,),
    )
    epoch = RefillEpoch(
        binding=binding,
        target=_target(),
        board_revision="board:1",
        operation_catalog_id="catalog:2",
        artifact_store_policy_id="artifact-policy:2",
        observation_window_start=NOW,
        observation_window_end=NOW + timedelta(hours=1),
        status=RefillEpochStatus.PROPOSED,
        successor_goal_ids=("ASI-G210",),
        successor_task_ids=("ASI-094",),
        trigger_dependency_ids=binding.semantic_dependency_ids,
        promotion_vector=_passing_vector(binding),
    )

    values = (
        _dependency(),
        binding,
        ArtifactBounds(),
        evidence,
        completed,
        receipt,
        capability,
        uncertainty,
        _passing_vector(binding),
        policy,
        _target(),
        failure,
        epoch,
    )
    for value in values:
        encoded = value.to_json()
        restored = type(value).from_json(encoded)
        assert restored == value
        assert restored.content_id == value.content_id
        assert json.loads(encoded)["contract_version"] == CONTRACT_VERSION

    with pytest.raises(FrozenInstanceError):
        binding.tree_id = "tree:forged"  # type: ignore[misc]
    with pytest.raises(TypeError):
        _passing_vector().safety_gates["safety"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        _passing_vector().metrics_millionths["token_reduction"] = 0  # type: ignore[index]


def test_binding_freezes_every_required_identity_and_dependency_population() -> None:
    binding = _binding()
    payload = binding.to_dict()

    assert {
        "repository_id",
        "tree_id",
        "objective_id",
        "objective_revision",
        "task_id",
        "task_revision",
        "policy_id",
        "policy_revision",
        "producer_id",
        "producer_revision",
        "capability_id",
        "capability_revision",
        "environment_id",
        "environment_revision",
        "semantic_dependencies",
    }.issubset(payload)
    assert binding.semantic_dependency_ids == (
        binding.semantic_dependencies[0].dependency_id,
    )

    payload["task_id"] = ""
    with pytest.raises(SupervisorV2ContractError, match="task_id"):
        ResultBinding.from_dict(payload)

    with pytest.raises(SupervisorV2ContractError, match="namespace/key"):
        ResultBinding(
            **{
                **{
                    name: getattr(binding, name)
                    for name in (
                        "repository_id",
                        "tree_id",
                        "objective_id",
                        "objective_revision",
                        "task_id",
                        "task_revision",
                        "policy_id",
                        "policy_revision",
                        "producer_id",
                        "producer_revision",
                        "capability_id",
                        "capability_revision",
                        "environment_id",
                        "environment_revision",
                    )
                },
                "semantic_dependencies": (
                    _dependency(revision="one"),
                    _dependency(revision="two"),
                ),
            }
        )


def test_semantic_dependency_identity_is_order_independent_and_revision_sensitive() -> None:
    first = _dependency("a", "rev:1")
    second = SemanticDependencyIdentity(
        namespace="policy",
        key="promotion",
        revision="rev:2",
        digest="sha256:" + "c" * 64,
    )

    assert semantic_dependency_set_id((first, second)) == semantic_dependency_set_id(
        (second, first)
    )
    assert semantic_dependency_set_id((first,)) != semantic_dependency_set_id(
        (_dependency("a", "rev:changed"),)
    )


def test_closed_decoders_reject_unknown_fields_and_forged_identities() -> None:
    payload = _binding().to_dict()
    payload["provider_confidence_override"] = True
    with pytest.raises(UnknownFieldError, match="unsupported fields"):
        ResultBinding.from_dict(payload)

    payload = _dependency().to_record()
    payload["dependency_id"] = "forged"
    with pytest.raises(ForgedSummaryError, match="identity"):
        SemanticDependencyIdentity.from_dict(payload)

    payload = _dependency().to_dict()
    payload["schema_version"] = CONTRACT_VERSION + 1
    with pytest.raises(SupervisorV2ContractError, match="version"):
        SemanticDependencyIdentity.from_dict(payload)

    with pytest.raises(SupervisorV2ContractError, match="object"):
        SemanticDependencyIdentity.from_dict([])  # type: ignore[arg-type]


def test_evidence_summary_and_derived_completion_claim_cannot_be_forged() -> None:
    reference = _evidence()
    payload = reference.to_record()
    payload["summary"] = "A forged passing summary."
    with pytest.raises(ForgedSummaryError, match="summary"):
        EvidenceReference.from_dict(payload)

    payload = reference.to_dict()
    payload["completion_authoritative"] = True
    with pytest.raises(ForgedSummaryError, match="derived"):
        EvidenceReference.from_dict(payload)

    completion = _evidence(
        authority=AuthorityClass.COMPLETION, suffix="completion"
    )
    assert completion.completion_authoritative is True
    stale = completion.to_dict()
    stale["freshness"] = EvidenceFreshness.STALE.value
    restored = EvidenceReference.from_dict(stale)
    assert restored.completion_authoritative is False

    with pytest.raises(PathEscapeError, match="unsafe content path"):
        EvidenceReference(
            binding=_binding(),
            kind="unsafe",
            authority=AuthorityClass.DIAGNOSTIC,
            artifact_uri="cas://bafy-safe/../../etc/passwd",
            artifact_content_id="bafy-safe",
            sha256="sha256:" + "b" * 64,
            byte_count=1,
            media_type="application/json",
            summary="Unsafe content path.",
        )


def test_stage_receipt_rejects_detached_references_and_noncontiguous_history() -> None:
    binding = _binding()
    foreign = _evidence(binding=_binding(tree_id="tree:foreign"))
    with pytest.raises(DetachedReferenceError, match="detached"):
        StageEvent(
            binding=binding,
            stage="validation",
            attempt=1,
            sequence=0,
            kind=StageEventKind.COMPLETED,
            authority=AuthorityClass.VALIDATION,
            occurred_at=NOW,
            evidence_references=(foreign,),
        )

    events = (
        StageEvent(
            binding=binding,
            stage="analysis",
            attempt=1,
            sequence=0,
            kind=StageEventKind.STARTED,
            authority=AuthorityClass.DIAGNOSTIC,
            occurred_at=NOW,
        ),
        StageEvent(
            binding=binding,
            stage="analysis",
            attempt=1,
            sequence=2,
            kind=StageEventKind.COMPLETED,
            authority=AuthorityClass.DIAGNOSTIC,
            occurred_at=NOW,
        ),
    )
    with pytest.raises(SupervisorV2ContractError, match="contiguous"):
        StageReceipt(
            binding=binding,
            stage="analysis",
            attempt=1,
            authority=AuthorityClass.DIAGNOSTIC,
            events=events,
            summary="Skipped an event.",
        )

    completed = StageEvent(
        binding=binding,
        stage="analysis",
        attempt=1,
        sequence=0,
        kind=StageEventKind.COMPLETED,
        authority=AuthorityClass.DIAGNOSTIC,
        occurred_at=NOW,
    )
    progressed_after_completion = StageEvent(
        binding=binding,
        stage="analysis",
        attempt=1,
        sequence=1,
        kind=StageEventKind.PROGRESSED,
        authority=AuthorityClass.DIAGNOSTIC,
        occurred_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(SupervisorV2ContractError, match="after a terminal"):
        StageReceipt(
            binding=binding,
            stage="analysis",
            attempt=1,
            authority=AuthorityClass.DIAGNOSTIC,
            events=(completed, progressed_after_completion),
            summary="Continued after completion.",
        )


def test_authority_classes_are_distinct_and_cannot_be_mixed_in_a_receipt() -> None:
    assert {item.value for item in AuthorityClass} == {
        "diagnostic",
        "proposal",
        "validation",
        "proof",
        "merge",
        "mutation",
        "completion",
    }
    binding = _binding()
    validation = StageEvent(
        binding=binding,
        stage="check",
        attempt=1,
        sequence=0,
        kind=StageEventKind.COMPLETED,
        authority=AuthorityClass.VALIDATION,
        occurred_at=NOW,
    )
    with pytest.raises(AuthorityClassError, match="authority"):
        StageReceipt(
            binding=binding,
            stage="check",
            attempt=1,
            authority=AuthorityClass.COMPLETION,
            events=(validation,),
            summary="Cannot upgrade validation into completion.",
        )


def test_mutation_capability_requires_all_security_semantics() -> None:
    values = {
        "operation": "objective_refine",
        "capability_id": "capability:refine",
        "capability_revision": "capability:refine@2",
        "authority": AuthorityClass.MUTATION,
        "request_schema": "schema:refine-request@2",
        "result_schema": "schema:refine-result@2",
        "target_kinds": (TargetKind.REPOSITORY_AND_STATE,),
        "allowed_roots": ("/srv/work/repository",),
        "supports_dry_run": True,
        "requires_idempotency": True,
        "requires_authorization": True,
        "requires_lease": True,
        "requires_fencing": True,
    }
    capability = OperationCapability(**values)
    assert OperationCapability.from_json(capability.to_json()) == capability

    for field in (
        "supports_dry_run",
        "requires_idempotency",
        "requires_authorization",
        "requires_lease",
        "requires_fencing",
    ):
        with pytest.raises(AuthorityClassError, match="mutation"):
            OperationCapability(**{**values, field: False})


@pytest.mark.parametrize(
    "path",
    (
        "../secret",
        "src/../../secret",
        "/etc/passwd",
        "C:/Windows/System32/config",
        "src//module.py",
        "src/./module.py",
        r"src\..\secret",
    ),
)
def test_target_descriptors_reject_path_escapes(path: str) -> None:
    with pytest.raises(PathEscapeError):
        TargetDescriptor(
            repository_id="repository:example",
            tree_id="tree:abc",
            state_revision="state:1",
            kind=TargetKind.REPOSITORY,
            repository_root="/srv/work/repository",
            state_root="/srv/work/state",
            relative_paths=(path,),
        )


def test_artifact_bounds_reject_over_depth_and_over_byte_payloads() -> None:
    bounds = ArtifactBounds(
        max_receipt_bytes=256,
        max_projection_bytes=512,
        max_reference_bytes=128,
        max_text_bytes=64,
        max_depth=4,
        max_references=4,
    )
    nested: object = "leaf"
    for _ in range(MAX_PAYLOAD_DEPTH):
        nested = {"next": nested}
    with pytest.raises(ContractBoundsError, match="depth"):
        bounds.validate(nested)
    with pytest.raises(ContractBoundsError, match="byte"):
        bounds.validate({"body": "x" * 1_000})
    with pytest.raises(ContractBoundsError):
        ArtifactBounds(max_receipt_bytes=MAX_RECEIPT_BYTES + 1)


def test_canonical_helper_rejects_projection_overflow() -> None:
    assert canonical_v2_json_bytes({"small": True}) == b'{"small":true}'
    with pytest.raises(ContractBoundsError, match="bytes"):
        canonical_v2_json_bytes({"body": "x" * 1_100_000})


def test_uncertainty_and_disagreement_retain_provenance() -> None:
    aggregate = _binding(producer_id="producer:resolver")
    local = _evidence(
        binding=_binding(producer_id="producer:local"),
        authority=AuthorityClass.PROPOSAL,
        suffix="local",
    )
    remote = _evidence(
        binding=_binding(
            producer_id="producer:datasets",
            capability_id="capability:remote-analysis",
        ),
        authority=AuthorityClass.PROPOSAL,
        suffix="remote",
    )
    resolver = _evidence(
        binding=aggregate,
        authority=AuthorityClass.VALIDATION,
        suffix="resolver",
    )
    disagreement = DisagreementRecord(
        binding=aggregate,
        subject="affected symbols",
        claims=(local, remote),
        resolution=DisagreementResolution.INDEPENDENT_VALIDATION,
        selected_reference_id=local.reference_id,
        resolver_reference=resolver,
    )
    assert {claim.binding.producer_id for claim in disagreement.claims} == {
        "producer:local",
        "producer:datasets",
    }
    assert DisagreementRecord.from_json(disagreement.to_json()) == disagreement

    with pytest.raises(SupervisorV2ContractError, match="independent producers"):
        DisagreementRecord(
            binding=aggregate,
            subject="not independent",
            claims=(local, _evidence(binding=local.binding, suffix="local-2")),
        )

    with pytest.raises(SupervisorV2ContractError, match="producer independent"):
        DisagreementRecord(
            binding=aggregate,
            subject="resolver is not independent",
            claims=(local, remote),
            resolution=DisagreementResolution.INDEPENDENT_VALIDATION,
            selected_reference_id=local.reference_id,
            resolver_reference=_evidence(
                binding=_binding(producer_id="producer:local"),
                authority=AuthorityClass.VALIDATION,
                suffix="non-independent-resolver",
            ),
        )


def test_failed_safety_gate_cannot_be_hidden_by_composite_score() -> None:
    failed = {name: True for name in NON_COMPENSABLE_GATES}
    failed["safety"] = False

    with pytest.raises(PromotionGateError, match="non-compensable"):
        PromotionVector(
            binding=_binding(),
            safety_gates=failed,
            metrics_millionths={"throughput_gain": 1_000_000},
            decision=PromotionDecision.PROMOTE,
            composite_score_millionths=1_000_000,
        )
    with pytest.raises(PromotionGateError, match="non-compensable"):
        PromotionVector(
            binding=_binding(),
            safety_gates=failed,
            metrics_millionths={"throughput_gain": 1_000_000},
            decision=PromotionDecision.SHADOW,
            composite_score_millionths=1,
        )

    shadow = PromotionVector(
        binding=_binding(),
        safety_gates=failed,
        metrics_millionths={"throughput_gain": 1_000_000},
        decision=PromotionDecision.SHADOW,
        composite_score_millionths=0,
    )
    assert shadow.hard_gates_pass is False
    assert shadow.promotion_eligible is False


def test_refill_epoch_is_bounded_target_bound_and_dependency_bound() -> None:
    binding = _binding()
    values = {
        "binding": binding,
        "target": _target(),
        "board_revision": "board:1",
        "operation_catalog_id": "catalog:2",
        "artifact_store_policy_id": "artifacts:2",
        "observation_window_start": NOW,
        "observation_window_end": NOW + timedelta(hours=1),
        "status": RefillEpochStatus.PROPOSED,
        "trigger_dependency_ids": binding.semantic_dependency_ids,
    }
    with pytest.raises(ContractBoundsError, match="successor_goal_ids"):
        RefillEpoch(
            **values,
            successor_goal_ids=tuple(
                f"goal:{index}" for index in range(MAX_REFILL_GOALS + 1)
            ),
        )
    with pytest.raises(ContractBoundsError, match="successor_task_ids"):
        RefillEpoch(
            **values,
            successor_task_ids=tuple(
                f"task:{index}" for index in range(MAX_REFILL_TASKS + 1)
            ),
        )
    with pytest.raises(DetachedReferenceError, match="trigger"):
        RefillEpoch(
            **{**values, "trigger_dependency_ids": ("dependency:forged",)},
            successor_goal_ids=("goal:one",),
        )
    with pytest.raises(DetachedReferenceError, match="target"):
        RefillEpoch(
            **{**values, "target": _target("tree:foreign")},
            successor_goal_ids=("goal:one",),
        )


def test_refill_materialization_requires_passing_promotion() -> None:
    binding = _binding()
    common = {
        "binding": binding,
        "target": _target(),
        "board_revision": "board:1",
        "operation_catalog_id": "catalog:2",
        "artifact_store_policy_id": "artifacts:2",
        "observation_window_start": NOW,
        "observation_window_end": NOW + timedelta(hours=1),
        "status": RefillEpochStatus.MATERIALIZED,
        "successor_goal_ids": ("goal:one",),
        "trigger_dependency_ids": binding.semantic_dependency_ids,
    }
    with pytest.raises(PromotionGateError, match="promotion"):
        RefillEpoch(**common)
    epoch = RefillEpoch(**common, promotion_vector=_passing_vector(binding))
    assert epoch.status is RefillEpochStatus.MATERIALIZED


def test_typed_failures_are_bounded_redacted_and_binding_checked() -> None:
    with pytest.raises(SupervisorV2ContractError, match="credential"):
        TypedFailure(
            binding=_binding(),
            code=FailureCode.INTERNAL_ERROR,
            authority=AuthorityClass.DIAGNOSTIC,
            retry=RetryDisposition.NEVER,
            reason_code="redaction_failure",
            public_message="authorization: bearer should-not-leak",
            occurred_at=NOW,
        )
    with pytest.raises(DetachedReferenceError, match="detached"):
        TypedFailure(
            binding=_binding(),
            code=FailureCode.STALE_TREE,
            authority=AuthorityClass.VALIDATION,
            retry=RetryDisposition.NEW_BINDING,
            reason_code="tree_changed",
            public_message="The repository tree changed.",
            occurred_at=NOW,
            evidence_references=(
                _evidence(binding=_binding(tree_id="tree:foreign")),
            ),
        )


def test_policy_cannot_raise_absolute_refill_or_artifact_limits() -> None:
    with pytest.raises(ContractBoundsError, match="refill_max_goals"):
        SupervisorV2Policy(
            "policy:v2",
            "policy:v2@1",
            refill_max_goals=MAX_REFILL_GOALS + 1,
        )
    with pytest.raises(ContractBoundsError, match="refill_max_tasks"):
        SupervisorV2Policy(
            "policy:v2",
            "policy:v2@1",
            refill_max_tasks=MAX_REFILL_TASKS + 1,
        )
