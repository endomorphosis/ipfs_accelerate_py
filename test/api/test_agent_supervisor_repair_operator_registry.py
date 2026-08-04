"""Contract tests for the reviewed repair-operator registry (PDR-050)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    ANALYTICAL_TRANSFORM_CAPABILITY_INTERFACE,
    ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS,
    analytical_transform_operator_bindings,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE,
    REPAIR_OPERATOR_REGISTRY_INTERFACE,
    DoctorRepairOperatorSpec,
    OperatorValueRequirement,
    RepairBehaviorClass,
    RepairOperatorCapability,
    RepairOperatorKind,
    RepairOperatorLookupDisposition,
    RepairOperatorLookupReason,
    RepairOperatorLookupRequest,
    RepairOperatorLookupResult,
    RepairOperatorRegistry,
    RepairOperatorRegistryAuthorityError,
    RepairOperatorRegistryError,
    ReviewedRepairHook,
    UnknownRepairOperatorError,
    build_default_repair_operator_registry,
    default_repair_operator_registry_id,
    normalize_repair_operator_kind,
)


def roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        file_root_id="file-root:fixture",
        ast_root_id="ast:fixture",
        graph_id="graph:fixture",
        corpus_id="corpus:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        cache_id="cache:fixture",
        operator_registry_id="operators:fixture",
        translator_id="translator:fixture",
        solver_id="solver:fixture",
        kernel_id="kernel:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        sandbox_id="sandbox:fixture",
        environment_id="environment:fixture",
        lease_id="lease:fixture",
    )


def registry() -> RepairOperatorRegistry:
    return build_default_repair_operator_registry()


def request(
    kind: RepairOperatorKind | str,
    *,
    target_paths: tuple[str, ...] = ("pkg/module.py",),
    placement_refs: tuple[str, ...] = ("placement:exact",),
    value_refs: tuple[str, ...] | None = None,
    behavior_classes: tuple[str, ...] = (RepairBehaviorClass.PURE_LOCAL.value,),
    dependency_paths: tuple[str, ...] = (),
    review_refs: tuple[str, ...] | None = None,
    capability_refs: tuple[str, ...] | None = None,
    proof_refs: tuple[str, ...] = ("proof:nomination",),
    repository_id: str = "repository:fixture",
    tree_id: str = "tree:fixture",
    requested_write_paths: tuple[str, ...] = (),
    language: str = "python",
    ast_shape: str = "",
) -> RepairOperatorLookupRequest:
    reg = registry()
    try:
        spec = reg.get(kind)
    except UnknownRepairOperatorError:
        spec = None
    if value_refs is None:
        value_refs = ("value:unique",) if spec and spec.requires_value else ()
    if review_refs is None:
        review_refs = ("review:admitted",) if spec and spec.review_requirement_refs else ()
    if capability_refs is None:
        capability_refs = spec.capability_refs if spec else ()
    return RepairOperatorLookupRequest(
        operator_kind=str(getattr(kind, "value", kind)),
        repository_id=repository_id,
        tree_id=tree_id,
        target_paths=target_paths,
        placement_refs=placement_refs,
        value_refs=value_refs,
        capability_refs=capability_refs,
        proof_refs=proof_refs,
        review_refs=review_refs,
        behavior_classes=behavior_classes,
        dependency_paths=dependency_paths,
        requested_write_paths=requested_write_paths,
        language=language,
        ast_shape=ast_shape,
    )


def test_interfaces_identity_and_round_trip_are_canonical() -> None:
    reg = registry()
    rebuilt = build_default_repair_operator_registry()
    assert reg.INTERFACE == REPAIR_OPERATOR_REGISTRY_INTERFACE
    assert REPAIR_OPERATOR_REGISTRY_INTERFACE == "RepairOperatorRegistry@1"
    assert DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE == "DoctorRepairOperatorSpec@2"
    assert reg.registry_id == rebuilt.registry_id
    assert reg.registry_id == default_repair_operator_registry_id()
    assert reg.registry_id == default_repair_operator_registry_id(roots())
    assert RepairOperatorRegistry.from_dict(reg.to_dict()).registry_id == reg.registry_id
    assert tuple(item.operator_id for item in reg.operators) == tuple(
        sorted(item.operator_id for item in reg.operators)
    )


def test_registry_covers_exact_and_all_analytical_transform_families() -> None:
    reg = registry()
    kinds = set(reg.kinds())
    assert {
        RepairOperatorKind.EXACT_RENAME,
        RepairOperatorKind.EXACT_MOVE,
        RepairOperatorKind.ADD_IMPORT,
        RepairOperatorKind.ADD_EXPORT,
        RepairOperatorKind.ADD_REGISTRATION,
        RepairOperatorKind.ADD_ARGUMENT,
        RepairOperatorKind.THREAD_ARGUMENT,
        RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        RepairOperatorKind.ADD_FACTORY_ROUTE,
        RepairOperatorKind.FINITE_ADAPTER,
        RepairOperatorKind.SCHEMA_PROJECTION,
        RepairOperatorKind.SERIALIZER_UPDATE,
        RepairOperatorKind.FIXTURE_UPDATE,
        RepairOperatorKind.MANIFEST_UPDATE,
        RepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
        RepairOperatorKind.SEMANTIC_PATCH,
        RepairOperatorKind.EQUALITY_REWRITE,
    }.issubset(kinds)
    analytical = {
        TransformKind(item.analytical_transform_kind)
        for item in reg.operators
        if item.analytical_transform_kind
    }
    assert analytical == set(TransformKind)


def test_deterministic_transform_inventory_is_exhaustive_and_descriptor_only() -> None:
    assert (
        ANALYTICAL_TRANSFORM_CAPABILITY_INTERFACE
        == "DeterministicDoctorAnalyticalTransformCapabilities@1"
    )
    assert set(ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS) == set(TransformKind)
    serialized = analytical_transform_operator_bindings()
    assert tuple(item[0] for item in serialized) == tuple(
        sorted(item.value for item in TransformKind)
    )
    assert all(isinstance(kinds, tuple) and kinds for _, kinds in serialized)
    assert not any(callable(value) for value in ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS.values())


def test_every_operator_is_idempotent_scope_bounded_and_capability_declared() -> None:
    base = {
        RepairOperatorCapability.EXACT_TARGET.value,
        RepairOperatorCapability.EXACT_PLACEMENT.value,
        RepairOperatorCapability.CLOSED_AST.value,
        RepairOperatorCapability.IDEMPOTENT_RENDER.value,
        RepairOperatorCapability.SCOPE_BOUND.value,
        RepairOperatorCapability.PROPOSAL_ONLY.value,
    }
    for spec in registry().operators:
        assert isinstance(spec, DoctorRepairOperatorSpec)
        assert spec.idempotent is True
        assert spec.proposal_only is True
        assert spec.inverse_or_compensation_ref
        assert spec.scope_constraints
        assert spec.precondition_refs
        assert spec.postcondition_refs
        assert spec.frame_condition_refs
        assert spec.proof_requirement_refs
        assert spec.validation_requirement_refs
        assert {
            RepairBehaviorClass.DYNAMIC.value,
            RepairBehaviorClass.GENERATED.value,
            RepairBehaviorClass.STATEFUL.value,
            RepairBehaviorClass.NATIVE.value,
            RepairBehaviorClass.PUBLIC_API.value,
            RepairBehaviorClass.DEPENDENCY_CHANGING.value,
        } == set(spec.approval_classes)
        assert spec.abstain_classes == (RepairBehaviorClass.UNKNOWN.value,)
        assert base.issubset(spec.capability_refs)
        assert spec.semantic_authority is False
        assert spec.grants_proof_authority is False
        assert spec.grants_write_authority is False
        restored = DoctorRepairOperatorSpec.from_dict(spec.to_dict())
        assert restored.content_id == spec.content_id


def test_semantic_patch_equality_and_move_are_reviewed_non_executable_hooks() -> None:
    reg = registry()
    expected = {
        RepairOperatorKind.EXACT_MOVE: ReviewedRepairHook.EXACT_MOVE,
        RepairOperatorKind.SEMANTIC_PATCH: ReviewedRepairHook.SEMANTIC_PATCH,
        RepairOperatorKind.EQUALITY_REWRITE: ReviewedRepairHook.EQUALITY_REWRITE,
    }
    for kind, hook in expected.items():
        spec = reg.get(kind)
        assert spec.reviewed_hook is hook
        assert spec.review_requirement_refs
        assert not spec.analytical_transform_kind
        assert spec.renderer_id.startswith("reviewed-repair-hook:")
        assert spec.proposal_only
    equality = reg.get(RepairOperatorKind.EQUALITY_REWRITE)
    assert (
        RepairOperatorCapability.DECLARED_EQUALITY_THEORY.value
        in equality.capability_refs
    )


@pytest.mark.parametrize(
    ("alias", "kind"),
    (
        ("missing_argument", RepairOperatorKind.ADD_ARGUMENT),
        ("value_threading", RepairOperatorKind.THREAD_ARGUMENT),
        ("exact_import", RepairOperatorKind.ADD_IMPORT),
        ("exact_export", RepairOperatorKind.ADD_EXPORT),
        ("exact_registration", RepairOperatorKind.ADD_REGISTRATION),
        ("constructor", RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE),
        ("factory", RepairOperatorKind.ADD_FACTORY_ROUTE),
        ("adapter", RepairOperatorKind.FINITE_ADAPTER),
        ("schema", RepairOperatorKind.SCHEMA_PROJECTION),
        ("serializer", RepairOperatorKind.SERIALIZER_UPDATE),
        ("manifest", RepairOperatorKind.MANIFEST_UPDATE),
        ("reviewed_semantic_patch", RepairOperatorKind.SEMANTIC_PATCH),
    ),
)
def test_aliases_normalize_to_one_canonical_operator(
    alias: str,
    kind: RepairOperatorKind,
) -> None:
    reg = registry()
    assert reg.get(alias).kind is kind
    if alias != "reviewed_semantic_patch":
        assert normalize_repair_operator_kind(alias) is kind


def test_unknown_operator_abstains_and_direct_metadata_lookup_fails_closed() -> None:
    reg = registry()
    with pytest.raises(UnknownRepairOperatorError, match="unknown_operator"):
        reg.lookup("arbitrary_runtime_code")
    result = reg.resolve(request("arbitrary_runtime_code"))
    assert result.disposition is RepairOperatorLookupDisposition.ABSTAINED
    assert result.reason_codes == (RepairOperatorLookupReason.UNKNOWN_OPERATOR.value,)
    assert result.operator_id == ""
    assert not result.admitted


@pytest.mark.parametrize(
    ("field", "values", "reason"),
    (
        (
            "target_paths",
            ("pkg/a.py", "pkg/b.py"),
            RepairOperatorLookupReason.TARGET_AMBIGUOUS,
        ),
        (
            "value_refs",
            ("value:a", "value:b"),
            RepairOperatorLookupReason.VALUE_AMBIGUOUS,
        ),
        (
            "placement_refs",
            ("placement:a", "placement:b"),
            RepairOperatorLookupReason.PLACEMENT_AMBIGUOUS,
        ),
    ),
)
def test_target_value_and_placement_ambiguity_rejects(
    field: str,
    values: tuple[str, ...],
    reason: RepairOperatorLookupReason,
) -> None:
    base = request(RepairOperatorKind.ADD_ARGUMENT)
    result = registry().resolve(replace(base, **{field: values}))
    assert result.disposition is RepairOperatorLookupDisposition.REJECTED
    assert reason.value in result.reason_codes
    assert not result.proposal_eligible


def test_missing_target_value_placement_scope_capability_and_proof_abstain() -> None:
    spec = registry().get(RepairOperatorKind.ADD_ARGUMENT)
    incomplete = request(
        RepairOperatorKind.ADD_ARGUMENT,
        target_paths=(),
        placement_refs=(),
        value_refs=(),
        capability_refs=tuple(item for item in spec.capability_refs if item != "unique_value"),
        proof_refs=(),
        repository_id="",
        tree_id="",
    )
    result = registry().resolve(incomplete)
    assert result.disposition is RepairOperatorLookupDisposition.ABSTAINED
    assert {
        RepairOperatorLookupReason.TARGET_MISSING.value,
        RepairOperatorLookupReason.VALUE_MISSING.value,
        RepairOperatorLookupReason.PLACEMENT_MISSING.value,
        RepairOperatorLookupReason.SCOPE_MISSING.value,
        RepairOperatorLookupReason.CAPABILITY_MISSING.value,
        RepairOperatorLookupReason.PROOF_REFERENCE_MISSING.value,
    }.issubset(result.reason_codes)


def test_total_mapping_and_reviewed_rule_require_exactly_one_value_reference() -> None:
    reg = registry()
    for kind, requirement in (
        (RepairOperatorKind.SERIALIZER_UPDATE, OperatorValueRequirement.TOTAL_MAPPING),
        (RepairOperatorKind.SEMANTIC_PATCH, OperatorValueRequirement.REVIEWED_RULE),
        (RepairOperatorKind.EQUALITY_REWRITE, OperatorValueRequirement.REVIEWED_RULE),
    ):
        spec = reg.get(kind)
        assert spec.value_requirement is requirement
        missing = reg.resolve(request(kind, value_refs=()))
        assert missing.disposition is RepairOperatorLookupDisposition.ABSTAINED
        assert RepairOperatorLookupReason.VALUE_MISSING.value in missing.reason_codes


def test_reviewed_hooks_require_review_references() -> None:
    result = registry().resolve(
        request(RepairOperatorKind.SEMANTIC_PATCH, review_refs=())
    )
    assert result.disposition is RepairOperatorLookupDisposition.ABSTAINED
    assert RepairOperatorLookupReason.REVIEW_REFERENCE_MISSING.value in result.reason_codes


@pytest.mark.parametrize(
    ("behavior", "reason"),
    (
        (RepairBehaviorClass.DYNAMIC, RepairOperatorLookupReason.DYNAMIC_APPROVAL),
        (RepairBehaviorClass.GENERATED, RepairOperatorLookupReason.GENERATED_APPROVAL),
        (RepairBehaviorClass.STATEFUL, RepairOperatorLookupReason.STATEFUL_APPROVAL),
        (RepairBehaviorClass.NATIVE, RepairOperatorLookupReason.NATIVE_APPROVAL),
        (RepairBehaviorClass.PUBLIC_API, RepairOperatorLookupReason.PUBLIC_API_APPROVAL),
        (
            RepairBehaviorClass.DEPENDENCY_CHANGING,
            RepairOperatorLookupReason.DEPENDENCY_APPROVAL,
        ),
    ),
)
def test_risky_behavior_is_approval_required_but_not_authorized(
    behavior: RepairBehaviorClass,
    reason: RepairOperatorLookupReason,
) -> None:
    result = registry().resolve(
        request(
            RepairOperatorKind.ADD_IMPORT,
            behavior_classes=(behavior.value,),
        )
    )
    assert result.disposition is RepairOperatorLookupDisposition.APPROVAL_REQUIRED
    assert reason.value in result.reason_codes
    assert result.approval_validation_required
    assert result.admitted is False
    assert result.semantic_authority is False
    assert result.grants_proof_authority is False
    assert result.grants_write_authority is False


def test_unknown_behavior_abstains_and_dependency_paths_require_approval() -> None:
    reg = registry()
    unknown = reg.resolve(
        request(RepairOperatorKind.ADD_IMPORT, behavior_classes=("quantum_runtime",))
    )
    assert unknown.disposition is RepairOperatorLookupDisposition.ABSTAINED
    assert RepairOperatorLookupReason.UNKNOWN_BEHAVIOR.value in unknown.reason_codes
    dependency = reg.resolve(
        request(
            RepairOperatorKind.ADD_IMPORT,
            dependency_paths=("external/new_dependency.py",),
        )
    )
    assert dependency.disposition is RepairOperatorLookupDisposition.APPROVAL_REQUIRED
    assert RepairOperatorLookupReason.DEPENDENCY_APPROVAL.value in dependency.reason_codes


def test_scope_escape_unsupported_language_and_shape_abstain() -> None:
    reg = registry()
    result = reg.resolve(
        request(
            RepairOperatorKind.ADD_ARGUMENT,
            requested_write_paths=("pkg/other.py",),
            language="rust",
            ast_shape="unsafe_macro",
        )
    )
    assert result.disposition is RepairOperatorLookupDisposition.ABSTAINED
    assert {
        RepairOperatorLookupReason.SCOPE_ESCAPE.value,
        RepairOperatorLookupReason.UNSUPPORTED_LANGUAGE.value,
        RepairOperatorLookupReason.UNSUPPORTED_AST_SHAPE.value,
    }.issubset(result.reason_codes)


def test_complete_lookup_only_nominates_a_candidate() -> None:
    reg = registry()
    original = request(
        RepairOperatorKind.ADD_ARGUMENT,
        ast_shape="call",
        requested_write_paths=("pkg/module.py",),
    )
    result = reg.resolve(original)
    assert result.disposition is RepairOperatorLookupDisposition.PROPOSAL_ELIGIBLE
    assert result.reason_codes == (RepairOperatorLookupReason.CANDIDATE_ONLY.value,)
    assert result.proposal_eligible
    assert result.proof_verification_required
    assert result.admitted is False
    assert result.proposal_only
    assert not result.semantic_authority
    assert not result.grants_proof_authority
    assert not result.grants_write_authority
    restored_request = RepairOperatorLookupRequest.from_dict(original.to_dict())
    restored_result = RepairOperatorLookupResult.from_dict(result.to_dict())
    assert restored_request.content_id == original.content_id
    assert restored_result.content_id == result.content_id


def test_lookup_result_and_spec_reject_authority_forgery() -> None:
    result = registry().resolve(request(RepairOperatorKind.ADD_ARGUMENT))
    with pytest.raises(RepairOperatorRegistryAuthorityError):
        replace(result, grants_write_authority=True)
    with pytest.raises(RepairOperatorRegistryAuthorityError):
        replace(result, grants_proof_authority=True)
    with pytest.raises(RepairOperatorRegistryAuthorityError):
        replace(registry().get(RepairOperatorKind.ADD_ARGUMENT), semantic_authority=True)


def test_registry_serialization_rejects_authority_and_identity_forgery() -> None:
    payload = registry().to_dict()
    payload["grants_write_authority"] = True
    with pytest.raises(RepairOperatorRegistryAuthorityError):
        RepairOperatorRegistry.from_dict(payload)
    payload = registry().to_dict()
    payload["registry_id"] = "forged:registry"
    with pytest.raises(RepairOperatorRegistryError, match="registry_id mismatch"):
        RepairOperatorRegistry.from_dict(payload)


def test_legacy_adapter_is_root_bound_and_does_not_adapt_unrendered_hooks() -> None:
    reg = registry()
    legacy = reg.build_legacy_registry(roots())
    assert legacy.roots == roots()
    assert reg.legacy_kind(RepairOperatorKind.ADD_ARGUMENT) is DoctorOperatorKind.ADD_ARGUMENT
    assert reg.legacy_kind(RepairOperatorKind.SEMANTIC_PATCH) is None
    assert reg.legacy_kind(RepairOperatorKind.EQUALITY_REWRITE) is None
    assert reg.legacy_kind(RepairOperatorKind.EXACT_MOVE) is None
