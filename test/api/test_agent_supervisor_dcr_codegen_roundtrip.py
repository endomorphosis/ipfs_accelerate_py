"""DCR-047: codegen roundtrip and generated-source synchronization.

Acceptance:
* Two clean generations are byte-identical.
* Stale generated artifacts fail validation.
* Rollback restores the exact prior tree.
* Only pinned deterministic generators may run; generated files name their
  authority source and never overwrite hand-owned code.
* Operators remain proposal-only and never grant write/proof/semantic authority.
"""

from __future__ import annotations

import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.codegen_repairs import (
    CODEGEN_OPERATOR_VECTORS_PATH,
    CODEGEN_REPAIR_EVIDENCE,
    CODEGEN_REPAIR_OPERATORS_INTERFACE,
    GENERATED_ARTIFACT_MANIFEST_INTERFACE,
    PINNED_GENERATORS,
    ArtifactKind,
    ArtifactOwnership,
    AuthoritySource,
    CodegenRepairError,
    CodegenRepairRequest,
    GeneratedArtifact,
    GeneratedArtifactManifest,
    GeneratedTreeSnapshot,
    GeneratorPin,
    GoldenRoundtripValidator,
    OperatorRole,
    RegenerateProjectionOperator,
    RepairDisposition,
    SemanticAuthoritySource,
    apply_manifest_to_tree,
    build_codegen_repair_operators,
    build_semantic_authority_source,
    default_generator_pins,
    generate_projection_manifest,
    materialize_codegen_operator_vectors,
    restore_semantic_from_manifest,
    rollback_tree,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


HAND_OWNED = "src/hand_owned/contract.py"
HAND_OWNED_BODY = "# hand-owned contract surface\n"


def _source(
    *,
    source_id: str = "semantic:dcr047:demo",
    authority: AuthoritySource = AuthoritySource.REVIEWED,
    body: dict | None = None,
    hand_owned_paths: tuple[str, ...] = (HAND_OWNED,),
) -> SemanticAuthoritySource:
    return build_semantic_authority_source(
        source_id=source_id,
        semantic_body=body
        or {
            "interface": "DemoContract@1",
            "fields": ["id", "status"],
            "status_enum": ["ready", "blocked"],
        },
        authority=authority,
        hand_owned_paths=hand_owned_paths,
        source_refs=("source:reviewed-semantic-ir",),
    )


def _prior_tree(manifest: GeneratedArtifactManifest | None = None) -> dict[str, str]:
    tree = {HAND_OWNED: HAND_OWNED_BODY}
    if manifest is not None:
        tree.update(dict(manifest.tree_map()))
    return tree


def _stale_manifest(clean: GeneratedArtifactManifest) -> GeneratedArtifactManifest:
    stale_artifacts = []
    for artifact in clean.artifacts:
        if artifact.kind is ArtifactKind.SCHEMA:
            stale_body = artifact.body + "\n/* stale drift */\n"
            stale_artifacts.append(
                GeneratedArtifact(
                    path=artifact.path,
                    kind=artifact.kind,
                    body=stale_body,
                    content_digest=(
                        "sha256:"
                        + hashlib.sha256(stale_body.encode("utf-8")).hexdigest()
                    ),
                    authority_source_cid=artifact.authority_source_cid,
                    generator=artifact.generator,
                    ownership=ArtifactOwnership.GENERATED,
                    semantic_digest=artifact.semantic_digest,
                )
            )
        else:
            stale_artifacts.append(artifact)
    return GeneratedArtifactManifest(
        manifest_id="manifest:stale",
        authority_source_cid=clean.authority_source_cid,
        artifacts=tuple(stale_artifacts),
        generator_digest=clean.generator_digest,
        generator_args=clean.generator_args,
        hand_owned_paths=clean.hand_owned_paths,
        tree_digest="",
        authority=clean.authority,
    )


# ---------------------------------------------------------------------------
# Interface / registry binding
# ---------------------------------------------------------------------------


def test_interfaces_and_evidence_are_declared() -> None:
    assert CODEGEN_REPAIR_OPERATORS_INTERFACE == "CodegenRepairOperators@1"
    assert GENERATED_ARTIFACT_MANIFEST_INTERFACE == "GeneratedArtifactManifest@1"
    assert CODEGEN_REPAIR_EVIDENCE == "dcr/codegen-roundtrip@1"
    assert (
        CODEGEN_OPERATOR_VECTORS_PATH
        == "data/agent_supervisor/deterministic_contract_repair/operator-vectors/codegen.json"
    )
    ops = build_codegen_repair_operators()
    assert ops.INTERFACE == CODEGEN_REPAIR_OPERATORS_INTERFACE
    assert ops.EVIDENCE_ID == CODEGEN_REPAIR_EVIDENCE
    assert ops.MANIFEST_INTERFACE == GENERATED_ARTIFACT_MANIFEST_INTERFACE
    assert isinstance(ops.regenerate_projection, RegenerateProjectionOperator)
    assert isinstance(ops.golden_roundtrip, GoldenRoundtripValidator)


def test_registry_binds_regenerate_projection_to_codegen_family() -> None:
    reg = build_default_operator_registry()
    descriptor = reg.require_known(OperatorKind.REGENERATE_PROJECTION)
    assert descriptor.family is OperatorFamily.CODEGEN
    assert descriptor.kind is OperatorKind.REGENERATE_PROJECTION
    assert descriptor.proposal_only is True
    assert descriptor.grants_write_authority is False
    assert descriptor.grants_proof_authority is False
    assert descriptor.semantic_authority is False
    assert "scope:closed_generated_projection" in descriptor.write_scope
    assert "scope:closed_generated_manifest" in descriptor.write_scope
    assert reg.get("codegen_roundtrip").kind is OperatorKind.REGENERATE_PROJECTION
    assert reg.get("regenerate_codecs").kind is OperatorKind.REGENERATE_PROJECTION


def test_only_pinned_generators_are_admissible() -> None:
    pins = default_generator_pins()
    assert {pin.generator_id for pin in pins} == set(PINNED_GENERATORS)
    for pin in pins:
        assert pin.generator_digest == PINNED_GENERATORS[pin.generator_id]
    with pytest.raises(CodegenRepairError, match="not a pinned"):
        GeneratorPin(
            generator_id="dcr-codegen/unknown/v1",
            generator_digest="sha256:" + ("a" * 64),
        )
    known_id = next(iter(PINNED_GENERATORS))
    with pytest.raises(CodegenRepairError, match="does not match the pinned"):
        GeneratorPin(
            generator_id=known_id,
            generator_digest="sha256:" + ("0" * 64),
        )


# ---------------------------------------------------------------------------
# Two clean generations are byte-identical
# ---------------------------------------------------------------------------


def test_two_clean_generations_are_byte_identical() -> None:
    source = _source()
    generators = default_generator_pins(args={"profile": "deterministic"})
    first = generate_projection_manifest(
        source, generators=generators, manifest_id="manifest:run-1"
    )
    second = generate_projection_manifest(
        source, generators=generators, manifest_id="manifest:run-2"
    )
    assert dict(first.tree_map()) == dict(second.tree_map())
    assert dict(first.output_hashes()) == dict(second.output_hashes())
    assert first.tree_digest == second.tree_digest
    # Bodies are exact byte strings.
    for path in first.tree_map():
        assert first.tree_map()[path].encode("utf-8") == second.tree_map()[
            path
        ].encode("utf-8")


def test_golden_roundtrip_validator_passes_on_clean_pair() -> None:
    source = _source()
    generators = default_generator_pins()
    clean = generate_projection_manifest(source, generators=generators)
    result = GoldenRoundtripValidator().validate(
        source, current_manifest=clean, generators=generators
    )
    assert result.byte_identical is True
    assert result.semantic_roundtrip_ok is True
    assert result.generation_one_digest == result.generation_two_digest
    assert "two_clean_generations_byte_identical" in result.reason_codes
    assert "source_generated_semantic_roundtrip_ok" in result.reason_codes
    assert "golden_roundtrip_passed" in result.reason_codes
    assert "stale_generated_artifacts" not in result.reason_codes


def test_source_generated_semantic_roundtrip() -> None:
    source = _source()
    manifest = generate_projection_manifest(source)
    restored = restore_semantic_from_manifest(manifest)
    assert restored["source_id"] == source.source_id
    assert restored["semantic_body"] == dict(source.semantic_body)
    assert restored["authority_source_cid"] == source.authority_source_cid
    # Every generated artifact names its authority source.
    for artifact in manifest.artifacts:
        assert artifact.authority_source_cid == source.authority_source_cid
        assert artifact.ownership is ArtifactOwnership.GENERATED


# ---------------------------------------------------------------------------
# Stale generated artifacts fail validation
# ---------------------------------------------------------------------------


def test_stale_generated_artifacts_fail_validation() -> None:
    source = _source()
    generators = default_generator_pins()
    clean = generate_projection_manifest(source, generators=generators)
    stale = _stale_manifest(clean)
    assert dict(stale.tree_map()) != dict(clean.tree_map())

    result = GoldenRoundtripValidator().validate(
        source, current_manifest=stale, generators=generators
    )
    assert "stale_generated_artifacts" in result.reason_codes
    # Dual clean generation still proves byte identity of the generator.
    assert result.byte_identical is True

    with pytest.raises(CodegenRepairError, match="stale generated artifacts"):
        GoldenRoundtripValidator().assert_valid(
            source, current_manifest=stale, generators=generators
        )


def test_operator_validation_failed_on_stale_current_manifest() -> None:
    source = _source()
    generators = default_generator_pins()
    clean = generate_projection_manifest(source, generators=generators)
    stale = _stale_manifest(clean)
    receipt = RegenerateProjectionOperator().apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            current_manifest=stale,
            prior_tree=_prior_tree(stale),
            generators=generators,
            require_roundtrip=True,
        )
    )
    assert receipt.disposition is RepairDisposition.VALIDATION_FAILED
    assert "stale_generated_artifacts" in receipt.reason_codes
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    # Preview still carries the clean regeneration for repair planning.
    assert receipt.preview_manifest is not None
    assert receipt.preview_manifest.tree_digest == clean.tree_digest


# ---------------------------------------------------------------------------
# Rollback restores the exact prior tree
# ---------------------------------------------------------------------------


def test_rollback_restores_exact_prior_tree() -> None:
    source = _source()
    clean = generate_projection_manifest(source)
    prior = _prior_tree()  # hand-owned only
    snapshot = GeneratedTreeSnapshot.capture(
        snapshot_id="snapshot:prior",
        tree=prior,
        hand_owned_paths=source.hand_owned_paths,
    )
    mutated = apply_manifest_to_tree(prior, clean)
    assert mutated != prior
    assert HAND_OWNED in mutated
    assert mutated[HAND_OWNED] == HAND_OWNED_BODY

    restored = rollback_tree(snapshot, current_tree=mutated)
    assert restored == prior
    assert snapshot.exact_equals(restored)
    assert snapshot.tree_digest == GeneratedTreeSnapshot.capture(
        snapshot_id="check", tree=restored
    ).tree_digest


def test_operator_inverse_rollback_restores_exact_prior_tree() -> None:
    source = _source()
    generators = default_generator_pins()
    prior = _prior_tree()
    operator = RegenerateProjectionOperator()
    receipt = operator.apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            current_manifest=None,
            prior_tree=prior,
            generators=generators,
            require_roundtrip=True,
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert "rollback_restores_exact_prior_tree" in receipt.reason_codes
    assert receipt.prior_snapshot is not None
    assert receipt.prior_snapshot.exact_equals(prior)
    assert receipt.prior_snapshot.exact_equals(dict(receipt.rolled_back_tree))

    inverse = operator.inverse(receipt)
    assert isinstance(inverse, GeneratedTreeSnapshot)
    restored = rollback_tree(inverse)
    assert restored == prior
    assert inverse.exact_equals(restored)

    # Preview tree includes generated paths; inverse does not keep them.
    assert receipt.preview_tree
    assert set(prior).issubset(set(receipt.preview_tree))
    for path in receipt.preview_manifest.tree_map():  # type: ignore[union-attr]
        assert path in receipt.preview_tree
        assert path not in restored or restored.get(path) == prior.get(path)


# ---------------------------------------------------------------------------
# Conflict policy / authority / proposal-only
# ---------------------------------------------------------------------------


def test_generated_artifacts_never_overwrite_hand_owned_code() -> None:
    source = _source(hand_owned_paths=(HAND_OWNED,))
    generators = default_generator_pins()
    # Force a generator output onto a hand-owned path.
    with pytest.raises(CodegenRepairError, match="hand-owned"):
        generate_projection_manifest(
            source,
            generators=generators,
            path_overrides={"schema": HAND_OWNED},
        )


def test_generated_manifest_requires_authority_source_on_every_artifact() -> None:
    source = _source()
    manifest = generate_projection_manifest(source)
    assert manifest.INTERFACE == GENERATED_ARTIFACT_MANIFEST_INTERFACE
    for artifact in manifest.artifacts:
        assert artifact.authority_source_cid
        assert artifact.authority_source_cid == manifest.authority_source_cid
    # Round-trip the manifest contract.
    rebuilt = GeneratedArtifactManifest.from_dict(manifest.to_dict())
    assert rebuilt.tree_digest == manifest.tree_digest
    assert rebuilt.content_id == manifest.content_id


@pytest.mark.parametrize(
    "authority",
    (
        AuthoritySource.PROSE_INFERRED,
        AuthoritySource.INVENTED,
        AuthoritySource.MISSING,
    ),
)
def test_non_admissible_authority_abstains(authority: AuthoritySource) -> None:
    source = _source(authority=authority)
    receipt = RegenerateProjectionOperator().apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            prior_tree=_prior_tree(),
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "conflict_policy_abstain" in receipt.reason_codes
    assert receipt.proposal_only is True


def test_already_aligned_is_idempotent() -> None:
    source = _source()
    generators = default_generator_pins()
    clean = generate_projection_manifest(source, generators=generators)
    receipt = RegenerateProjectionOperator().apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            current_manifest=clean,
            prior_tree=_prior_tree(clean),
            generators=generators,
            require_roundtrip=True,
        )
    )
    assert receipt.disposition is RepairDisposition.ALREADY_ALIGNED
    assert receipt.roundtrip is not None
    assert receipt.roundtrip.byte_identical is True
    assert receipt.roundtrip.semantic_roundtrip_ok is True
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    assert receipt.grants_proof_authority is False
    assert receipt.semantic_authority is False
    assert receipt.evidence_id == CODEGEN_REPAIR_EVIDENCE
    assert receipt.operator_kind == OperatorKind.REGENERATE_PROJECTION.value


def test_preview_ready_carries_evidence_subset() -> None:
    source = _source()
    generators = default_generator_pins(args={"profile": "deterministic"})
    receipt = RegenerateProjectionOperator().apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            prior_tree=_prior_tree(),
            generators=generators,
            require_roundtrip=True,
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.authority_source_cid == source.authority_source_cid
    assert receipt.generator_digest
    assert receipt.generator_args
    assert receipt.output_hashes
    assert receipt.roundtrip is not None
    assert receipt.roundtrip.byte_identical is True
    assert receipt.prior_snapshot is not None
    # Inverse path present on the receipt.
    assert receipt.rolled_back_tree == dict(receipt.prior_snapshot.tree)


def test_operator_bundle_dispatch_and_vectors() -> None:
    ops = build_codegen_repair_operators()
    source = _source()
    receipt = ops.apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            prior_tree=_prior_tree(),
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY

    vectors = materialize_codegen_operator_vectors(source)
    assert vectors["interface"] == CODEGEN_REPAIR_OPERATORS_INTERFACE
    assert vectors["manifest_interface"] == GENERATED_ARTIFACT_MANIFEST_INTERFACE
    assert vectors["evidence_id"] == CODEGEN_REPAIR_EVIDENCE
    assert vectors["stale_fails"] is True
    assert vectors["rollback_restores_exact_prior_tree"] is True
    assert vectors["proposal_only"] is True
    assert vectors["grants_write_authority"] is False
    assert vectors["authority_source_cid"] == source.authority_source_cid
    # Deterministic identity for vector payload itself.
    assert content_identity(vectors) == content_identity(dict(vectors))


def test_receipt_and_request_are_content_addressed() -> None:
    source = _source()
    request = CodegenRepairRequest(
        semantic_source=source,
        role=OperatorRole.REGENERATE_PROJECTION,
        prior_tree=_prior_tree(),
    )
    rebuilt_request = CodegenRepairRequest.from_dict(request.to_dict())
    assert rebuilt_request.content_id == request.content_id
    receipt = RegenerateProjectionOperator().apply(request)
    rebuilt_receipt = type(receipt).from_dict(receipt.to_dict())
    assert rebuilt_receipt.content_id == receipt.content_id
    assert rebuilt_receipt.disposition is receipt.disposition
