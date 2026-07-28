"""CBP-110: semantic-roundtrip residual / structural bridge tests.

Uses in-process fixtures only — never live gold IR dumps or sealed PLAT
promotion report rewrites.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
)
from ipfs_accelerate_py.agent_supervisor.code_edit_packet import (
    CODE_EDIT_PACKET_INTERFACE,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_srt_bridge import (
    CODE_PROOF_SRT_BRIDGE_INTERFACE,
    METHOD_ROLES_TABLE,
    PLATEAU_CODEX_PACKET_INTERFACE,
    PLAT2_HOLDOUT_REGISTRY_INTERFACE,
    PROMOTION_AUTHORITIES,
    STRUCTURAL_ADMISSION_INTERFACE,
    STRUCTURAL_SEMANTIC_AUTHORITY,
    CodeProofSrtBridgeError,
    Plat2HoldoutArtifact,
    Plat2HoldoutRegistry,
    PlatResidualCatalog,
    ResidualCatalogEntry,
    StructuralAdmission,
    StructuralAdmissionDisposition,
    SrtMethodRole,
    build_plat_residual_catalog,
    build_srt_cache_key_handles,
    build_structural_admission,
    method_role_description,
    method_roles_manifest,
    project_plateau_codex_packet,
    project_plateau_packet_bundle,
    project_plat_residual_catalog,
    project_residual_to_claim,
    project_residual_to_code_edit_packet,
    project_residual_to_context_capsule,
    project_residual_to_counterexample,
    project_structural_admission_to_graph,
    reject_gold_ir_bodies,
    resolve_method_role,
    structural_admission_to_claims,
)
from ipfs_accelerate_py.agent_supervisor.code_property_catalog import (
    SRT_STRUCTURAL_TAGS,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)


# ---------------------------------------------------------------------------
# Fixtures (synthetic handles only — no live gold dumps)
# ---------------------------------------------------------------------------


def _residual_entry(
    *,
    residual_ref_id: str = "residual:fixture-r1",
    status: str = ClaimStatus.OPEN.value,
    with_counterexample: bool = False,
) -> ResidualCatalogEntry:
    return ResidualCatalogEntry(
        residual_ref_id=residual_ref_id,
        residual_kind="structural",
        structural_tags=("non_vacuous_candidate", "rule_cardinality_preserved"),
        property_ids=("property:srt-fixture-r1",),
        claim_ids=("claim:fixture-1",),
        obligation_ids=("obligation:srt-fixture-1",),
        counterexample_ref_ids=(
            ("ce:fixture-1",) if with_counterexample else ()
        ),
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/code_proof_srt_bridge.py",
        ),
        assumption_ids=("assumption:fixture-a",),
        status=status,
        summary="fixture residual for CBP-110",
        metadata={"fixture": True, "gold_ir_excluded": True},
    )


def _plateau_fixture() -> dict:
    return {
        "packet_id": "plateau:fixture-1",
        "repository_tree_id": "git-tree:srt-fixture",
        "repository_id": "repository:srt-fixture",
        "residual_ref_ids": ("residual:fixture-r1",),
        "claim_ids": ("claim:fixture-1",),
        "obligation_ids": ("obligation:srt-fixture-1",),
        "assumption_ids": ("assumption:fixture-a",),
        "property_ids": ("property:srt-fixture-r1",),
        "predicted_files": (
            "ipfs_accelerate_py/agent_supervisor/code_proof_srt_bridge.py",
        ),
        "status": ClaimStatus.OPEN.value,
        "acceptance_ids": ("accept:srt-fixture",),
    }


def _admission_fixture() -> StructuralAdmission:
    return build_structural_admission(
        residual_ref_ids=("residual:fixture-r1",),
        structural_tags=("non_vacuous_candidate",),
        disposition=StructuralAdmissionDisposition.ADMITTED,
        gate_method_ids=("hammer", "cvc5", "lean"),
        repository_tree_id="git-tree:srt-fixture",
        repository_id="repository:srt-fixture",
        property_ids=("property:srt-fixture-r1",),
        obligation_ids=("obligation:srt-fixture-1",),
        receipt_id="receipt:structural-fixture-1",
        reason_codes=("structural_tags_satisfied",),
        metadata={"fixture": True},
    )


# ---------------------------------------------------------------------------
# Method roles
# ---------------------------------------------------------------------------


def test_method_roles_map_measured_and_not_interchangeable() -> None:
    assert resolve_method_role("autoencoder") is (
        SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS
    )
    assert resolve_method_role("spacy") is (
        SrtMethodRole.BOUNDED_GUIDANCE_DIAGNOSTICS
    )
    assert resolve_method_role("symai") is SrtMethodRole.ORCHESTRATION
    assert resolve_method_role("leanstral") is SrtMethodRole.PROPOSAL_TEACHER
    assert resolve_method_role("hammer") is SrtMethodRole.STRUCTURAL_GATE
    assert resolve_method_role("cvc5") is SrtMethodRole.STRUCTURAL_GATE
    assert resolve_method_role("lean") is SrtMethodRole.STRUCTURAL_GATE
    assert resolve_method_role("compiler") is SrtMethodRole.EDIT_TARGET
    assert resolve_method_role("ir") is SrtMethodRole.EDIT_TARGET
    assert resolve_method_role("decompiler") is SrtMethodRole.EDIT_TARGET

    # Roles are distinct — guidance is not a structural gate.
    assert (
        resolve_method_role("autoencoder")
        is not resolve_method_role("hammer")
    )
    assert (
        resolve_method_role("leanstral")
        is not resolve_method_role("lean")
    )

    manifest = method_roles_manifest()
    assert manifest["methods_interchangeable"] is False
    assert manifest["structural_semantic_authority"] is False
    assert manifest["gold_ir_in_cache_keys"] is False
    assert set(manifest["promotion_authorities"]) == set(PROMOTION_AUTHORITIES)
    assert "semantic_roundtrip_e2e_loss" in PROMOTION_AUTHORITIES
    assert "plat2_holdout_promotion_gate" in PROMOTION_AUTHORITIES

    for role, methods in METHOD_ROLES_TABLE.items():
        assert methods
        assert method_role_description(role)


def test_unknown_method_role_fails_closed() -> None:
    with pytest.raises(CodeProofSrtBridgeError, match="unknown SRT method"):
        resolve_method_role("invented-interchangeable-method")


# ---------------------------------------------------------------------------
# Residual → claim / counterexample / capsule / CodeEditPacket
# ---------------------------------------------------------------------------


def test_residual_projects_to_typed_srt_structural_claim() -> None:
    entry = _residual_entry()
    claim = project_residual_to_claim(
        entry,
        repository_tree_id="git-tree:srt-fixture",
        repository_id="repository:srt-fixture",
    )
    assert claim.claim_family is ClaimFamily.SRT_STRUCTURAL
    assert claim.status is ClaimStatus.OPEN
    assert claim.premise_ids == ("residual:fixture-r1",)
    assert claim.property_id == "property:srt-fixture-r1"
    assert claim.derived_assurance is AssuranceLevel.UNVERIFIED
    assert claim.metadata["semantic_authority"] is False
    assert set(claim.metadata["promotion_authorities"]) == set(
        PROMOTION_AUTHORITIES
    )
    assert claim.metadata["bridge"] == CODE_PROOF_SRT_BRIDGE_INTERFACE
    # Round-trip.
    restored = claim.__class__.from_dict(claim.to_dict())
    assert restored.claim_id == claim.claim_id


def test_residual_projects_to_counterexample_without_gold() -> None:
    entry = _residual_entry(with_counterexample=True, status=ClaimStatus.REFUTED.value)
    ce = project_residual_to_counterexample(
        entry, repository_tree_id="git-tree:srt-fixture"
    )
    payload = ce.to_dict()
    body = str(payload)
    assert "SECRET" not in body
    assert "gold_ir_body" not in body
    assert "SECRET_GOLD" not in body
    assert ce.property_class == "srt_structural"
    assert ce.violated_property == "property:srt-fixture-r1"
    # Residual identity rides on bindings / summary (normalizer redacts freeform
    # residual maps for GENERIC_FAILURE into a public failure code).
    assert "git-tree:srt-fixture" in ce.bindings.tree_ids
    assert "obligation:srt-fixture-1" in ce.bindings.obligation_ids
    assert "residual:fixture-r1" in ce.summary or "fixture residual" in ce.summary
    assert ce.minimized is True


def test_residual_projects_to_context_capsule() -> None:
    entry = _residual_entry(with_counterexample=True)
    capsule = project_residual_to_context_capsule(
        entry,
        repository_id="repository:srt-fixture",
        repository_tree_id="git-tree:srt-fixture",
        plateau_packet_id="plateau:fixture-1",
    )
    assert capsule.tree_id == "git-tree:srt-fixture"
    assert capsule.authority["semantic_authority"] is False
    assert set(capsule.authority["promotion_authorities"]) == set(
        PROMOTION_AUTHORITIES
    )
    assert capsule.acceptance["requires_e2e_loss"] is True
    assert capsule.acceptance["requires_holdout_gate"] is True
    assert capsule.acceptance["structural_admission_insufficient"] is True
    assert capsule.scope["residual_ref_id"] == "residual:fixture-r1"
    assert capsule.truncated is False
    assert any(ref.kind == "residual_ref" for ref in capsule.evidence)
    # No gold bodies in serialized capsule.
    serialized = str(capsule.to_dict())
    assert "gold_ir_body" not in serialized
    assert "SECRET_GOLD" not in serialized


def test_residual_projects_to_code_edit_packet() -> None:
    entry = _residual_entry()
    packet = project_residual_to_code_edit_packet(
        entry,
        repository_tree_id="git-tree:srt-fixture",
        plateau_packet_id="plateau:fixture-1",
    )
    assert packet.interface == CODE_EDIT_PACKET_INTERFACE
    assert packet.implementable is True
    assert packet.residual_ref_ids == ("residual:fixture-r1",)
    assert packet.prover.semantic_authority is False
    assert packet.metadata["gold_ir_excluded"] is True
    assert packet.metadata["edit_target_role"] == SrtMethodRole.EDIT_TARGET.value
    assert set(packet.metadata["promotion_authorities"]) == set(
        PROMOTION_AUTHORITIES
    )
    body = str(packet.to_dict())
    assert "gold_ir_body" not in body
    assert "SECRET" not in body


# ---------------------------------------------------------------------------
# PlateauCodexPacket@1
# ---------------------------------------------------------------------------


def test_plateau_codex_packet_projects_to_code_edit_packet() -> None:
    packet = project_plateau_codex_packet(_plateau_fixture())
    assert packet.plateau_packet_id == "plateau:fixture-1"
    assert packet.residual_ref_ids == ("residual:fixture-r1",)
    assert packet.implementable is True
    assert packet.prover.semantic_authority is False
    assert packet.metadata["source"] == PLATEAU_CODEX_PACKET_INTERFACE
    assert packet.metadata["gold_ir_excluded"] is True
    assert packet.metadata["bridge"] == CODE_PROOF_SRT_BRIDGE_INTERFACE


def test_plateau_packet_rejects_gold_ir_bodies() -> None:
    bad = {**_plateau_fixture(), "gold_ir_body": {"ir": "SECRET_GOLD_BODY"}}
    with pytest.raises(CodeProofSrtBridgeError, match="gold IR|proof bodies"):
        project_plateau_codex_packet(bad)


def test_plateau_packet_bundle_projects_claims_and_packets() -> None:
    projection = project_plateau_packet_bundle(
        _plateau_fixture(),
        residual_entries=(_residual_entry(),),
        structural_admission=_admission_fixture(),
    )
    assert projection.semantic_authority is False
    assert projection.gold_ir_excluded is True
    assert "plateau:fixture-1" in projection.plateau_packet_ids
    assert "residual:fixture-r1" in projection.residual_ref_ids
    assert projection.code_edit_packets
    assert projection.claims
    assert projection.graph_projections
    assert all(c.claim_family is ClaimFamily.SRT_STRUCTURAL for c in projection.claims)
    notes = set(projection.notes)
    assert "gold_ir_excluded" in notes
    assert "e2e_loss_and_holdout_remain_promotion_authority" in notes
    assert "methods_not_interchangeable" in notes


# ---------------------------------------------------------------------------
# StructuralAdmission@1 → graph / query (non-semantic)
# ---------------------------------------------------------------------------


def test_structural_admission_forces_non_semantic_authority() -> None:
    admission = _admission_fixture()
    assert admission.interface == STRUCTURAL_ADMISSION_INTERFACE
    assert admission.semantic_authority is False
    assert STRUCTURAL_SEMANTIC_AUTHORITY is False
    payload = admission.to_dict()
    assert payload["semantic_authority"] is False
    # Caller cannot force true via from_dict.
    restored = StructuralAdmission.from_dict(
        {**payload, "semantic_authority": True}
    )
    assert restored.semantic_authority is False


def test_structural_admission_rejects_non_gate_methods() -> None:
    with pytest.raises(CodeProofSrtBridgeError, match="structural_gate"):
        build_structural_admission(
            residual_ref_ids=("residual:fixture-r1",),
            structural_tags=("non_vacuous_candidate",),
            gate_method_ids=("autoencoder",),  # guidance, not a gate
        )


def test_structural_admission_projects_to_graph_with_non_semantic_authority() -> None:
    admission = _admission_fixture()
    graph = project_structural_admission_to_graph(admission)
    assert graph.semantic_authority is False
    payload = graph.to_dict()
    assert payload["semantic_authority"] is False
    assert payload["non_authoritative"] is True
    assert graph.nodes
    assert graph.edges
    assert graph.query_facts
    for fact in graph.query_facts:
        assert fact["semantic_authority"] is False
        assert fact["non_authoritative"] is True
        assert set(fact["promotion_authorities"]) == set(PROMOTION_AUTHORITIES)
        assert fact["evidence_tier"] == "query_fact"
    for node in graph.nodes:
        assert node.get("semantic_authority") is False


def test_structural_admission_projects_to_typed_claims() -> None:
    claims = structural_admission_to_claims(_admission_fixture())
    assert claims
    claim = claims[0]
    assert claim.claim_family is ClaimFamily.SRT_STRUCTURAL
    assert claim.status is ClaimStatus.OPEN  # admitted → open for further evidence
    assert claim.metadata["semantic_authority"] is False
    assert claim.derived_assurance is AssuranceLevel.UNVERIFIED


# ---------------------------------------------------------------------------
# PLAT residual catalog aggregate
# ---------------------------------------------------------------------------


def test_plat_residual_catalog_projects_into_cbp_surfaces() -> None:
    catalog = build_plat_residual_catalog(
        (_residual_entry(with_counterexample=True),),
        repository_tree_id="git-tree:srt-fixture",
        repository_id="repository:srt-fixture",
        plateau_packet_id="plateau:fixture-1",
        metadata={"fixture": True},
    )
    assert catalog.catalog_id
    assert catalog.residual_ref_ids() == ("residual:fixture-r1",)
    # Content-addressed stability.
    again = PlatResidualCatalog.from_dict(catalog.to_dict())
    assert again.catalog_id == catalog.catalog_id

    projection = project_plat_residual_catalog(
        catalog,
        structural_admission=_admission_fixture(),
    )
    assert projection.claims
    assert projection.counterexamples
    assert projection.context_capsules
    assert projection.code_edit_packets
    assert projection.cache_key_handles
    assert projection.graph_projections
    assert projection.semantic_authority is False
    assert projection.gold_ir_excluded is True
    assert set(projection.promotion_authorities) == set(PROMOTION_AUTHORITIES)

    for handles in projection.cache_key_handles:
        assert handles.gold_ir_excluded is True
        key = handles.build_proof_cache_key()
        key_blob = str(key.to_dict())
        assert "SECRET" not in key_blob
        assert "gold_ir_body" not in key_blob
        assert "residual:fixture-r1" in key_blob or key.key_id


def test_residual_catalog_rejects_gold_ir_entries() -> None:
    with pytest.raises(CodeProofSrtBridgeError, match="gold IR|proof bodies"):
        ResidualCatalogEntry.from_dict(
            {
                "residual_ref_id": "residual:bad",
                "gold_ir": "SECRET_GOLD_BODY",
            }
        )


# ---------------------------------------------------------------------------
# PLAT2 holdout — separate preregistration and query
# ---------------------------------------------------------------------------


def test_plat2_holdout_separately_preregistered_and_queryable() -> None:
    registry = Plat2HoldoutRegistry()
    assert registry.interface == PLAT2_HOLDOUT_REGISTRY_INTERFACE

    art = registry.register(
        {
            "artifact_id": "holdout:fixture-a",
            "holdout_split": "plat2-holdout-a",
            "residual_ref_ids": ("residual:fixture-r1",),
            "property_ids": ("property:srt-fixture-r1",),
            "metric_ids": ("metric:e2e_loss",),
            "repository_tree_id": "git-tree:srt-fixture",
            "preregistered": True,
            "queryable": True,
        }
    )
    assert isinstance(art, Plat2HoldoutArtifact)
    assert art.preregistered is True
    assert art.queryable is True
    assert art.promotion_gate == "plat2_holdout_promotion_gate"

    # Separate training residual is not a holdout hit unless registered.
    hits = registry.query(residual_ref_id="residual:fixture-r1")
    assert [h.artifact_id for h in hits] == ["holdout:fixture-a"]

    by_split = registry.query(holdout_split="plat2-holdout-a")
    assert len(by_split) == 1

    by_prop = registry.query(property_id="property:srt-fixture-r1")
    assert len(by_prop) == 1

    empty = registry.query(holdout_split="does-not-exist")
    assert empty == ()

    # PLAT residual catalog remains independent of holdout registry.
    catalog = build_plat_residual_catalog(
        (_residual_entry(),),
        repository_tree_id="git-tree:srt-fixture",
    )
    assert "holdout:fixture-a" not in catalog.to_dict().get("entries", [])


def test_plat2_holdout_rejects_non_preregistered_and_gold() -> None:
    registry = Plat2HoldoutRegistry()
    with pytest.raises(CodeProofSrtBridgeError, match="preregistered"):
        registry.register(
            Plat2HoldoutArtifact(
                artifact_id="holdout:bad",
                holdout_split="x",
                preregistered=False,
            )
        )
    with pytest.raises(CodeProofSrtBridgeError, match="gold IR|proof bodies"):
        Plat2HoldoutArtifact.from_dict(
            {
                "artifact_id": "holdout:gold",
                "holdout_split": "x",
                "gold_ir_body": "SECRET",
            }
        )


# ---------------------------------------------------------------------------
# Cache keys exclude gold IR bodies
# ---------------------------------------------------------------------------


def test_cache_key_handles_exclude_gold_ir_bodies() -> None:
    handles = build_srt_cache_key_handles(
        residual_ref_ids=("residual:fixture-r1",),
        obligation_ids=("obligation:srt-fixture-1",),
        property_ids=("property:srt-fixture-r1",),
        structural_tags=SRT_STRUCTURAL_TAGS[:1],
        repository_tree_id="git-tree:srt-fixture",
        gate_method_ids=("hammer", "cvc5", "lean"),
    )
    assert handles.gold_ir_excluded is True
    assert "residual:fixture-r1" in handles.premise_handles()
    key = handles.build_proof_cache_key()
    blob = str(key.to_dict()) + key.key_id
    assert "SECRET_GOLD" not in blob
    assert "gold_ir_body" not in blob
    # Same handles → same key.
    again = handles.build_proof_cache_key()
    assert again.key_id == key.key_id

    with pytest.raises(CodeProofSrtBridgeError, match="gold IR|proof bodies"):
        build_srt_cache_key_handles(
            residual_ref_ids=("residual:fixture-r1",),
            payload={"gold_ir": "SECRET_GOLD_BODY"},
        )


def test_reject_gold_ir_bodies_nested() -> None:
    reject_gold_ir_bodies(
        {"residual_ref_id": "residual:ok", "gold_ir_excluded": True},
        where="fixture",
    )
    with pytest.raises(CodeProofSrtBridgeError):
        reject_gold_ir_bodies(
            {"nested": {"proof_body": "x"}},
            where="fixture",
        )


# ---------------------------------------------------------------------------
# Promotion authority doctrine
# ---------------------------------------------------------------------------


def test_promotion_authorities_remain_e2e_and_holdout() -> None:
    assert PROMOTION_AUTHORITIES == (
        "semantic_roundtrip_e2e_loss",
        "plat2_holdout_promotion_gate",
    )
    projection = project_plat_residual_catalog(
        build_plat_residual_catalog(
            (_residual_entry(),),
            repository_tree_id="git-tree:srt-fixture",
        )
    )
    assert projection.promotion_authorities == PROMOTION_AUTHORITIES
    # Structural graph facts restate the same authorities.
    facts = project_structural_admission_to_graph(_admission_fixture()).query_facts
    for fact in facts:
        assert tuple(fact["promotion_authorities"]) == PROMOTION_AUTHORITIES


def test_structural_tags_align_with_property_catalog() -> None:
    entry = _residual_entry()
    for tag in entry.structural_tags:
        assert tag in SRT_STRUCTURAL_TAGS
    admission = _admission_fixture()
    for tag in admission.structural_tags:
        assert tag in SRT_STRUCTURAL_TAGS


def test_bridge_projection_serializes_without_gold() -> None:
    projection = project_plateau_packet_bundle(
        _plateau_fixture(),
        residual_entries=(_residual_entry(with_counterexample=True),),
        structural_admission=_admission_fixture(),
    )
    payload = projection.to_dict()
    assert payload["interface"] == CODE_PROOF_SRT_BRIDGE_INTERFACE
    assert payload["semantic_authority"] is False
    assert payload["gold_ir_excluded"] is True
    text = str(payload)
    assert "SECRET_GOLD" not in text
    assert "gold_ir_body" not in text
