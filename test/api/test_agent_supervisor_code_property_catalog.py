"""CBP-020: reviewed property catalog tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_property_catalog import (
    CODE_PROPERTY_CATALOG_INTERFACE,
    DEFAULT_CODE_PROPERTY_CATALOG,
    SRT_STRUCTURAL_TAGS,
    CodeProperty,
    CodePropertyCatalog,
    CodePropertyCatalogError,
    UnknownCodePropertyError,
    build_default_code_property_catalog,
    build_seed_code_properties,
    register_code_property,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof_obligation_templates import (
    ReviewedCodeShape,
)


def test_default_catalog_is_content_addressed_and_stable() -> None:
    first = build_default_code_property_catalog()
    second = build_default_code_property_catalog()
    assert first.catalog_id == second.catalog_id
    assert first.catalog_id  # content-addressed non-empty identity
    round_trip = CodePropertyCatalog.from_dict(first.to_dict())
    assert round_trip.catalog_id == first.catalog_id
    assert round_trip.property_ids() == first.property_ids()


def test_seeds_one_property_per_reviewed_code_shape() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    shapes = {prop.code_shape for prop in catalog.properties}
    assert shapes == {shape.value for shape in ReviewedCodeShape}
    assert len(catalog.properties) == len(ReviewedCodeShape)


def test_srt_structural_tags_declared_and_present_on_seeds() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    assert set(SRT_STRUCTURAL_TAGS).issubset(set(catalog.declared_tags))
    for prop in catalog.properties:
        for tag in SRT_STRUCTURAL_TAGS:
            assert tag in prop.query_tags


def test_semantic_authority_defaults_false() -> None:
    for prop in DEFAULT_CODE_PROPERTY_CATALOG.properties:
        assert prop.semantic_authority is False
    payload = DEFAULT_CODE_PROPERTY_CATALOG.properties[0].to_dict()
    assert payload["semantic_authority"] is False


def test_unknown_ids_fail_closed() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    assert catalog.get("property:does-not-exist") is None
    with pytest.raises(UnknownCodePropertyError):
        catalog.require("property:does-not-exist")


def test_require_returns_seeded_property() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    prop = catalog.require("property:lease-uniqueness-and-fencing")
    assert prop.template_id == "lease-uniqueness-and-fencing"
    assert prop.code_shape == ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING.value
    assert prop.required_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_closed_registration_rejects_unknown_template() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    bad = CodeProperty(
        property_id="property:invented",
        template_id="not-a-reviewed-template",
        template_version="1",
        template_semantic_hash="sha256:x",
        code_shape=ReviewedCodeShape.DAG_ACYCLICITY.value,
        sorts=("code",),
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        query_tags=("x",),
    )
    with pytest.raises(CodePropertyCatalogError, match="unknown reviewed template"):
        register_code_property(catalog, bad)


def test_closed_registration_rejects_shape_template_mismatch() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    bad = CodeProperty(
        property_id="property:mismatch",
        template_id="lease-uniqueness-and-fencing",
        template_version="1.0.0",
        template_semantic_hash="sha256:x",
        code_shape=ReviewedCodeShape.DAG_ACYCLICITY.value,
        sorts=("code",),
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        query_tags=("x",),
    )
    with pytest.raises(CodePropertyCatalogError, match="code_shape"):
        register_code_property(catalog, bad)


def test_closed_registration_rejects_semantic_authority_true() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    bad = CodeProperty(
        property_id="property:authority",
        template_id="dag-acyclicity",
        template_version="1.0.0",
        template_semantic_hash="sha256:x",
        code_shape=ReviewedCodeShape.DAG_ACYCLICITY.value,
        sorts=("code",),
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        query_tags=("x",),
        semantic_authority=True,
    )
    with pytest.raises(CodePropertyCatalogError, match="semantic_authority"):
        register_code_property(catalog, bad)


def test_register_accepts_reviewed_duplicate_shape_with_new_id() -> None:
    catalog = DEFAULT_CODE_PROPERTY_CATALOG
    seed = catalog.require("property:dag-acyclicity")
    extra = CodeProperty(
        property_id="property:dag-acyclicity-alias",
        template_id=seed.template_id,
        template_version=seed.template_version,
        template_semantic_hash=seed.template_semantic_hash,
        code_shape=seed.code_shape,
        sorts=seed.sorts,
        required_assurance=seed.required_assurance,
        query_tags=seed.query_tags + ("alias",),
        semantic_authority=False,
        invariant_class=seed.invariant_class,
    )
    expanded = register_code_property(catalog, extra)
    assert expanded.require("property:dag-acyclicity-alias").template_id == seed.template_id
    assert catalog.get("property:dag-acyclicity-alias") is None


def test_catalog_interface_constant() -> None:
    assert CODE_PROPERTY_CATALOG_INTERFACE == "CodePropertyCatalog@1"
    assert DEFAULT_CODE_PROPERTY_CATALOG.to_dict()["interface"] == (
        CODE_PROPERTY_CATALOG_INTERFACE
    )


def test_seed_builder_is_deterministic() -> None:
    a = build_seed_code_properties()
    b = build_seed_code_properties()
    assert [p.property_id for p in a] == [p.property_id for p in b]
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_tampered_catalog_id_rejected() -> None:
    payload = DEFAULT_CODE_PROPERTY_CATALOG.to_dict()
    payload["catalog_id"] = "sha256:tampered"
    with pytest.raises(CodePropertyCatalogError, match="catalog_id"):
        CodePropertyCatalog.from_dict(payload)
