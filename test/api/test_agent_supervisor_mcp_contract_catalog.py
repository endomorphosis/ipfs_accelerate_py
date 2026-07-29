"""SCA-040: reviewed MCP contract catalog tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    CATALOG_VERSION,
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_MCP_CONTRACT_CATALOG,
    MCP_CONTRACT_CATALOG_INTERFACE,
    ClaimFamilyDescriptor,
    ContractInvalidationKind,
    ContractInvalidator,
    ContractRecord,
    ContractSourceKind,
    ContractSourceRecord,
    ContradictionRecord,
    McpClaimFamily,
    McpContractCatalog,
    McpContractCatalogError,
    ReviewState,
    SourceAuthorityClass,
    UnknownMcpClaimFamilyError,
    UnknownMcpContractError,
    UnreviewedContractError,
    admit_source,
    authority_for_source_kind,
    build_contract_from_sources,
    build_default_mcp_contract_catalog,
    build_seed_claim_families,
    build_source_invalidators,
    detect_source_contradictions,
    evaluate_invalidation,
    make_source_record,
    nominate_from_prose,
    register_contract,
    register_sources_and_contract,
    reject_natural_language_claim,
    require_complete_version_invalidators,
)


# ---------------------------------------------------------------------------
# Canonical IDs
# ---------------------------------------------------------------------------


def test_default_catalog_is_content_addressed_and_stable() -> None:
    first = build_default_mcp_contract_catalog()
    second = build_default_mcp_contract_catalog()
    assert first.catalog_id == second.catalog_id
    assert first.catalog_id  # non-empty CID-shaped identity
    assert first.catalog_id.startswith("b")
    round_trip = McpContractCatalog.from_dict(first.to_dict())
    assert round_trip.catalog_id == first.catalog_id
    assert round_trip.family_ids() == first.family_ids()


def test_source_and_contract_ids_are_canonical() -> None:
    source = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="tool:echo",
        source_version="1.0.0",
        schema_version="2020-12",
        path="schemas/echo.json",
        payload_fingerprint="fp-echo-v1",
    )
    assert source.source_id.startswith("b")
    again = ContractSourceRecord.from_dict(source.to_dict())
    assert again.source_id == source.source_id

    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, source)
    contract, _ = build_contract_from_sources(
        claim_family=McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
        subject="tool:echo",
        sources=(source,),
        tool_name="echo",
    )
    assert contract.contract_id.startswith("b")
    registered = register_contract(catalog, contract)
    assert registered.require_contract(contract.contract_id).contract_id == (
        contract.contract_id
    )


def test_tampered_catalog_id_rejected() -> None:
    payload = DEFAULT_MCP_CONTRACT_CATALOG.to_dict()
    payload["catalog_id"] = "baguqeeratampered0000000000000000000000000000000000000"
    with pytest.raises(McpContractCatalogError, match="catalog_id"):
        McpContractCatalog.from_dict(payload)


def test_tampered_source_id_rejected() -> None:
    source = make_source_record(
        kind=ContractSourceKind.MCP_IDL,
        subject="iface:tools",
        source_version="2",
    )
    payload = source.to_dict()
    payload["source_id"] = "baguqeeratampered0000000000000000000000000000000000000"
    with pytest.raises(McpContractCatalogError, match="source_id"):
        ContractSourceRecord.from_dict(payload)


# ---------------------------------------------------------------------------
# Closed claim families + interface
# ---------------------------------------------------------------------------


def test_seed_claim_families_cover_plan_set() -> None:
    families = build_seed_claim_families()
    names = {f.family for f in families}
    assert names == set(McpClaimFamily)
    assert len(families) == len(McpClaimFamily)
    for desc in families:
        assert desc.review_state is ReviewState.REVIEWED
        assert desc.family_id.startswith("b")


def test_default_catalog_exposes_all_closed_families() -> None:
    catalog = DEFAULT_MCP_CONTRACT_CATALOG
    assert set(catalog.family_ids()) == {f.value for f in McpClaimFamily}
    for family in McpClaimFamily:
        assert catalog.require_family(family).family is family


def test_unknown_claim_family_fails_closed() -> None:
    catalog = DEFAULT_MCP_CONTRACT_CATALOG
    assert catalog.get_family("InventedFamily") is None
    with pytest.raises(UnknownMcpClaimFamilyError):
        catalog.require_family("InventedFamily")
    with pytest.raises(McpContractCatalogError, match="unknown claim_family"):
        ContractRecord(
            claim_family="NotAFamily",  # type: ignore[arg-type]
            subject="x",
            source_ids=("s1",),
            authority_class=SourceAuthorityClass.AUTHORITATIVE,
            review_state=ReviewState.REVIEWED,
            source_version="1",
            schema_version="1",
        )


def test_catalog_interface_constant() -> None:
    assert MCP_CONTRACT_CATALOG_INTERFACE == "McpContractCatalog@1"
    assert DEFAULT_MCP_CONTRACT_CATALOG.to_dict()["interface"] == (
        MCP_CONTRACT_CATALOG_INTERFACE
    )
    assert DEFAULT_MCP_CONTRACT_CATALOG.catalog_version == CATALOG_VERSION


# ---------------------------------------------------------------------------
# Explicit authority and review state
# ---------------------------------------------------------------------------


def test_source_kinds_retain_authority_class() -> None:
    expected = {
        ContractSourceKind.MCP_IDL: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.JSON_SCHEMA: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.TYPED_INTERFACE: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.POLICY_CONTRACT: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.CONFORMANCE_TEST: SourceAuthorityClass.CONFORMANCE,
        ContractSourceKind.REGISTRATION: SourceAuthorityClass.REGISTRATION,
        ContractSourceKind.MANIFEST: SourceAuthorityClass.MANIFEST,
        ContractSourceKind.DOCUMENTATION: SourceAuthorityClass.NOMINATING,
        ContractSourceKind.INFERRED_PROSE: SourceAuthorityClass.NONE,
    }
    for kind, authority in expected.items():
        assert authority_for_source_kind(kind) is authority
        source = make_source_record(
            kind=kind,
            subject="tool:demo",
            source_version="1",
        )
        assert source.authority_class is authority
        assert source.review_state is not None


def test_cannot_promote_documentation_to_authoritative() -> None:
    with pytest.raises(McpContractCatalogError, match="exceeds allowed class"):
        ContractSourceRecord(
            kind=ContractSourceKind.DOCUMENTATION,
            authority_class=SourceAuthorityClass.AUTHORITATIVE,
            review_state=ReviewState.REVIEWED,
            source_version="1",
            schema_version="1",
            subject="tool:docs",
        )


def test_authority_rank_ordering() -> None:
    ranks = [
        SourceAuthorityClass.AUTHORITATIVE.rank,
        SourceAuthorityClass.CONFORMANCE.rank,
        SourceAuthorityClass.REGISTRATION.rank,
        SourceAuthorityClass.MANIFEST.rank,
        SourceAuthorityClass.NOMINATING.rank,
        SourceAuthorityClass.NONE.rank,
    ]
    assert ranks == sorted(ranks)


def test_reviewed_contract_requires_authorizing_authority() -> None:
    with pytest.raises(UnreviewedContractError, match="authorizing authority"):
        ContractRecord(
            claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
            subject="tool:x",
            source_ids=("sid",),
            authority_class=SourceAuthorityClass.NOMINATING,
            review_state=ReviewState.REVIEWED,
            source_version="1",
            schema_version="1",
        )


# ---------------------------------------------------------------------------
# Contradictory sources remain contradictory
# ---------------------------------------------------------------------------


def test_contradictory_sources_remain_unresolved() -> None:
    a = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="tool:conflict",
        source_version="1.0.0",
        schema_version="2020-12",
        payload_fingerprint="schema-a",
        path="a.json",
    )
    b = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="tool:conflict",
        source_version="1.0.0",
        schema_version="2020-12",
        payload_fingerprint="schema-b",
        path="b.json",
    )
    contradictions = detect_source_contradictions((a, b))
    assert contradictions
    for ctr in contradictions:
        assert ctr.resolved is False
        assert len(ctr.values) >= 2
        assert a.source_id in ctr.source_ids
        assert b.source_id in ctr.source_ids

    contract, ctrs = build_contract_from_sources(
        claim_family=McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
        subject="tool:conflict",
        sources=(a, b),
    )
    assert contract.review_state is ReviewState.CONTRADICTED
    assert contract.contradiction_ids
    assert all(not c.resolved for c in ctrs)

    catalog = DEFAULT_MCP_CONTRACT_CATALOG
    catalog = admit_source(catalog, a)
    catalog = admit_source(catalog, b)
    catalog = register_contract(catalog, contract, contradictions=ctrs)
    stored = catalog.contradictions_for("tool:conflict")
    assert stored
    assert all(c.resolved is False for c in stored)


def test_contradiction_cannot_be_marked_resolved() -> None:
    with pytest.raises(McpContractCatalogError, match="remain unresolved"):
        ContradictionRecord(
            subject="tool:x",
            field_name="payload_fingerprint",
            source_ids=("a", "b"),
            values=("v1", "v2"),
            resolved=True,
        )
    payload = {
        "subject": "tool:x",
        "field_name": "payload_fingerprint",
        "source_ids": ["a", "b"],
        "values": ["v1", "v2"],
        "resolved": True,
    }
    with pytest.raises(McpContractCatalogError, match="remain unresolved"):
        ContradictionRecord.from_dict(payload)


def test_reviewed_contract_rejects_bound_contradictions() -> None:
    with pytest.raises(McpContractCatalogError, match="contradiction"):
        ContractRecord(
            claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
            subject="tool:x",
            source_ids=("s1",),
            authority_class=SourceAuthorityClass.AUTHORITATIVE,
            review_state=ReviewState.REVIEWED,
            source_version="1",
            schema_version="1",
            contradiction_ids=("ctr-1",),
        )


# ---------------------------------------------------------------------------
# Unknown / unreviewed prose fails closed
# ---------------------------------------------------------------------------


def test_inferred_prose_cannot_become_reviewed_contract() -> None:
    prose = nominate_from_prose(
        subject="tool:maybe",
        prose="The echo tool probably accepts a message string.",
        path="docs/tools.md",
    )
    assert prose.authority_class is SourceAuthorityClass.NONE
    assert prose.review_state is ReviewState.NOMINATED
    assert prose.may_authorize_contract is False

    with pytest.raises(UnreviewedContractError):
        build_contract_from_sources(
            claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
            subject="tool:maybe",
            sources=(prose,),
            require_reviewed=True,
        )


def test_documentation_alone_fails_closed_for_reviewed_registration() -> None:
    doc = make_source_record(
        kind=ContractSourceKind.DOCUMENTATION,
        subject="tool:readme",
        source_version="docs-1",
        review_state=ReviewState.NOMINATED,
    )
    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, doc)
    with pytest.raises(UnreviewedContractError):
        register_sources_and_contract(
            catalog,
            claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
            subject="tool:readme",
            sources=(doc,),
        )


def test_natural_language_markers_fail_closed() -> None:
    with pytest.raises(UnreviewedContractError, match="fails closed"):
        reject_natural_language_claim({"freeform_statement": "anything goes"})
    with pytest.raises(UnreviewedContractError, match="fails closed"):
        reject_natural_language_claim("this is a natural_language claim")
    with pytest.raises(UnreviewedContractError, match="fails closed"):
        ContractRecord(
            claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
            subject="tool:nl",
            source_ids=("s",),
            authority_class=SourceAuthorityClass.AUTHORITATIVE,
            review_state=ReviewState.REVIEWED,
            source_version="1",
            schema_version="1",
            metadata={"nl_claim": True},
        )


def test_unknown_contract_id_fails_closed() -> None:
    catalog = DEFAULT_MCP_CONTRACT_CATALOG
    assert catalog.get_contract("missing") is None
    with pytest.raises(UnknownMcpContractError):
        catalog.require_contract("missing")


# ---------------------------------------------------------------------------
# Source and schema version invalidators are complete
# ---------------------------------------------------------------------------


def test_version_invalidators_always_present() -> None:
    invs = build_source_invalidators(
        source_version="1.2.3",
        schema_version="2020-12",
    )
    kinds = {i.kind for i in invs}
    assert ContractInvalidationKind.SOURCE_VERSION in kinds
    assert ContractInvalidationKind.SCHEMA_VERSION in kinds
    require_complete_version_invalidators(invs)


def test_incomplete_invalidators_fail_closed() -> None:
    partial = (
        ContractInvalidator(
            kind=ContractInvalidationKind.SOURCE_VERSION,
            value="1",
            reason_code="source_version_drift",
        ),
    )
    with pytest.raises(McpContractCatalogError, match="incomplete version"):
        require_complete_version_invalidators(partial)


def test_source_record_auto_builds_complete_invalidators() -> None:
    source = make_source_record(
        kind=ContractSourceKind.REGISTRATION,
        subject="tool:reg",
        source_version="pkg-3",
        schema_version="iface-2",
    )
    kinds = {i.kind for i in source.invalidators}
    assert ContractInvalidationKind.SOURCE_VERSION in kinds
    assert ContractInvalidationKind.SCHEMA_VERSION in kinds
    source_inv = next(
        i
        for i in source.invalidators
        if i.kind is ContractInvalidationKind.SOURCE_VERSION
    )
    schema_inv = next(
        i
        for i in source.invalidators
        if i.kind is ContractInvalidationKind.SCHEMA_VERSION
    )
    assert source_inv.value == "pkg-3"
    assert schema_inv.value == "iface-2"


def test_evaluate_invalidation_on_version_drift() -> None:
    invs = build_source_invalidators(
        source_version="1.0.0",
        schema_version="a",
        catalog_version=CATALOG_VERSION,
    )
    # No drift.
    assert evaluate_invalidation(
        invs,
        current={
            "source_version": "1.0.0",
            "schema_version": "a",
            "catalog_version": CATALOG_VERSION,
        },
    ) == ()
    # Source version drift.
    matched = evaluate_invalidation(
        invs,
        current={
            "source_version": "1.0.1",
            "schema_version": "a",
            "catalog_version": CATALOG_VERSION,
        },
    )
    assert len(matched) == 1
    assert matched[0].kind is ContractInvalidationKind.SOURCE_VERSION
    # Schema version drift.
    matched = evaluate_invalidation(
        invs,
        current={
            "source_version": "1.0.0",
            "schema_version": "b",
            "catalog_version": CATALOG_VERSION,
        },
    )
    assert len(matched) == 1
    assert matched[0].kind is ContractInvalidationKind.SCHEMA_VERSION


def test_contract_invalidators_complete_after_registration() -> None:
    schema = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="tool:ok",
        source_version="sv-1",
        schema_version="sc-1",
        payload_fingerprint="fp",
    )
    catalog = register_sources_and_contract(
        DEFAULT_MCP_CONTRACT_CATALOG,
        claim_family=McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
        subject="tool:ok",
        sources=(schema,),
        tool_name="ok",
    )
    contract = catalog.contracts[0]
    require_complete_version_invalidators(contract.invalidators)
    assert any(
        i.kind is ContractInvalidationKind.SOURCE_VERSION
        and i.value == "sv-1"
        for i in contract.invalidators
    )
    assert any(
        i.kind is ContractInvalidationKind.SCHEMA_VERSION
        and i.value == "sc-1"
        for i in contract.invalidators
    )


# ---------------------------------------------------------------------------
# Registration / composition
# ---------------------------------------------------------------------------


def test_authoritative_registration_round_trip() -> None:
    idl = make_source_record(
        kind=ContractSourceKind.MCP_IDL,
        subject="tool:list_files",
        source_version="idl-9",
        schema_version=CONTRACT_SCHEMA_VERSION,
        path="idl/list_files.mcp",
        payload_fingerprint="idl-body",
    )
    reg = make_source_record(
        kind=ContractSourceKind.REGISTRATION,
        subject="tool:list_files",
        source_version="pkg-1",
        schema_version=CONTRACT_SCHEMA_VERSION,
        path="mcp/tools.py",
        payload_fingerprint="idl-body",  # same payload → no contradiction
    )
    catalog = register_sources_and_contract(
        DEFAULT_MCP_CONTRACT_CATALOG,
        claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
        subject="tool:list_files",
        sources=(idl, reg),
        tool_name="list_files",
        package_id="ipfs_accelerate_py",
    )
    assert len(catalog.contracts) == 1
    contract = catalog.contracts[0]
    assert contract.review_state is ReviewState.REVIEWED
    assert contract.authority_class is SourceAuthorityClass.AUTHORITATIVE
    assert contract.tool_name == "list_files"
    assert not catalog.contradictions

    payload = catalog.to_dict()
    restored = McpContractCatalog.from_dict(payload)
    assert restored.catalog_id == catalog.catalog_id
    assert restored.contract_ids() == catalog.contract_ids()


def test_seed_builder_is_deterministic() -> None:
    a = build_seed_claim_families()
    b = build_seed_claim_families()
    assert [f.family.value for f in a] == [f.family.value for f in b]
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]


def test_claim_family_descriptor_rejects_unreviewed_seed() -> None:
    with pytest.raises(McpContractCatalogError, match="review_state=reviewed"):
        ClaimFamilyDescriptor(
            family=McpClaimFamily.FAILURE_PARITY,
            review_state=ReviewState.UNREVIEWED,
        )


def test_duplicate_source_admission_fails() -> None:
    source = make_source_record(
        kind=ContractSourceKind.MANIFEST,
        subject="tool:m",
        source_version="1",
    )
    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, source)
    with pytest.raises(McpContractCatalogError, match="already registered"):
        admit_source(catalog, source)


def test_default_singleton_matches_builder() -> None:
    assert (
        DEFAULT_MCP_CONTRACT_CATALOG.catalog_id
        == build_default_mcp_contract_catalog().catalog_id
    )
