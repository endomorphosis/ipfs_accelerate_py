"""Prevent loss of native IR semantics at the shared-envelope boundary.

These are test-only reviewed registry envelopes containing fields shaped like
the native serializers.  They neither export native IR nor authenticate any
production source, reviewer, effect stream, or owner grant.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import (
    IRAdapterRegistry,
    IRAdapterStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRDeclaredAuthority,
    IRFailureCode,
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    VerifiedIRArtifact,
    deterministic_ir_fixture,
)


def _verified(family: IRFamily, updates: dict[str, object]) -> tuple[VerifiedIRArtifact, bytes]:
    reference, encoded = deterministic_ir_fixture(
        family, authority=IRDeclaredAuthority.CONTEXT_ONLY, updates=updates,
    )
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    loaded = registry.load(IRLoadRequest(reference=reference, family=family))
    assert loaded.status is IRLoadStatus.VERIFIED
    assert loaded.artifact is not None
    # Read the verified test object directly; require_artifact mirrors metadata
    # and is unnecessary for this pure normalization-boundary test.
    return loaded.artifact, encoded


_NATIVE_COLLECTIONS = (
    (IRFamily.INTENT, "statements"),
    (IRFamily.INTENT, "actions"),
    (IRFamily.INTENT, "control_edges"),
    (IRFamily.INTENT, "entry_action_ids"),
    (IRFamily.INTENT, "terminal_action_ids"),
    (IRFamily.LEGAL, "rules"),
    (IRFamily.LEGAL, "statements"),
    (IRFamily.LEGAL, "frame_candidates"),
    (IRFamily.SECURITY, "principals"),
    (IRFamily.SECURITY, "assets"),
    (IRFamily.SECURITY, "trust_zones"),
    (IRFamily.SECURITY, "channels"),
    (IRFamily.SECURITY, "resources"),
    (IRFamily.SECURITY, "policies"),
    (IRFamily.SECURITY, "state_machines"),
    (IRFamily.SECURITY, "extensions"),
    (IRFamily.UI, "components"),
    (IRFamily.UI, "entry_components"),
    (IRFamily.UI, "terminal_outcomes"),
    (IRFamily.UI, "composition_edges"),
    (IRFamily.UI, "layout_regions"),
    (IRFamily.UI, "layout_constraints"),
    (IRFamily.UI, "design_token_refs"),
    (IRFamily.UI, "state_variables"),
    (IRFamily.UI, "states"),
    (IRFamily.UI, "events"),
    (IRFamily.UI, "transitions"),
    (IRFamily.UI, "guards"),
    (IRFamily.UI, "effects"),
    (IRFamily.UI, "ux_tasks"),
    (IRFamily.UI, "journeys"),
    (IRFamily.UI, "success_failure_recovery"),
    (IRFamily.UI, "feedback_contracts"),
    (IRFamily.UI, "accessibility"),
    (IRFamily.UI, "localization"),
    (IRFamily.UI, "input_modality_requirements"),
    (IRFamily.UI, "output_modality_requirements"),
    (IRFamily.UI, "modality_alternatives"),
    (IRFamily.UI, "device_capability_requirements"),
    (IRFamily.UI, "adaptive_variants"),
    (IRFamily.UI, "data_bindings"),
    (IRFamily.UI, "content_references"),
    (IRFamily.UI, "program_bindings"),
    (IRFamily.UI, "intent_ir_bindings"),
    (IRFamily.UI, "invocation_bindings"),
    (IRFamily.UI, "mcp_idl_bindings"),
    (IRFamily.UI, "formal_constraint_refs"),
    (IRFamily.UI, "proof_obligation_refs"),
    (IRFamily.UI, "initial_states"),
    (IRFamily.UI, "extensions"),
)


@pytest.mark.parametrize(("family", "field"), _NATIVE_COLLECTIONS)
def test_nonempty_native_collection_is_not_silently_discarded(
    family: IRFamily, field: str,
) -> None:
    artifact, encoded = _verified(family, {
        field: ["native-semantic-record"],
        "declarations": [{"declaration_id": "declaration:mapped-but-incomplete"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.code is IRFailureCode.UNSUPPORTED
    assert result.failure.details == (field,)
    assert result.failure.reason == f"unmapped semantic fields: {field}"
    assert result.fail_closed
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize("family", tuple(IRFamily))
def test_genuinely_empty_shared_fixture_remains_valid(family: IRFamily) -> None:
    artifact, _ = _verified(family, {})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.NORMALIZED
    assert result.artifact is not None
    assert result.artifact.nodes == ()


@pytest.mark.parametrize("family", tuple(IRFamily))
def test_empty_native_collections_and_arbitrary_metadata_do_not_block_shared_envelopes(
    family: IRFamily,
) -> None:
    empty_fields = {field: [] for item_family, field in _NATIVE_COLLECTIONS if item_family is family}
    artifact, encoded = _verified(family, {
        **empty_fields,
        "formulas": [], "cross_view_links": [],
        "symbol_table": {"table_id": "symbols:empty", "symbols": [], "metadata": {"note": "empty"}},
        "view_registry": {"registry_id": "views:empty", "views": []},
        "frame_logic": {"ontology_name": "modal_flogic_ir", "triples": [], "selected_frame": None},
        "metadata": {"policies": ["metadata-only"], "formulas": ["metadata-only"]},
        "producer_note": {"actions": ["metadata-only"]},
        "declarations": [{"declaration_id": "declaration:retained", "kind": "statement"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.NORMALIZED
    assert result.artifact is not None
    assert [node.node_id for node in result.artifact.declarations] == ["declaration:retained"]
    assert not result.artifact.grants_execution_authority
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize("family", (
    IRFamily.FORMALIZATION, IRFamily.INTENT, IRFamily.LEGAL, IRFamily.SECURITY, IRFamily.UI,
))
@pytest.mark.parametrize(("field", "value", "path"), (
    ("formulas", [{"formula_id": "formula:one", "expression": "P(x)"}], "formulas"),
    ("cross_view_links", [{"link_id": "link:one"}], "cross_view_links"),
    ("symbol_table", {"table_id": "symbols:one", "symbols": [{"symbol_id": "symbol:P"}]}, "symbol_table.symbols"),
    ("view_registry", {"registry_id": "views:one", "views": [{"view_id": "view:fol"}]}, "view_registry.views"),
))
def test_native_formalization_sections_fail_even_beside_supported_sections(
    family: IRFamily, field: str, value: object, path: str,
) -> None:
    artifact, encoded = _verified(family, {
        field: value,
        "formal_views": [{"view_id": "view:shared", "view_kind": "first_order"}],
        "declarations": [{"declaration_id": "declaration:shared"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (path,)
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize(("body", "fields"), (
    ({"statements": [{"statement_id": "statement:goal", "kind": "goal", "modality": "intended",
                      "normalized_text": "Train the LegalIR model.", "review_status": "machine_extracted",
                      "source_ref_ids": ["source:instruction"]}],
      "actions": [{"action_id": "action:request", "actor": "user", "verb": "request",
                   "effect_ids": [], "source_ref_ids": ["source:instruction"]}],
      "control_edges": [], "entry_action_ids": ["action:request"], "terminal_action_ids": ["action:request"]},
     ("actions", "entry_action_ids", "statements", "terminal_action_ids")),
))
def test_native_prompt_intent_is_refused_without_inventing_execution_effects(
    body: dict[str, object], fields: tuple[str, ...],
) -> None:
    artifact, encoded = _verified(IRFamily.INTENT, {
        **body,
        "sources": [{"ref_id": "source:instruction", "review_status": "unreviewed",
                     "content_sha256": "0" * 64}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.failure is not None
    assert result.failure.details == fields
    assert result.artifact is None
    assert artifact.payload["actions"][0]["effect_ids"] == ()
    assert artifact.payload["sources"][0]["review_status"] == "unreviewed"
    assert artifact.payload["statements"][0]["review_status"] == "machine_extracted"
    assert artifact.canonical_bytes == encoded


def test_native_security_policy_and_state_machine_cannot_be_mistaken_for_empty_policy() -> None:
    artifact, encoded = _verified(IRFamily.SECURITY, {
        "declaration_id": "security:access",
        "principals": [{"principal_id": "principal:operator", "kind": "human", "role_ids": ["role:operator"],
                        "trust_zone_ids": [], "source_ids": ["source:security"], "attributes": {}}],
        "resources": [{"resource_id": "resource:file", "kind": "file", "owner_principal_ids": [],
                       "asset_ids": [], "trust_zone_ids": [], "source_ids": ["source:security"],
                       "attributes": {}}],
        "policies": [{"policy_id": "policy:access", "name": "read-access", "effect": "allow",
                      "principal_ids": ["principal:operator"], "resource_ids": ["resource:file"],
                      "channel_ids": [], "source_ids": ["source:security"],
                      "attributes": {"condition": "authenticated"}}],
        "state_machines": [{"state_machine_id": "state:session", "states": ["closed", "open"],
                            "initial_state": "closed", "transitions": [],
                            "source_ids": ["source:security"], "attributes": {}}],
        "sources": [{"source_id": "source:security", "review_status": "unreviewed",
                     "uri": "source:security-policy", "revision": "1", "content_sha256": "0" * 64,
                     "attributes": {}}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == ("policies", "principals", "resources", "state_machines")
    assert artifact.payload["sources"][0]["review_status"] == "unreviewed"
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize(("field", "value", "path"), (
    ("modal_ir", {"document_id": "legal:one", "formulas": [{"formula_id": "norm:one"}]}, "modal_ir"),
    ("frame_logic", {"triples": [{"subject": "agency", "predicate": "must", "object": "retain"}]}, "frame_logic.triples"),
    ("frame_logic", {"triples": [], "selected_frame": "frame:obligation"}, "frame_logic.selected_frame"),
    ("frame_logic", {"neo4j_node_labels": ["Actor"]}, "frame_logic.neo4j_node_labels"),
    ("frame_logic", {"neo4j_relationship_types": ["OBLIGATION"]}, "frame_logic.neo4j_relationship_types"),
))
def test_legacy_legal_semantics_are_not_lost(field: str, value: object, path: str) -> None:
    artifact, encoded = _verified(IRFamily.LEGAL, {field: value})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (path,)
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize(("canonical", "alias", "node_id_field", "section"), (
    ("formal_views", "views", "view_id", "formal_views"),
    ("formal_views", "logical_views", "view_id", "formal_views"),
    ("obligations", "proof_obligations", "obligation_id", "obligations"),
    ("result_authority", "result_authorities", "result_id", "result_authority"),
))
def test_one_selected_alias_remains_supported(
    canonical: str, alias: str, node_id_field: str, section: str,
) -> None:
    artifact, encoded = _verified(IRFamily.LEGAL, {
        canonical: None, alias: [{node_id_field: "node:alias"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.NORMALIZED
    assert result.artifact is not None
    assert [node.node_id for node in getattr(result.artifact, section)] == ["node:alias"]
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize(("canonical", "alias", "id_field"), (
    ("formal_views", "views", "view_id"),
    ("formal_views", "logical_views", "view_id"),
    ("obligations", "proof_obligations", "obligation_id"),
    ("result_authority", "result_authorities", "result_id"),
))
@pytest.mark.parametrize("canonical_nonempty", (False, True))
def test_nonempty_shadowed_alias_is_refused_including_identical_records(
    canonical: str, alias: str, id_field: str, canonical_nonempty: bool,
) -> None:
    records = [{id_field: "node:same"}]
    artifact, encoded = _verified(IRFamily.LEGAL, {
        canonical: records if canonical_nonempty else [], alias: records,
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (alias,)
    assert artifact.canonical_bytes == encoded


def test_first_empty_alias_cannot_hide_a_second_nonempty_alias() -> None:
    artifact, _ = _verified(IRFamily.LEGAL, {
        "formal_views": None, "views": [], "logical_views": [{"view_id": "view:lost"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.failure is not None
    assert result.failure.details == ("logical_views",)


def test_optional_native_input_retains_typed_refusal_without_claiming_completion() -> None:
    artifact, _ = _verified(IRFamily.SECURITY, {"policies": [{"policy_id": "policy:one"}]})
    result = IRAdapterRegistry().normalize(artifact, required=False)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert not result.fail_closed
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == ("policies",)


def test_guard_is_family_specific_not_an_arbitrary_field_denylist() -> None:
    artifact, _ = _verified(IRFamily.INTENT, {
        "metadata": {"policies": [{"policy_id": "metadata:one"}]},
        "policies": [{"policy_id": "producer:extra-not-an-intent-native-field"}],
        "declarations": [{"declaration_id": "declaration:one"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.NORMALIZED
    assert result.artifact is not None
    assert [node.node_id for node in result.artifact.declarations] == ["declaration:one"]


@pytest.mark.parametrize(("family", "field"), (
    (IRFamily.INTENT, "actions"),
    (IRFamily.LEGAL, "rules"),
    (IRFamily.SECURITY, "policies"),
    (IRFamily.UI, "components"),
    (IRFamily.FORMALIZATION, "formulas"),
    (IRFamily.FORMALIZATION, "cross_view_links"),
))
@pytest.mark.parametrize("malformed", (False, 0, ""))
def test_falsy_scalar_native_collection_cannot_masquerade_as_empty(
    family: IRFamily, field: str, malformed: object,
) -> None:
    artifact, encoded = _verified(family, {
        field: malformed, "declarations": [{"declaration_id": "declaration:shared"}],
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (field,)
    assert artifact.canonical_bytes == encoded


@pytest.mark.parametrize(("family", "parent"), (
    (IRFamily.FORMALIZATION, "symbol_table"),
    (IRFamily.FORMALIZATION, "view_registry"),
    (IRFamily.LEGAL, "frame_logic"),
))
@pytest.mark.parametrize("malformed", (False, 0, ""))
def test_falsy_scalar_nested_native_parent_cannot_masquerade_as_empty(
    family: IRFamily, parent: str, malformed: object,
) -> None:
    artifact, _ = _verified(family, {parent: malformed})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (parent,)


@pytest.mark.parametrize(("family", "parent", "child"), (
    (IRFamily.FORMALIZATION, "symbol_table", "symbols"),
    (IRFamily.FORMALIZATION, "view_registry", "views"),
    (IRFamily.LEGAL, "frame_logic", "triples"),
))
@pytest.mark.parametrize("malformed", (False, 0, ""))
def test_falsy_scalar_nested_native_collection_cannot_masquerade_as_empty(
    family: IRFamily, parent: str, child: str, malformed: object,
) -> None:
    artifact, _ = _verified(family, {parent: {child: malformed}})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (f"{parent}.{child}",)


@pytest.mark.parametrize(("canonical", "alias"), (
    ("formal_views", "views"),
    ("formal_views", "logical_views"),
    ("obligations", "proof_obligations"),
    ("result_authority", "result_authorities"),
))
@pytest.mark.parametrize("malformed", (False, 0, ""))
def test_falsy_scalar_shadowed_alias_cannot_be_discarded(
    canonical: str, alias: str, malformed: object,
) -> None:
    artifact, _ = _verified(IRFamily.LEGAL, {canonical: [], alias: malformed})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.details == (alias,)


@pytest.mark.parametrize("empty", (None, {}, [], ()))
def test_actual_empty_native_fields_remain_ignorable(empty: object) -> None:
    artifact, _ = _verified(IRFamily.FORMALIZATION, {
        "formulas": empty, "cross_view_links": empty,
        "symbol_table": {"symbols": empty}, "view_registry": {"views": empty},
    })
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.NORMALIZED
    assert result.artifact is not None
    assert result.artifact.nodes == ()


@pytest.mark.parametrize("malformed", (False, 0, ""))
def test_selected_malformed_alias_still_reaches_existing_section_validation(malformed: object) -> None:
    artifact, _ = _verified(IRFamily.LEGAL, {"formal_views": None, "views": malformed})
    result = IRAdapterRegistry().normalize(artifact)
    assert result.status is IRAdapterStatus.PARTIAL
    assert result.artifact is None
