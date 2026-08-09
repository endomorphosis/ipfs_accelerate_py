"""DCR-040: finite typed deterministic contract repair operator registry.

Acceptance:
* Unknown fields/operators are rejected before planning.
* Non-invertible or unbounded mutations are rejected at construction.
* Every registered operator declares input schema, exact write scope,
  before/after predicates, preview, inverse, validations, and applicability
  proof.
* Descriptors never grant write/proof/semantic authority or allow arbitrary
  source generation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OPERATOR_REGISTRY_EVIDENCE,
    REPAIR_OPERATOR_INTERFACE,
    REPAIR_OPERATOR_REGISTRY_INTERFACE,
    OperatorDescriptor,
    OperatorFamily,
    OperatorKind,
    OperatorRegistry,
    RepairOperatorRegistryError,
    build_default_operator_registry,
    default_operator_registry_id,
)


def registry() -> OperatorRegistry:
    return build_default_operator_registry()


def _valid_descriptor(**overrides: object) -> OperatorDescriptor:
    values: dict[str, object] = {
        "operator_id": "dcr-operator:add_alias@1",
        "kind": OperatorKind.ADD_ALIAS,
        "family": OperatorFamily.REGISTRY,
        "input_schema_ref": "schema:dcr-operator/add_alias/input@1",
        "write_scope": ("scope:closed_alias_registry",),
        "before_predicates": ("pre:add_alias:applicable", "pre:unique_anchor"),
        "after_predicates": ("post:add_alias:applied", "post:scope_closed"),
        "preview_ref": "preview:add_alias@1",
        "inverse_ref": "inverse:add_alias@1",
        "validation_refs": ("validation:parse", "validation:inverse_roundtrip"),
        "applicability_proof_ref": "proof:applicability:add_alias@1",
        "max_write_paths": 1,
    }
    values.update(overrides)
    return OperatorDescriptor(**values)  # type: ignore[arg-type]


def test_interfaces_and_evidence_identity_are_canonical() -> None:
    reg = registry()
    rebuilt = build_default_operator_registry()
    assert REPAIR_OPERATOR_INTERFACE == "RepairOperator@1"
    assert REPAIR_OPERATOR_REGISTRY_INTERFACE == "RepairOperatorRegistry@1"
    assert reg.INTERFACE == REPAIR_OPERATOR_REGISTRY_INTERFACE
    assert reg.evidence_id == OPERATOR_REGISTRY_EVIDENCE == "dcr/operator-registry@1"
    assert reg.registry_id == rebuilt.registry_id
    assert reg.registry_id == default_operator_registry_id()
    assert OperatorRegistry.from_dict(reg.to_dict()).registry_id == reg.registry_id
    assert tuple(item.operator_id for item in reg.operators) == tuple(
        sorted(item.operator_id for item in reg.operators)
    )


def test_default_catalogue_covers_reviewed_operator_families() -> None:
    reg = registry()
    kinds = set(reg.kinds())
    required = {
        OperatorKind.ADD_ALIAS,
        OperatorKind.REMOVE_ALIAS,
        OperatorKind.RENAME_ALIAS,
        OperatorKind.BIND_REGISTRATION,
        OperatorKind.DISAMBIGUATE_ANCHOR,
        OperatorKind.REPAIR_JSONRPC_SCHEMA,
        OperatorKind.REPAIR_REQUEST_ADAPTER,
        OperatorKind.REPAIR_ERROR_ENVELOPE,
        OperatorKind.REPAIR_PROFILE_BINDING,
        OperatorKind.REPAIR_DISPATCH_BINDING,
        OperatorKind.REPAIR_TRANSPORT_ADAPTER,
        OperatorKind.REPAIR_CAPABILITY_TRUTH,
        OperatorKind.REPAIR_UI_PROJECTION,
        OperatorKind.REPAIR_AUTHORIZATION_GUARD,
        OperatorKind.REGENERATE_PROJECTION,
        OperatorKind.UPDATE_SUBMODULE_PIN,
    }
    assert required == kinds
    assert set(reg.families()) == {
        OperatorFamily.REGISTRY,
        OperatorFamily.PROTOCOL,
        OperatorFamily.DISPATCH,
        OperatorFamily.TRANSPORT,
        OperatorFamily.UI,
        OperatorFamily.SECURITY,
        OperatorFamily.CODEGEN,
        OperatorFamily.ROOT,
    }
    assert len(reg.operators) == len(required)
    assert len(reg.operators) <= 64


def test_every_operator_declares_closed_contract_surface() -> None:
    for descriptor in registry().operators:
        assert isinstance(descriptor, OperatorDescriptor)
        assert descriptor.INTERFACE == REPAIR_OPERATOR_INTERFACE
        assert descriptor.input_schema_ref
        assert descriptor.write_scope
        assert descriptor.before_predicates
        assert descriptor.after_predicates
        assert descriptor.preview_ref
        assert descriptor.inverse_ref
        assert descriptor.validation_refs
        assert descriptor.applicability_proof_ref
        assert descriptor.idempotent is True
        assert descriptor.invertible is True
        assert descriptor.proposal_only is True
        assert descriptor.grants_write_authority is False
        assert descriptor.grants_proof_authority is False
        assert descriptor.semantic_authority is False
        assert descriptor.allows_source_generation is False
        assert descriptor.max_write_paths >= len(descriptor.write_scope)
        assert descriptor.max_write_paths <= 16
        restored = OperatorDescriptor.from_dict(descriptor.to_dict())
        assert restored.content_id == descriptor.content_id
        assert restored.operator_id == descriptor.operator_id


@pytest.mark.parametrize(
    ("alias", "kind"),
    (
        ("register_alias", OperatorKind.ADD_ALIAS),
        ("add_registration", OperatorKind.BIND_REGISTRATION),
        ("unique_anchor", OperatorKind.DISAMBIGUATE_ANCHOR),
        ("jsonrpc_schema", OperatorKind.REPAIR_JSONRPC_SCHEMA),
        ("handler_binding", OperatorKind.REPAIR_DISPATCH_BINDING),
        ("transport_adapter", OperatorKind.REPAIR_TRANSPORT_ADAPTER),
        ("orb_idl_binding", OperatorKind.REPAIR_UI_PROJECTION),
        ("authorization_guard", OperatorKind.REPAIR_AUTHORIZATION_GUARD),
        ("codegen_roundtrip", OperatorKind.REGENERATE_PROJECTION),
        ("submodule_pin", OperatorKind.UPDATE_SUBMODULE_PIN),
    ),
)
def test_aliases_resolve_uniquely(alias: str, kind: OperatorKind) -> None:
    reg = registry()
    assert reg.get(alias).kind is kind
    assert reg.lookup(kind.value).kind is kind
    assert reg.require_known(kind).kind is kind
    assert reg.contains(alias) is True


def test_unknown_operator_is_rejected_before_planning() -> None:
    reg = registry()
    with pytest.raises(RepairOperatorRegistryError, match="unknown operator"):
        reg.get("arbitrary_source_synthesis")
    with pytest.raises(RepairOperatorRegistryError, match="unknown operator"):
        reg.require_known("broad_search_replace")
    with pytest.raises(RepairOperatorRegistryError, match="unknown operator"):
        reg.lookup("dynamic_import_operator")
    assert reg.contains("weaken_tests") is False


def test_unknown_fields_and_forbidden_payloads_fail_closed() -> None:
    base = registry().operators[0].to_dict()
    with pytest.raises(RepairOperatorRegistryError, match="unknown fields"):
        OperatorDescriptor.from_dict({**base, "extra_field": "nope"})
    with pytest.raises(RepairOperatorRegistryError, match="forbidden fields"):
        OperatorDescriptor.from_dict({**base, "source_body": "print('hi')"})
    with pytest.raises(RepairOperatorRegistryError, match="forbidden fields"):
        OperatorDescriptor.from_dict({**base, "shell_fragment": "rm -rf /"})
    reg_payload = registry().to_dict()
    with pytest.raises(RepairOperatorRegistryError, match="unknown fields"):
        OperatorRegistry.from_dict({**reg_payload, "dynamic_hook": True})
    with pytest.raises(RepairOperatorRegistryError, match="cannot claim"):
        OperatorRegistry.from_dict({**reg_payload, "grants_write_authority": True})


def test_non_invertible_and_unbounded_mutations_are_rejected() -> None:
    with pytest.raises(RepairOperatorRegistryError, match="non-invertible"):
        _valid_descriptor(invertible=False)
    with pytest.raises(RepairOperatorRegistryError, match="inverse_ref"):
        _valid_descriptor(inverse_ref="maybe-later")
    with pytest.raises(RepairOperatorRegistryError, match="exact path"):
        _valid_descriptor(write_scope=("pkg/**/*.py",))
    with pytest.raises(RepairOperatorRegistryError, match="exact path|unbounded"):
        _valid_descriptor(write_scope=("*",))
    with pytest.raises(RepairOperatorRegistryError, match="exact path|escape"):
        _valid_descriptor(write_scope=("../escape",))
    with pytest.raises(RepairOperatorRegistryError, match="must not be empty"):
        _valid_descriptor(write_scope=())
    with pytest.raises(RepairOperatorRegistryError, match="unbounded"):
        _valid_descriptor(write_scope=("workspace",))
    with pytest.raises(RepairOperatorRegistryError, match="max_write_paths"):
        _valid_descriptor(
            write_scope=("scope:a", "scope:b"),
            max_write_paths=1,
        )


def test_authority_and_source_generation_flags_cannot_be_enabled() -> None:
    for flag in (
        "grants_write_authority",
        "grants_proof_authority",
        "semantic_authority",
        "allows_source_generation",
        "proposal_only",
        "idempotent",
    ):
        kwargs = {flag: False if flag in {"proposal_only", "idempotent"} else True}
        with pytest.raises(RepairOperatorRegistryError):
            _valid_descriptor(**kwargs)


def test_duplicate_kinds_and_alias_collisions_fail_closed() -> None:
    first = registry().operators[0]
    second = replace(
        first,
        operator_id="dcr-operator:remove_alias@1",
        kind=OperatorKind.REMOVE_ALIAS,
        aliases=("register_alias",),
        input_schema_ref="schema:dcr-operator/remove_alias/input@1",
        preview_ref="preview:remove_alias@1",
        inverse_ref="inverse:remove_alias@1",
        applicability_proof_ref="proof:applicability:remove_alias@1",
        before_predicates=("pre:remove_alias:applicable", "pre:unique_anchor"),
        after_predicates=("post:remove_alias:applied", "post:scope_closed"),
    )
    with pytest.raises(RepairOperatorRegistryError, match="aliases must resolve uniquely"):
        OperatorRegistry(operators=(first, second))
    # Duplicate ids (and therefore kinds) are rejected before planning.
    with pytest.raises(RepairOperatorRegistryError, match="ids must be unique"):
        OperatorRegistry(operators=(first, first))


def test_artifact_projection_is_content_addressed_and_non_authoritative() -> None:
    reg = registry()
    artifact = reg.to_artifact_dict()
    assert artifact["evidence_id"] == "dcr/operator-registry@1"
    assert artifact["operator_count"] == len(reg.operators)
    assert artifact["grants_write_authority"] is False
    assert artifact["allows_source_generation"] is False
    assert artifact["artifact_digest"].startswith("sha256:")
    assert set(artifact["kinds"]) == {item.kind.value for item in reg.operators}
    # Round-trip stability: same catalogue, same digest.
    assert build_default_operator_registry().to_artifact_dict()["artifact_digest"] == (
        artifact["artifact_digest"]
    )


def test_canonical_operator_id_shape_is_enforced() -> None:
    with pytest.raises(RepairOperatorRegistryError, match="canonical"):
        _valid_descriptor(operator_id="operator:add_alias@1")
    with pytest.raises(RepairOperatorRegistryError, match="canonical"):
        _valid_descriptor(operator_id="dcr-operator:add_alias@2")
