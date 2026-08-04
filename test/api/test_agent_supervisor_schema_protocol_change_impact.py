"""Tests for schema/constructor/serialization/protocol impact analysis (RPR-030)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ConsumerDisposition,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.schema_protocol_change_impact import (
    AuthorityKind,
    CompatibilityDirection,
    ConstructionKind,
    ConstructorImpact,
    FieldChangeKind,
    ProtocolImpact,
    SCHEMA_PROTOCOL_IMPACT_SCHEMA,
    SchemaConsumerObservation,
    SchemaConsumerRole,
    SchemaFieldChange,
    SchemaProtocolChangeAnalyzer,
    SchemaProtocolChangeImpact,
    SchemaProtocolChangeImpactAuthorityError,
    SchemaProtocolChangeImpactError,
    SchemaProtocolImpact,
    SchemaSurfaceKind,
    SerializationFacet,
    SerializationImpact,
    WriteMode,
    all_compatibility_directions,
    all_field_change_kinds,
    all_serialization_facets,
    build_schema_protocol_impact,
    classify_field_compatibility,
    extract_field_changes,
    required_consumer_roles,
    required_schema_surfaces,
)


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def _clause(
    *,
    clause_id: str = "clause:schema-1",
    kind: DeltaKind = DeltaKind.SCHEMA_CHANGE,
    disposition: DeltaDisposition = DeltaDisposition.BREAKING,
    subject: str = "symbol:Order",
    reason: str = "added required field=context",
    before: str = "contract:Order{id:str}",
    after: str = "contract:Order{id:str,context:str}",
) -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id=clause_id,
        kind=kind,
        disposition=disposition,
        subject_symbol_id=subject,
        consumer_domain="domain:schema-consumers",
        before_contract_ref=before,
        after_contract_ref=after,
        reason=reason,
    )


def _delta(
    roots: PropagationAuthorityRoots,
    *clauses: ContractClauseDelta,
    subject: str = "symbol:Order",
) -> ProgramContractDelta:
    if not clauses:
        clauses = (_clause(subject=subject),)
    return ProgramContractDelta(
        roots=roots,
        change_set_id="changeset:order-schema",
        subject_symbol_id=subject,
        before_contract_ref="contract:Order@1",
        after_contract_ref="contract:Order@2",
        clauses=clauses,
        evidence_refs=("evidence:schema-extract",),
    )


def _node(consumer_id: str, path: str, symbol_id: str, *, kind: str = "schema") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{consumer_id}",
        kind=kind,
        path=path,
        symbol_id=symbol_id,
        artifact_id=f"blob:{consumer_id}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:test",
    )


def _consumer(
    *,
    consumer_id: str,
    role: SchemaConsumerRole | str,
    path: str,
    symbol_id: str,
    subject: str = "symbol:Order",
    surface: SchemaSurfaceKind | str | None = None,
    construction_kind: ConstructionKind | str | None = None,
    serialization_facet: SerializationFacet | str | None = None,
    codec_status: str = "present",
    generated: bool = False,
    read_only: bool = False,
    supplies: tuple[str, ...] = (),
    ignores_unknown: bool = False,
    accepts_missing_optional: bool = True,
    authority_kinds: tuple[str, ...] = (),
    authority_refs: tuple[str, ...] = (),
) -> SchemaConsumerObservation:
    return SchemaConsumerObservation(
        consumer_id=consumer_id,
        role=role,
        path=path,
        symbol_id=symbol_id,
        subject_symbol_id=subject,
        surface=surface,
        construction_kind=construction_kind,
        serialization_facet=serialization_facet,
        codec_status=codec_status,
        generated=generated,
        read_only=read_only,
        supplies_field_names=supplies,
        ignores_unknown_fields=ignores_unknown,
        accepts_missing_optional=accepts_missing_optional,
        authority_kinds=authority_kinds,
        authority_refs=authority_refs,
        node=_node(consumer_id, path, symbol_id),
        evidence_refs=(f"evidence:{consumer_id}",),
    )


def _field(
    kind: FieldChangeKind | str,
    name: str,
    *,
    previous_name: str = "",
    previous_type: str = "",
    type_ref: str = "",
    required: bool = False,
    has_default: bool = False,
    default_ref: str = "",
    default_authority: AuthorityKind | str = AuthorityKind.NONE,
    variant: bool = False,
    clause_ids: tuple[str, ...] = ("clause:schema-1",),
) -> SchemaFieldChange:
    return SchemaFieldChange(
        kind=kind,
        field_name=name,
        previous_name=previous_name,
        previous_type_ref=previous_type,
        type_ref=type_ref,
        required=required,
        has_default=has_default,
        default_ref=default_ref or ("default:x" if has_default else ""),
        default_authority=default_authority,
        variant=variant,
        clause_ids=clause_ids,
        reason=f"{kind}:{name}",
    )


# ---------------------------------------------------------------------------
# Catalogue / vocabulary
# ---------------------------------------------------------------------------


def test_required_roles_and_surfaces_cover_acceptance_catalogue() -> None:
    roles = required_consumer_roles()
    assert roles == {
        "field_reader",
        "field_writer",
        "constructor",
        "factory",
        "builder",
        "serializer",
        "deserializer",
        "persistence",
        "cache_key",
        "equality_hash",
        "version_negotiation",
        "migration",
        "generated_client",
        "schema_surface",
        "protocol_surface",
        "dynamic_codec",
        "missing_codec",
    }
    surfaces = required_schema_surfaces()
    assert surfaces == {
        "json",
        "protobuf",
        "idl",
        "database",
        "message",
        "rpc",
        "http",
        "cli",
    }
    kinds = {item.value for item in all_field_change_kinds()}
    assert kinds == {
        "added",
        "removed",
        "renamed",
        "retyped",
        "variant_added",
        "variant_removed",
        "variant_renamed",
    }
    facets = {item.value for item in all_serialization_facets()}
    assert "serializer" in facets
    assert "deserializer" in facets
    assert "persistence" in facets
    assert "cache_key" in facets
    assert "equality" in facets
    assert "hash" in facets
    assert "version_negotiation" in facets
    assert "migration" in facets
    assert "generated_client" in facets
    directions = {item.value for item in all_compatibility_directions()}
    assert directions == {
        "backward",
        "forward",
        "full",
        "incompatible",
        "unknown",
    }


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("json_schema", SchemaSurfaceKind.JSON),
        ("proto", SchemaSurfaceKind.PROTOBUF),
        ("grpc", SchemaSurfaceKind.RPC),
        ("sql", SchemaSurfaceKind.DATABASE),
        ("rest", SchemaSurfaceKind.HTTP),
        ("argparse", SchemaSurfaceKind.CLI),
        ("avro", SchemaSurfaceKind.IDL),
    ],
)
def test_surface_aliases(alias: str, expected: SchemaSurfaceKind) -> None:
    assert SchemaSurfaceKind.coerce(alias) is expected


# ---------------------------------------------------------------------------
# Field change detection
# ---------------------------------------------------------------------------


def test_detects_added_removed_renamed_retyped_fields_and_variants(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(
            clause_id="clause:add",
            reason="added required field=context",
        ),
        _clause(
            clause_id="clause:remove",
            reason="removed field=legacy_id",
        ),
        _clause(
            clause_id="clause:rename",
            reason="renamed field old_name -> new_name",
        ),
        _clause(
            clause_id="clause:retype",
            reason="retyped field amount: int -> Decimal",
        ),
        _clause(
            clause_id="clause:var-add",
            reason="variant added field=case_express",
        ),
        _clause(
            clause_id="clause:var-remove",
            reason="variant removed field=case_legacy",
        ),
    )
    explicit = [
        _field(FieldChangeKind.VARIANT_RENAMED, "case_b", previous_name="case_a", variant=True),
    ]
    changes = extract_field_changes(delta, explicit)
    kinds = {(item.kind, item.field_name) for item in changes}
    assert (FieldChangeKind.ADDED, "context") in kinds
    assert (FieldChangeKind.REMOVED, "legacy_id") in kinds
    assert (FieldChangeKind.RENAMED, "new_name") in kinds
    assert (FieldChangeKind.RETYPED, "amount") in kinds
    assert (FieldChangeKind.VARIANT_ADDED, "case_express") in kinds
    assert (FieldChangeKind.VARIANT_REMOVED, "case_legacy") in kinds
    assert (FieldChangeKind.VARIANT_RENAMED, "case_b") in kinds

    renamed = next(item for item in changes if item.kind is FieldChangeKind.RENAMED)
    assert renamed.previous_name == "old_name"
    retyped = next(item for item in changes if item.kind is FieldChangeKind.RETYPED)
    assert "int" in retyped.previous_type_ref
    assert "Decimal" in retyped.type_ref


def test_field_change_round_trip_identity() -> None:
    change = _field(
        FieldChangeKind.ADDED,
        "context",
        required=True,
        has_default=True,
        default_authority=AuthorityKind.REVIEWED_IDL,
    )
    assert SchemaFieldChange.from_dict(change.to_dict()) == change
    assert change.default_has_independent_authority is True


# ---------------------------------------------------------------------------
# Compatibility directions per consumer role
# ---------------------------------------------------------------------------


def test_compatibility_directions_per_field_change() -> None:
    required_add = _field(FieldChangeKind.ADDED, "context", required=True)
    optional_add = _field(FieldChangeKind.ADDED, "note", required=False)
    removed = _field(FieldChangeKind.REMOVED, "legacy")
    renamed = _field(FieldChangeKind.RENAMED, "new", previous_name="old")
    retyped = _field(
        FieldChangeKind.RETYPED, "amount", previous_type="int", type_ref="str"
    )

    direction, reasons, _ = classify_field_compatibility(
        required_add, SchemaConsumerRole.CONSTRUCTOR
    )
    assert direction is CompatibilityDirection.INCOMPATIBLE
    assert any("required_field_added" in item for item in reasons)

    direction, _, _ = classify_field_compatibility(
        optional_add, SchemaConsumerRole.FIELD_READER
    )
    assert direction is CompatibilityDirection.BACKWARD

    direction, _, _ = classify_field_compatibility(
        removed, SchemaConsumerRole.DESERIALIZER
    )
    assert direction is CompatibilityDirection.INCOMPATIBLE

    direction, _, _ = classify_field_compatibility(
        renamed, SchemaConsumerRole.SERIALIZER
    )
    assert direction is CompatibilityDirection.INCOMPATIBLE

    direction, _, _ = classify_field_compatibility(
        retyped, SchemaConsumerRole.CACHE_KEY
    )
    assert direction is CompatibilityDirection.INCOMPATIBLE

    direction, _, _ = classify_field_compatibility(
        _field(FieldChangeKind.VARIANT_ADDED, "v2", variant=True),
        SchemaConsumerRole.FIELD_READER,
        ignores_unknown_fields=True,
    )
    assert direction is CompatibilityDirection.FULL


def test_distinguish_backward_forward_full_incompatible_unknown_per_consumer(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(reason="added optional field=note"),
    )
    field_changes = [_field(FieldChangeKind.ADDED, "note", required=False)]
    consumers = [
        _consumer(
            consumer_id="consumer:reader",
            role=SchemaConsumerRole.FIELD_READER,
            path="src/read.py",
            symbol_id="symbol:read",
        ),
        _consumer(
            consumer_id="consumer:writer",
            role=SchemaConsumerRole.FIELD_WRITER,
            path="src/write.py",
            symbol_id="symbol:write",
            supplies=("note",),
        ),
        _consumer(
            consumer_id="consumer:ctor-break",
            role=SchemaConsumerRole.CONSTRUCTOR,
            path="src/model.py",
            symbol_id="symbol:Order.__init__",
            # Required add without supply → incompatible (separate field).
        ),
        _consumer(
            consumer_id="consumer:dynamic",
            role=SchemaConsumerRole.DYNAMIC_CODEC,
            path="src/dyn_codec.py",
            symbol_id="symbol:dyn",
            codec_status="dynamic",
        ),
    ]
    # Mix optional add for most + required for constructor via second change set.
    impact = build_schema_protocol_impact(
        delta,
        consumers[:2] + [consumers[3]],
        field_changes=field_changes,
    )
    by_id = {item.observation.consumer_id: item for item in impact.entries}
    assert by_id["consumer:reader"].compatibility is CompatibilityDirection.BACKWARD
    assert by_id["consumer:writer"].compatibility in {
        CompatibilityDirection.FULL,
        CompatibilityDirection.FORWARD,
    }
    assert by_id["consumer:dynamic"].compatibility is CompatibilityDirection.UNKNOWN

    required_delta = _delta(
        roots,
        _clause(reason="added required field=context"),
    )
    required_impact = build_schema_protocol_impact(
        required_delta,
        [
            _consumer(
                consumer_id="consumer:ctor",
                role=SchemaConsumerRole.CONSTRUCTOR,
                path="src/model.py",
                symbol_id="symbol:Order.__init__",
            ),
            _consumer(
                consumer_id="consumer:already",
                role=SchemaConsumerRole.CONSTRUCTOR,
                path="src/model_v2.py",
                symbol_id="symbol:OrderV2.__init__",
                supplies=("context",),
            ),
        ],
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    by_id = {
        item.observation.consumer_id: item for item in required_impact.entries
    }
    assert by_id["consumer:ctor"].compatibility is CompatibilityDirection.INCOMPATIBLE
    assert by_id["consumer:already"].compatibility is CompatibilityDirection.FULL
    assert by_id["consumer:ctor"].disposition is ConsumerDisposition.MIGRATE
    assert by_id["consumer:already"].disposition is ConsumerDisposition.COMPATIBLE
    # One compatible cannot discharge the other.
    assert required_impact.one_compatible_cannot_discharge_others()
    assert len(required_impact.migrate_entries) == 1
    assert len(required_impact.compatible_entries) == 1


# ---------------------------------------------------------------------------
# Constructor / factory / builder
# ---------------------------------------------------------------------------


def test_constructor_factory_builder_impacts(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots, _clause(reason="added required field=context"))
    impact = SchemaProtocolChangeAnalyzer().analyze(
        delta,
        [
            _consumer(
                consumer_id="consumer:ctor",
                role=SchemaConsumerRole.CONSTRUCTOR,
                path="src/order.py",
                symbol_id="symbol:Order.__init__",
                construction_kind=ConstructionKind.CONSTRUCTOR,
            ),
            _consumer(
                consumer_id="consumer:factory",
                role=SchemaConsumerRole.FACTORY,
                path="src/factories.py",
                symbol_id="symbol:create_order",
                construction_kind=ConstructionKind.FACTORY,
            ),
            _consumer(
                consumer_id="consumer:builder",
                role=SchemaConsumerRole.BUILDER,
                path="src/builder.py",
                symbol_id="symbol:OrderBuilder.build",
                construction_kind=ConstructionKind.BUILDER,
            ),
        ],
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    assert len(impact.constructor_impacts) == 3
    kinds = {item.kind for item in impact.constructor_impacts}
    assert kinds == {
        ConstructionKind.CONSTRUCTOR,
        ConstructionKind.FACTORY,
        ConstructionKind.BUILDER,
    }
    for item in impact.constructor_impacts:
        assert isinstance(item, ConstructorImpact)
        assert "context" in item.added_required_fields
        assert item.compatibility is CompatibilityDirection.INCOMPATIBLE
        assert item.subject_symbol_id
    for entry in impact.entries:
        assert entry.disposition is ConsumerDisposition.MIGRATE
        assert entry.obligation is not None
        assert entry.obligation.missing_input_ids
        assert "missing-field:context" in entry.obligation.missing_input_ids


# ---------------------------------------------------------------------------
# Serialization facets
# ---------------------------------------------------------------------------


def test_serializers_persistence_cache_equality_version_migration_generated(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(
            kind=DeltaKind.SERIALIZATION_CHANGE,
            reason="retyped field amount: int -> Decimal",
        ),
    )
    field_changes = [
        _field(
            FieldChangeKind.RETYPED,
            "amount",
            previous_type="int",
            type_ref="Decimal",
        )
    ]
    consumers = [
        _consumer(
            consumer_id="consumer:ser",
            role=SchemaConsumerRole.SERIALIZER,
            path="src/codecs.py",
            symbol_id="symbol:OrderEncoder",
            serialization_facet=SerializationFacet.SERIALIZER,
        ),
        _consumer(
            consumer_id="consumer:de",
            role=SchemaConsumerRole.DESERIALIZER,
            path="src/codecs.py",
            symbol_id="symbol:OrderDecoder",
            serialization_facet=SerializationFacet.DESERIALIZER,
        ),
        _consumer(
            consumer_id="consumer:persist",
            role=SchemaConsumerRole.PERSISTENCE,
            path="src/store.py",
            symbol_id="symbol:OrderStore",
            serialization_facet=SerializationFacet.PERSISTENCE,
        ),
        _consumer(
            consumer_id="consumer:cache",
            role=SchemaConsumerRole.CACHE_KEY,
            path="src/cache.py",
            symbol_id="symbol:order_cache_key",
            serialization_facet=SerializationFacet.CACHE_KEY,
        ),
        _consumer(
            consumer_id="consumer:eq",
            role=SchemaConsumerRole.EQUALITY_HASH,
            path="src/model.py",
            symbol_id="symbol:Order.__eq__",
            serialization_facet=SerializationFacet.EQUALITY,
        ),
        _consumer(
            consumer_id="consumer:ver",
            role=SchemaConsumerRole.VERSION_NEGOTIATION,
            path="src/versioning.py",
            symbol_id="symbol:negotiate_order_version",
            serialization_facet=SerializationFacet.VERSION_NEGOTIATION,
        ),
        _consumer(
            consumer_id="consumer:mig",
            role=SchemaConsumerRole.MIGRATION,
            path="src/migrations/v2.py",
            symbol_id="symbol:migrate_order_v2",
            serialization_facet=SerializationFacet.MIGRATION,
        ),
        _consumer(
            consumer_id="consumer:gen",
            role=SchemaConsumerRole.GENERATED_CLIENT,
            path="generated/clients/order_pb2.py",
            symbol_id="symbol:generated.Order",
            serialization_facet=SerializationFacet.GENERATED_CLIENT,
            generated=True,
        ),
    ]
    impact = build_schema_protocol_impact(
        delta, consumers, field_changes=field_changes
    )
    assert len(impact.serialization_impacts) == len(consumers)
    facets = {item.facet for item in impact.serialization_impacts}
    assert SerializationFacet.SERIALIZER in facets
    assert SerializationFacet.DESERIALIZER in facets
    assert SerializationFacet.PERSISTENCE in facets
    assert SerializationFacet.CACHE_KEY in facets
    assert SerializationFacet.MIGRATION in facets
    assert SerializationFacet.GENERATED_CLIENT in facets

    by_id = {item.observation.consumer_id: item for item in impact.entries}
    for consumer_id in (
        "consumer:ser",
        "consumer:de",
        "consumer:persist",
        "consumer:cache",
        "consumer:eq",
    ):
        assert by_id[consumer_id].compatibility is CompatibilityDirection.INCOMPATIBLE
        assert by_id[consumer_id].disposition is ConsumerDisposition.MIGRATE

    gen = by_id["consumer:gen"]
    assert gen.write_mode is WriteMode.REGENERATION
    assert gen.disposition is ConsumerDisposition.MIGRATE
    assert gen.serialization_impact is not None
    assert gen.serialization_impact.write_mode is WriteMode.REGENERATION

    mig = by_id["consumer:mig"]
    assert AuthorityKind.MIGRATION_MANIFEST in mig.required_authority or any(
        "migration" in code for code in mig.reason_codes
    )


# ---------------------------------------------------------------------------
# Protocol surfaces
# ---------------------------------------------------------------------------


def test_json_protobuf_idl_database_message_rpc_http_cli_surfaces(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(
            kind=DeltaKind.PROTOCOL_CHANGE,
            reason="added required field=tenant_id",
        ),
    )
    field_changes = [_field(FieldChangeKind.ADDED, "tenant_id", required=True)]
    surface_specs = [
        ("json", SchemaSurfaceKind.JSON, "schemas/order.json"),
        ("protobuf", SchemaSurfaceKind.PROTOBUF, "proto/order.proto"),
        ("idl", SchemaSurfaceKind.IDL, "idl/order.thrift"),
        ("database", SchemaSurfaceKind.DATABASE, "db/migrations/002_order.sql"),
        ("message", SchemaSurfaceKind.MESSAGE, "events/order_created.avsc"),
        ("rpc", SchemaSurfaceKind.RPC, "rpc/order_service.proto"),
        ("http", SchemaSurfaceKind.HTTP, "openapi/order.yaml"),
        ("cli", SchemaSurfaceKind.CLI, "cli/order_cmd.py"),
    ]
    consumers = [
        _consumer(
            consumer_id=f"consumer:{name}",
            role=SchemaConsumerRole.PROTOCOL_SURFACE,
            path=path,
            symbol_id=f"symbol:{name}.Order",
            surface=surface,
        )
        for name, surface, path in surface_specs
    ]
    impact = build_schema_protocol_impact(
        delta, consumers, field_changes=field_changes
    )
    assert len(impact.protocol_impacts) == 8
    surfaces = {item.surface for item in impact.protocol_impacts}
    assert surfaces == set(required_schema_surfaces())
    for item in impact.protocol_impacts:
        assert isinstance(item, ProtocolImpact)
        assert item.compatibility is CompatibilityDirection.INCOMPATIBLE
        assert "tenant_id" in item.affected_field_names
    assert all(
        entry.disposition is ConsumerDisposition.MIGRATE for entry in impact.entries
    )


# ---------------------------------------------------------------------------
# Authority: defaults and migrations
# ---------------------------------------------------------------------------


def test_required_defaults_need_independent_authority(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(reason="added field=region with default without reviewed authority"),
    )
    # Default present but no independent authority.
    unauth = _field(
        FieldChangeKind.ADDED,
        "region",
        required=False,
        has_default=True,
        default_authority=AuthorityKind.NONE,
    )
    impact = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:ctor-unauth",
                role=SchemaConsumerRole.CONSTRUCTOR,
                path="src/model.py",
                symbol_id="symbol:Order.__init__",
            )
        ],
        field_changes=[unauth],
    )
    entry = impact.entries[0]
    assert entry.compatibility in {
        CompatibilityDirection.UNKNOWN,
        CompatibilityDirection.FORWARD,
        CompatibilityDirection.FULL,
    }
    # Without authority, constructor path requires independent default authority.
    assert (
        AuthorityKind.SCHEMA_DEFAULT in entry.required_authority
        or "required_default_needs_independent_authority" in entry.reason_codes
        or entry.constructor_impact is not None
        and entry.constructor_impact.needs_independent_default_authority
    )
    assert entry.disposition is not ConsumerDisposition.COMPATIBLE or not entry.required_authority

    # With reviewed IDL authority, default may discharge.
    auth = _field(
        FieldChangeKind.ADDED,
        "region",
        required=False,
        has_default=True,
        default_authority=AuthorityKind.REVIEWED_IDL,
    )
    impact_auth = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:ctor-auth",
                role=SchemaConsumerRole.CONSTRUCTOR,
                path="src/model.py",
                symbol_id="symbol:Order.__init__",
                authority_kinds=(AuthorityKind.REVIEWED_IDL,),
                authority_refs=("idl:order@2",),
            )
        ],
        field_changes=[auth],
    )
    auth_entry = impact_auth.entries[0]
    assert auth_entry.disposition is ConsumerDisposition.COMPATIBLE
    assert not auth_entry.required_authority


def test_migrations_need_independent_authority(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(reason="removed field=legacy_id"),
    )
    impact = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:migration",
                role=SchemaConsumerRole.MIGRATION,
                path="src/migrations/drop_legacy.py",
                symbol_id="symbol:drop_legacy_id",
            )
        ],
        field_changes=[_field(FieldChangeKind.REMOVED, "legacy_id")],
    )
    entry = impact.entries[0]
    assert AuthorityKind.MIGRATION_MANIFEST in entry.required_authority
    assert entry.disposition is ConsumerDisposition.MIGRATE
    assert entry.obligation is not None
    assert any("migration" in item for item in entry.obligation.behavior_contract_ids)
    assert entry.serialization_impact is not None
    assert entry.serialization_impact.needs_migration_authority is True

    # With migration manifest authority, remaining authority is cleared.
    impact_auth = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:migration-auth",
                role=SchemaConsumerRole.MIGRATION,
                path="src/migrations/drop_legacy.py",
                symbol_id="symbol:drop_legacy_id",
                authority_kinds=(AuthorityKind.MIGRATION_MANIFEST,),
                authority_refs=("manifest:migrations@1",),
            )
        ],
        field_changes=[_field(FieldChangeKind.REMOVED, "legacy_id")],
    )
    # Still migrate (incompatible shape) but authority requirement is satisfied.
    auth_entry = impact_auth.entries[0]
    assert AuthorityKind.MIGRATION_MANIFEST not in auth_entry.required_authority
    assert auth_entry.disposition is ConsumerDisposition.MIGRATE


# ---------------------------------------------------------------------------
# Generated / read-only roots
# ---------------------------------------------------------------------------


def test_generated_and_read_only_produce_regeneration_or_external(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots, _clause(reason="added required field=context"))
    field_changes = [_field(FieldChangeKind.ADDED, "context", required=True)]
    impact = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:generated",
                role=SchemaConsumerRole.GENERATED_CLIENT,
                path="generated/order_client.py",
                symbol_id="symbol:GeneratedOrder",
            ),
            _consumer(
                consumer_id="consumer:vendor",
                role=SchemaConsumerRole.PROTOCOL_SURFACE,
                path="vendor/lib/order_schema.json",
                symbol_id="symbol:vendor.Order",
                surface=SchemaSurfaceKind.JSON,
                read_only=True,
            ),
            _consumer(
                consumer_id="consumer:direct",
                role=SchemaConsumerRole.SERIALIZER,
                path="src/order_codec.py",
                symbol_id="symbol:OrderCodec",
            ),
        ],
        field_changes=field_changes,
    )
    by_id = {item.observation.consumer_id: item for item in impact.entries}

    gen = by_id["consumer:generated"]
    assert gen.observation.generated is True
    assert gen.write_mode is WriteMode.REGENERATION
    assert gen.disposition is ConsumerDisposition.MIGRATE
    assert gen.obligation is not None
    assert any("regenerate" in item for item in gen.obligation.behavior_contract_ids)

    vendor = by_id["consumer:vendor"]
    assert vendor.observation.read_only is True
    assert vendor.write_mode is WriteMode.EXTERNAL_OBLIGATION
    assert vendor.disposition is ConsumerDisposition.UPSTREAM
    assert vendor.obligation is not None
    assert any("external" in item for item in vendor.obligation.behavior_contract_ids)

    direct = by_id["consumer:direct"]
    assert direct.write_mode is WriteMode.DIRECT
    assert direct.disposition is ConsumerDisposition.MIGRATE

    # Direct write mode is rejected on generated observations at entry level.
    with pytest.raises(SchemaProtocolChangeImpactAuthorityError):
        from ipfs_accelerate_py.agent_supervisor.analysis.schema_protocol_change_impact import (
            SchemaConsumerImpactEntry,
        )

        SchemaConsumerImpactEntry(
            observation=gen.observation,
            disposition=ConsumerDisposition.MIGRATE,
            compatibility=CompatibilityDirection.INCOMPATIBLE,
            write_mode=WriteMode.DIRECT,
            clause_ids=("clause:schema-1",),
            obligation=gen.obligation,
        )


# ---------------------------------------------------------------------------
# Missing / dynamic codecs remain frontiers
# ---------------------------------------------------------------------------


def test_missing_and_dynamic_codecs_remain_frontiers(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots, _clause(reason="added required field=context"))
    impact = build_schema_protocol_impact(
        delta,
        [
            _consumer(
                consumer_id="consumer:missing",
                role=SchemaConsumerRole.MISSING_CODEC,
                path="src/unknown_codec.py",
                symbol_id="symbol:missing_codec",
                codec_status="missing",
            ),
            _consumer(
                consumer_id="consumer:dynamic",
                role=SchemaConsumerRole.DYNAMIC_CODEC,
                path="src/runtime_codec.py",
                symbol_id="symbol:dynamic_codec",
                codec_status="dynamic",
            ),
            _consumer(
                consumer_id="consumer:known",
                role=SchemaConsumerRole.SERIALIZER,
                path="src/order_codec.py",
                symbol_id="symbol:OrderCodec",
            ),
        ],
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    by_id = {item.observation.consumer_id: item for item in impact.entries}
    assert by_id["consumer:missing"].disposition is ConsumerDisposition.FRONTIER
    assert by_id["consumer:dynamic"].disposition is ConsumerDisposition.FRONTIER
    assert by_id["consumer:missing"].compatibility is CompatibilityDirection.UNKNOWN
    assert by_id["consumer:dynamic"].write_mode is WriteMode.FRONTIER
    assert "consumer:missing" in impact.frontier_consumer_ids
    assert "consumer:dynamic" in impact.frontier_consumer_ids
    # Known serializer still gets its own migrate obligation independently.
    assert by_id["consumer:known"].disposition is ConsumerDisposition.MIGRATE
    assert by_id["consumer:known"].obligation is not None
    # Frontier cannot discharge the known consumer.
    assert len(impact.migrate_entries) >= 1
    assert impact.serialization_impacts
    for item in impact.serialization_impacts:
        if item.codec_status in {"missing", "dynamic"}:
            assert item.is_frontier
            assert item.write_mode is WriteMode.FRONTIER


def test_serialization_impact_rejects_direct_write_for_missing_codec() -> None:
    with pytest.raises(SchemaProtocolChangeImpactAuthorityError):
        SerializationImpact(
            facet=SerializationFacet.SERIALIZER,
            subject_symbol_id="symbol:x",
            path="src/x.py",
            codec_status="missing",
            write_mode=WriteMode.DIRECT,
        )
    with pytest.raises(SchemaProtocolChangeImpactAuthorityError):
        SerializationImpact(
            facet=SerializationFacet.GENERATED_CLIENT,
            subject_symbol_id="symbol:gen",
            path="generated/x.py",
            write_mode=WriteMode.DIRECT,
        )


# ---------------------------------------------------------------------------
# Analyzer integration / independence / round-trip
# ---------------------------------------------------------------------------


def test_analyzer_emits_schema_protocol_impact_report(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(
            clause_id="clause:schema-change",
            kind=DeltaKind.SCHEMA_CHANGE,
            reason="added required field=context",
        ),
        _clause(
            clause_id="clause:serialization-change",
            kind=DeltaKind.SERIALIZATION_CHANGE,
            reason="serialization impact for field=context",
        ),
        _clause(
            clause_id="clause:protocol-change",
            kind=DeltaKind.PROTOCOL_CHANGE,
            reason="protocol impact for field=context",
        ),
    )
    consumers = [
        _consumer(
            consumer_id="consumer:ctor",
            role="constructor",
            path="src/order.py",
            symbol_id="symbol:Order.__init__",
        ),
        _consumer(
            consumer_id="consumer:json",
            role="protocol_surface",
            path="schemas/order.json",
            symbol_id="symbol:OrderJSON",
            surface="json",
        ),
        _consumer(
            consumer_id="consumer:ser",
            role="serializer",
            path="src/ser.py",
            symbol_id="symbol:ser",
        ),
        _consumer(
            consumer_id="consumer:gen",
            role="generated_client",
            path="generated/order_pb2.py",
            symbol_id="symbol:pb.Order",
        ),
        _consumer(
            consumer_id="consumer:dyn",
            role="dynamic_codec",
            path="src/dyn.py",
            symbol_id="symbol:dyn",
            codec_status="dynamic",
        ),
    ]
    impact = SchemaProtocolChangeAnalyzer(roots=roots).analyze(
        delta,
        consumers,
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
        evidence_refs=("evidence:rpr-030",),
    )
    assert isinstance(impact, SchemaProtocolImpact)
    assert isinstance(impact, SchemaProtocolChangeImpact)
    assert impact.schema == SCHEMA_PROTOCOL_IMPACT_SCHEMA
    assert impact.subject_symbol_id == "symbol:Order"
    assert impact.field_changes
    assert impact.constructor_impacts
    assert impact.serialization_impacts
    assert impact.protocol_impacts
    assert impact.frontier_consumer_ids
    assert impact.obligations
    assert impact.obligation_set_id()
    assert impact.compatibility_by_consumer()
    payload = impact.to_dict()
    assert payload["schema"] == SCHEMA_PROTOCOL_IMPACT_SCHEMA
    assert payload["producer_id"]


def test_observation_round_trip(roots: PropagationAuthorityRoots) -> None:
    observation = _consumer(
        consumer_id="consumer:round",
        role=SchemaConsumerRole.FACTORY,
        path="src/factory.py",
        symbol_id="symbol:make_order",
        construction_kind=ConstructionKind.FACTORY,
        supplies=("id",),
        authority_kinds=(AuthorityKind.REVIEWED_IDL,),
    )
    assert SchemaConsumerObservation.from_dict(observation.to_dict()) == observation


def test_duplicate_consumer_ids_do_not_duplicate_entries(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    first = _consumer(
        consumer_id="consumer:dup",
        role=SchemaConsumerRole.SERIALIZER,
        path="src/ser.py",
        symbol_id="symbol:ser",
    )
    second = _consumer(
        consumer_id="consumer:dup",
        role=SchemaConsumerRole.SERIALIZER,
        path="src/ser.py",
        symbol_id="symbol:ser",
        supplies=("context",),
    )
    impact = build_schema_protocol_impact(
        delta,
        [first, second],
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    assert len(impact.entries) == 1


def test_cross_root_binding_rejected(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    other = PropagationAuthorityRoots(
        repository_id="repository:other",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:other",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    with pytest.raises(SchemaProtocolChangeImpactAuthorityError):
        SchemaProtocolChangeAnalyzer(roots=other).analyze(delta, [])


def test_empty_delta_clauses_rejected(roots: PropagationAuthorityRoots) -> None:
    # ProgramContractDelta itself requires clauses; simulate via direct error path.
    with pytest.raises(Exception):
        ProgramContractDelta(
            roots=roots,
            change_set_id="changeset:empty",
            subject_symbol_id="symbol:Order",
            before_contract_ref="contract:before",
            after_contract_ref="contract:after",
            clauses=(),
        )


def test_path_must_be_repository_relative() -> None:
    with pytest.raises(SchemaProtocolChangeImpactError):
        SchemaConsumerObservation(
            consumer_id="consumer:abs",
            role=SchemaConsumerRole.SERIALIZER,
            path="/abs/path.py",
            symbol_id="symbol:x",
            subject_symbol_id="symbol:Order",
        )
    with pytest.raises(SchemaProtocolChangeImpactError):
        SchemaConsumerObservation(
            consumer_id="consumer:esc",
            role=SchemaConsumerRole.SERIALIZER,
            path="../escape.py",
            symbol_id="symbol:x",
            subject_symbol_id="symbol:Order",
        )


def test_independent_obligations_for_each_consumer(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots, _clause(reason="added required field=context"))
    consumers = [
        _consumer(
            consumer_id=f"consumer:{index}",
            role=role,
            path=f"src/{role.value}_{index}.py",
            symbol_id=f"symbol:{role.value}_{index}",
        )
        for index, role in enumerate(
            (
                SchemaConsumerRole.CONSTRUCTOR,
                SchemaConsumerRole.SERIALIZER,
                SchemaConsumerRole.DESERIALIZER,
                SchemaConsumerRole.PERSISTENCE,
                SchemaConsumerRole.PROTOCOL_SURFACE,
            )
        )
    ]
    # Attach surfaces for protocol consumer.
    consumers[-1] = _consumer(
        consumer_id="consumer:4",
        role=SchemaConsumerRole.PROTOCOL_SURFACE,
        path="schemas/order.json",
        symbol_id="symbol:protocol_4",
        surface=SchemaSurfaceKind.JSON,
    )
    impact = build_schema_protocol_impact(
        delta,
        consumers,
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    assert len(impact.entries) == 5
    assert len(impact.obligations) == 5
    obligation_ids = {item.obligation_id for item in impact.obligations}
    assert len(obligation_ids) == 5
    # Compatible supply on one does not clear others.
    with_supply = _consumer(
        consumer_id="consumer:supplied",
        role=SchemaConsumerRole.CONSTRUCTOR,
        path="src/supplied.py",
        symbol_id="symbol:supplied",
        supplies=("context",),
    )
    impact2 = build_schema_protocol_impact(
        delta,
        [*consumers, with_supply],
        field_changes=[_field(FieldChangeKind.ADDED, "context", required=True)],
    )
    assert any(
        item.disposition is ConsumerDisposition.COMPATIBLE for item in impact2.entries
    )
    assert len(impact2.migrate_entries) >= 4
    assert impact2.one_compatible_cannot_discharge_others()
