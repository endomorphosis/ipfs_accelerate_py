"""DCR-041: alias, registration, and unique-anchor repair operators.

Acceptance
----------
* Mutation tests cover duplicates, wrong owners, stale spans, idempotence, and
  rollback.
* Behavior postconditions replace anchor-count-only validation.
* Ambiguous multi-anchor situations abstain unless a unique typed ownership
  edge proves ownership (never by lexical score).
* Operators remain proposal-only and never grant write authority.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry_repairs import (
    REGISTRY_REPAIR_EVIDENCE,
    REGISTRY_REPAIR_OPERATORS_INTERFACE,
    AddAliasOperator,
    AliasBinding,
    AliasRegistry,
    AnchorKind,
    AnchorRecord,
    AnchorTable,
    BehaviorPostcondition,
    BindRegistrationOperator,
    DisambiguateAnchorOperator,
    OwnershipEdgeKind,
    RegistrationBinding,
    RegistrationTable,
    RegistryOperatorKind,
    RegistryRepairAbstention,
    RegistryRepairError,
    RemoveAliasOperator,
    RenameAliasOperator,
    SourceSpan,
    build_registry_repair_operators,
    evaluate_behavior_postcondition,
    make_registry_repair_receipt,
    make_span,
    registry_operator_vectors,
    span_is_fresh,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

HANDLER_A = "pkg.tools.surface_a:handle_tool"
HANDLER_B = "pkg.tools.surface_b:handle_tool"
HANDLER_DEMO = "pkg.tools.demo:handle_demo"


def _span(
    path: str = "pkg/tools/registry.py",
    *,
    start: int = 0,
    end: int = 8,
    body: str = "binding",
    before_hash: str | None = None,
) -> SourceSpan:
    return make_span(path, start_offset=start, end_offset=end, body=body, before_hash=before_hash)


def _alias(
    *,
    alias_key: str = "demo_alias",
    target_tool: str = "demo_tool",
    owner_ref: str = HANDLER_DEMO,
    span: SourceSpan | None = None,
) -> AliasBinding:
    return AliasBinding(
        alias_key=alias_key,
        target_tool=target_tool,
        owner_ref=owner_ref,
        span=span or _span(body=f"alias:{alias_key}"),
    )


def _registration(
    *,
    tool_name: str = "demo_tool",
    owner_ref: str = HANDLER_DEMO,
    handler_ref: str = HANDLER_DEMO,
    span: SourceSpan | None = None,
) -> RegistrationBinding:
    return RegistrationBinding(
        tool_name=tool_name,
        owner_ref=owner_ref,
        handler_ref=handler_ref,
        input_schema_ref=f"schema:{tool_name}/input@1",
        span=span or _span(body=f"registration:{tool_name}"),
    )


def _anchor(
    *,
    anchor_id: str,
    registry_key: str = "registration:demo_tool",
    owner_ref: str = HANDLER_DEMO,
    semantic_target: str | None = None,
    span: SourceSpan | None = None,
    ownership_edge: OwnershipEdgeKind = OwnershipEdgeKind.REGISTRATION_TO_HANDLER,
    selected: bool = False,
    kind: AnchorKind = AnchorKind.REGISTRATION,
) -> AnchorRecord:
    return AnchorRecord(
        anchor_id=anchor_id,
        kind=kind,
        registry_key=registry_key,
        owner_ref=owner_ref,
        semantic_target=semantic_target or f"handler:{owner_ref}",
        span=span or _span(body=anchor_id),
        ownership_edge=ownership_edge,
        selected=selected,
    )


# ---------------------------------------------------------------------------
# Interface / registry binding
# ---------------------------------------------------------------------------


def test_interfaces_and_registry_family_binding() -> None:
    assert REGISTRY_REPAIR_OPERATORS_INTERFACE == "RegistryRepairOperators@1"
    assert REGISTRY_REPAIR_EVIDENCE == "dcr/registry-repair@1"
    ops = build_registry_repair_operators()
    assert ops.INTERFACE == REGISTRY_REPAIR_OPERATORS_INTERFACE
    assert ops.EVIDENCE_ID == REGISTRY_REPAIR_EVIDENCE
    assert isinstance(ops.add_alias, AddAliasOperator)
    assert isinstance(ops.remove_alias, RemoveAliasOperator)
    assert isinstance(ops.rename_alias, RenameAliasOperator)
    assert isinstance(ops.bind_registration, BindRegistrationOperator)
    assert isinstance(ops.disambiguate_anchor, DisambiguateAnchorOperator)

    reg = build_default_operator_registry()
    for kind, operator in (
        (OperatorKind.ADD_ALIAS, ops.add_alias),
        (OperatorKind.REMOVE_ALIAS, ops.remove_alias),
        (OperatorKind.RENAME_ALIAS, ops.rename_alias),
        (OperatorKind.BIND_REGISTRATION, ops.bind_registration),
        (OperatorKind.DISAMBIGUATE_ANCHOR, ops.disambiguate_anchor),
    ):
        descriptor = reg.require_known(kind)
        assert descriptor.family is OperatorFamily.REGISTRY
        assert operator.descriptor.kind is kind
        assert operator.descriptor.proposal_only is True
        assert operator.descriptor.grants_write_authority is False
        assert operator.operator_id.startswith("dcr-operator:")


def test_forbidden_bodies_and_model_owners_fail_closed() -> None:
    with pytest.raises(RegistryRepairError, match="forbidden body"):
        AliasBinding.from_dict(
            {
                "alias_key": "demo_alias",
                "target_tool": "demo_tool",
                "owner_ref": HANDLER_DEMO,
                "span": _span().to_dict(),
                "source_body": "def handler(): pass",
            }
        )
    with pytest.raises(RegistryRepairError, match="model|provider"):
        AliasBinding(
            alias_key="demo_alias",
            target_tool="demo_tool",
            owner_ref="pkg.tools.demo:llm_prompt",
            span=_span(),
        )


# ---------------------------------------------------------------------------
# Add / remove / rename alias operators
# ---------------------------------------------------------------------------


def test_add_alias_preview_inverse_idempotent() -> None:
    table = AliasRegistry.empty()
    op = AddAliasOperator()
    binding = _alias()
    preview, after = op.preview(table, binding)
    assert preview.operator_kind == RegistryOperatorKind.ADD_ALIAS.value
    assert preview.proposal_only is True
    assert preview.grants_write_authority is False
    assert after.contains("demo_alias")
    assert after.get("demo_alias") is not None
    assert after.get("demo_alias").target_tool == "demo_tool"
    assert preview.postcondition is not None
    assert preview.postcondition.behavior_satisfied is True
    assert preview.postcondition.count_only_sufficient is False

    restored = op.inverse(after, preview)
    assert restored.table_id == table.table_id
    assert restored.contains("demo_alias") is False

    # Idempotent re-apply leaves table identity stable.
    after2, preview2 = op.apply(after, binding)
    assert after2.table_id == after.table_id
    assert preview2.before_table_id == preview2.after_table_id
    assert "idempotent" in preview2.reason_codes
    assert op.inverse(after2, preview2).table_id == after2.table_id

    receipt = make_registry_repair_receipt(preview)
    assert receipt.receipt_id
    assert receipt.postcondition.behavior_satisfied is True


def test_add_alias_rejects_duplicates_wrong_owners_and_stale_spans() -> None:
    op = AddAliasOperator()
    original = _alias(owner_ref=HANDLER_A, target_tool="tool_a")
    table, _ = op.apply(AliasRegistry.empty(), original)

    # Duplicate alias with different target.
    conflict_target = _alias(
        owner_ref=HANDLER_A,
        target_tool="tool_b",
        span=_span(body="conflict-target"),
    )
    with pytest.raises(RegistryRepairAbstention, match="duplicate_alias|target_conflict"):
        op.preview(table, conflict_target)

    # Wrong owner on an existing alias key.
    conflict_owner = _alias(
        owner_ref=HANDLER_B,
        target_tool="tool_a",
        span=_span(body="conflict-owner"),
    )
    with pytest.raises(RegistryRepairAbstention, match="wrong_owner"):
        op.preview(table, conflict_owner)

    # Expected owner gate on a fresh alias.
    fresh = _alias(alias_key="other_alias", owner_ref=HANDLER_A, target_tool="tool_a")
    with pytest.raises(RegistryRepairAbstention, match="wrong_owner"):
        op.preview(table, fresh, expected_owner=HANDLER_B)

    # Stale span (observed before-hash does not match span evidence).
    stale = _alias(
        alias_key="stale_alias",
        owner_ref=HANDLER_A,
        span=_span(body="stale-body"),
    )
    with pytest.raises(RegistryRepairAbstention, match="stale span|before_hash"):
        op.preview(
            table,
            stale,
            observed_before_hash="sha256:" + ("0" * 64),
        )


def test_remove_and_rename_alias_rollback() -> None:
    add = AddAliasOperator()
    remove = RemoveAliasOperator()
    rename = RenameAliasOperator()
    binding = _alias(alias_key="old_alias")
    table, _ = add.apply(AliasRegistry.empty(), binding)

    renamed_preview, renamed = rename.preview(table, "old_alias", "new_alias")
    assert renamed.contains("new_alias")
    assert renamed.contains("old_alias") is False
    assert renamed_preview.postcondition is not None
    assert renamed_preview.postcondition.behavior_satisfied is True

    rolled = rename.inverse(
        renamed, renamed_preview, old_key="old_alias", new_key="new_alias"
    )
    assert rolled.table_id == table.table_id
    assert rolled.contains("old_alias")

    # Idempotent rename to same key.
    same_preview, same = rename.preview(table, "old_alias", "old_alias")
    assert same.table_id == table.table_id
    assert "idempotent" in same_preview.reason_codes

    removed_preview, removed = remove.preview(table, "old_alias")
    assert removed.contains("old_alias") is False
    restored = remove.inverse(
        removed, removed_preview, restored_binding=binding
    )
    assert restored.table_id == table.table_id

    # Wrong owner on remove.
    with pytest.raises(RegistryRepairError, match="wrong owner"):
        remove.preview(table, "old_alias", expected_owner=HANDLER_B)

    # Stale span on rename.
    with pytest.raises(RegistryRepairError, match="stale span"):
        rename.preview(
            table,
            "old_alias",
            "renamed_again",
            observed_before_hash="sha256:" + ("1" * 64),
        )


# ---------------------------------------------------------------------------
# Bind registration operator
# ---------------------------------------------------------------------------


def test_bind_registration_requires_known_handler_and_is_idempotent() -> None:
    op = BindRegistrationOperator()
    binding = _registration()
    empty = RegistrationTable.empty()
    with pytest.raises(RegistryRepairAbstention, match="handler_not_known"):
        op.preview(empty, binding)

    table = RegistrationTable.empty(known_handlers=(HANDLER_DEMO,))
    preview, after = op.preview(table, binding)
    assert after.contains("demo_tool")
    assert after.get("demo_tool").handler_ref == HANDLER_DEMO
    assert preview.postcondition is not None
    assert preview.postcondition.resolves is True
    assert preview.postcondition.unique_owner is True
    assert preview.postcondition.behavior_satisfied is True

    restored = op.inverse(after, preview)
    assert restored.contains("demo_tool") is False
    assert restored.table_id == table.table_id

    after2, preview2 = op.apply(after, binding)
    assert after2.table_id == after.table_id
    assert "idempotent" in preview2.reason_codes
    assert op.inverse(after2, preview2).table_id == after2.table_id


def test_bind_registration_rejects_duplicates_wrong_owners_stale_spans() -> None:
    op = BindRegistrationOperator()
    table = RegistrationTable.empty(known_handlers=(HANDLER_A, HANDLER_B))
    original = _registration(owner_ref=HANDLER_A, handler_ref=HANDLER_A)
    bound, _ = op.apply(table, original)

    # Duplicate registration with different handler.
    conflict = _registration(
        owner_ref=HANDLER_A,
        handler_ref=HANDLER_B,
        span=_span(body="conflict-handler"),
    )
    with pytest.raises(RegistryRepairError, match="duplicate_registration|handler_conflict"):
        op.preview(bound, conflict)

    # Wrong owner.
    wrong_owner = _registration(
        owner_ref=HANDLER_B,
        handler_ref=HANDLER_A,
        span=_span(body="wrong-owner"),
    )
    with pytest.raises(RegistryRepairError, match="wrong_owner"):
        op.preview(bound, wrong_owner)

    # Expected owner gate.
    fresh = _registration(
        tool_name="other_tool",
        owner_ref=HANDLER_A,
        handler_ref=HANDLER_A,
        span=_span(body="other"),
    )
    with pytest.raises(RegistryRepairError, match="wrong_owner"):
        op.preview(bound, fresh, expected_owner=HANDLER_B)

    # Stale span.
    stale = _registration(
        tool_name="stale_tool",
        owner_ref=HANDLER_A,
        handler_ref=HANDLER_A,
        span=_span(body="stale-reg"),
    )
    with pytest.raises(RegistryRepairError, match="stale span|before_hash"):
        op.preview(
            bound,
            stale,
            observed_before_hash="sha256:" + ("a" * 64),
        )


# ---------------------------------------------------------------------------
# Unique-anchor operator + behavior postcondition
# ---------------------------------------------------------------------------


def test_disambiguate_anchor_unique_typed_edge_and_rollback() -> None:
    op = DisambiguateAnchorOperator()
    key = "registration:demo_tool"
    prior = (
        _anchor(anchor_id="anchor:z_path", registry_key=key, owner_ref=HANDLER_A),
        _anchor(
            anchor_id="anchor:a_path",
            registry_key=key,
            owner_ref=HANDLER_A,
            span=_span(start=20, end=40, body="second"),
        ),
    )
    table = AnchorTable(anchors=prior)
    assert len(table.for_key(key)) == 2

    preview, after = op.preview(table, key)
    retained = after.for_key(key)
    assert len(retained) == 1
    assert retained[0].selected is True
    assert retained[0].owner_ref == HANDLER_A
    assert preview.postcondition is not None
    assert preview.postcondition.behavior_satisfied is True
    assert preview.postcondition.count_only_sufficient is False
    # Behavior axes must hold — not merely the count.
    assert preview.postcondition.resolves is True
    assert preview.postcondition.unique_owner is True
    assert preview.postcondition.span_fresh is True
    assert preview.postcondition.semantic_target
    assert preview.postcondition.inverse_patch_id

    restored = op.inverse(after, preview, prior_anchors=prior)
    assert restored.table_id == table.table_id
    assert len(restored.for_key(key)) == 2

    # Idempotent re-apply on already unique selected set.
    after2, preview2 = op.apply(after, key)
    assert after2.table_id == after.table_id
    assert "idempotent" in preview2.reason_codes or "already_unique" in preview2.reason_codes


def test_disambiguate_anchor_abstains_on_ambiguous_owners_not_lexical_score() -> None:
    op = DisambiguateAnchorOperator()
    key = "registration:demo_tool"
    # Different owners: must abstain even if one path sorts first lexically.
    table = AnchorTable(
        anchors=(
            _anchor(anchor_id="anchor:aaa", registry_key=key, owner_ref=HANDLER_A),
            _anchor(anchor_id="anchor:zzz", registry_key=key, owner_ref=HANDLER_B),
        )
    )
    with pytest.raises(
        RegistryRepairAbstention,
        match="ambiguous_owners|lexical_score_forbidden",
    ):
        op.preview(table, key)

    # Ambiguous semantic targets under same owner also abstain.
    table2 = AnchorTable(
        anchors=(
            _anchor(
                anchor_id="anchor:1",
                registry_key=key,
                owner_ref=HANDLER_A,
                semantic_target="handler:one",
            ),
            _anchor(
                anchor_id="anchor:2",
                registry_key=key,
                owner_ref=HANDLER_A,
                semantic_target="handler:two",
            ),
        )
    )
    with pytest.raises(RegistryRepairAbstention, match="ambiguous_semantic_targets"):
        op.preview(table2, key)


def test_disambiguate_anchor_rejects_stale_spans_and_wrong_owners() -> None:
    op = DisambiguateAnchorOperator()
    key = "registration:demo_tool"
    span = _span(body="fresh")
    table = AnchorTable(
        anchors=(
            _anchor(
                anchor_id="anchor:1",
                registry_key=key,
                owner_ref=HANDLER_A,
                span=span,
            ),
            _anchor(
                anchor_id="anchor:2",
                registry_key=key,
                owner_ref=HANDLER_A,
                span=_span(start=50, end=60, body="also-fresh"),
            ),
        )
    )
    with pytest.raises(RegistryRepairError, match="stale_span|stale span"):
        op.preview(
            table,
            key,
            observed_before_hash="sha256:" + ("f" * 64),
        )

    # Unique owner but expected_owner gate fails.
    with pytest.raises(RegistryRepairError, match="wrong owner"):
        op.preview(table, key, expected_owner=HANDLER_B)


def test_behavior_postcondition_replaces_anchor_count_only_validation() -> None:
    # A single anchor with broken ownership/resolution must not pass.
    with pytest.raises(RegistryRepairError, match="count_only_sufficient"):
        BehaviorPostcondition(
            registry_key="registration:demo_tool",
            semantic_target=f"handler:{HANDLER_A}",
            owner_ref=HANDLER_A,
            resolves=True,
            unique_owner=True,
            span_fresh=True,
            inverse_patch_id="patch:demo",
            before_hash="sha256:" + ("b" * 64),
            after_table_id="table:demo",
            anchor_count=1,
            count_only_sufficient=True,
        )

    # Count == 1 is recorded but insufficient when resolution fails.
    broken = evaluate_behavior_postcondition(
        registry_key="registration:demo_tool",
        semantic_target=f"handler:{HANDLER_A}",
        owner_ref=HANDLER_A,
        resolves=False,
        unique_owner=True,
        span_fresh=True,
        inverse_patch_id="patch:demo",
        before_hash="sha256:" + ("b" * 64),
        after_table_id="table:demo",
        anchor_count=1,
    )
    assert broken.anchor_count == 1
    assert broken.count_only_sufficient is False
    assert broken.behavior_satisfied is False

    # Stale span with count == 1 still fails.
    stale = evaluate_behavior_postcondition(
        registry_key="registration:demo_tool",
        semantic_target=f"handler:{HANDLER_A}",
        owner_ref=HANDLER_A,
        resolves=True,
        unique_owner=True,
        span_fresh=False,
        inverse_patch_id="patch:demo",
        before_hash="sha256:" + ("b" * 64),
        after_table_id="table:demo",
        anchor_count=1,
    )
    assert stale.behavior_satisfied is False

    # Full behavioral axes pass.
    ok = evaluate_behavior_postcondition(
        registry_key="registration:demo_tool",
        semantic_target=f"handler:{HANDLER_A}",
        owner_ref=HANDLER_A,
        resolves=True,
        unique_owner=True,
        span_fresh=True,
        inverse_patch_id="patch:demo",
        before_hash="sha256:" + ("b" * 64),
        after_table_id="table:demo",
        anchor_count=1,
    )
    assert ok.behavior_satisfied is True
    assert ok.count_only_sufficient is False


def test_span_freshness_helper() -> None:
    span = _span(body="exact")
    assert span_is_fresh(span, span.before_hash) is True
    assert span_is_fresh(span, "sha256:" + ("0" * 64)) is False


def test_tables_are_content_addressed_and_reject_duplicate_keys() -> None:
    binding = _alias()
    table = AliasRegistry(bindings=(binding,))
    rebuilt = AliasRegistry.from_dict(table.to_dict())
    assert rebuilt.table_id == table.table_id
    assert table.table_id == content_identity(table._payload_without_table_id())
    with pytest.raises(RegistryRepairError, match="unique"):
        AliasRegistry(bindings=(binding, _alias(span=_span(body="other"))))

    reg = _registration()
    reg_table = RegistrationTable(bindings=(reg,), known_handlers=(HANDLER_DEMO,))
    assert RegistrationTable.from_dict(reg_table.to_dict()).table_id == reg_table.table_id
    assert reg_table.table_id == content_identity(reg_table._payload_without_table_id())
    with pytest.raises(RegistryRepairError, match="unique"):
        RegistrationTable(
            bindings=(reg, _registration(span=_span(body="dup-reg"))),
            known_handlers=(HANDLER_DEMO,),
        )


def test_registry_operator_vectors_are_content_addressed() -> None:
    vectors = registry_operator_vectors()
    assert vectors["interface"] == REGISTRY_REPAIR_OPERATORS_INTERFACE
    assert vectors["evidence_id"] == REGISTRY_REPAIR_EVIDENCE
    assert vectors["count_only_sufficient"] is False
    assert vectors["behavior_postcondition_required"] is True
    assert len(vectors["operators"]) == 5
    assert vectors["vector_digest"].startswith("sha256:")
    assert all(op_id.startswith("dcr-operator:") for op_id in vectors["operators"])


def test_disambiguate_requires_matching_ownership_edge_when_requested() -> None:
    op = DisambiguateAnchorOperator()
    key = "alias:demo_alias"
    table = AnchorTable(
        anchors=(
            _anchor(
                anchor_id="anchor:1",
                registry_key=key,
                owner_ref=HANDLER_A,
                kind=AnchorKind.ALIAS,
                ownership_edge=OwnershipEdgeKind.ALIAS_TO_TOOL,
            ),
            _anchor(
                anchor_id="anchor:2",
                registry_key=key,
                owner_ref=HANDLER_A,
                kind=AnchorKind.ALIAS,
                ownership_edge=OwnershipEdgeKind.ALIAS_TO_TOOL,
                span=_span(start=5, end=15, body="alias-2"),
            ),
        )
    )
    preview, after = op.preview(
        table,
        key,
        required_edge=OwnershipEdgeKind.ALIAS_TO_TOOL,
    )
    assert after.for_key(key)[0].ownership_edge is OwnershipEdgeKind.ALIAS_TO_TOOL
    assert preview.postcondition is not None
    assert preview.postcondition.behavior_satisfied is True

    with pytest.raises(RegistryRepairAbstention, match="no_matching_ownership_edge"):
        op.preview(
            table,
            key,
            required_edge=OwnershipEdgeKind.REGISTRATION_TO_HANDLER,
        )
