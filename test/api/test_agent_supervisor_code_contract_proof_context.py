from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.code_contract_proof_context import (
    CodeContractProofContextCompiler,
    CodeContractProofContextError,
    ProofContextItem,
    ProofContextItemKind,
    ProofContextLimits,
    ProofContextRequest,
    ProofContextStatus,
    compile_proof_context_delta,
)


def _item(
    item_id: str,
    kind: ProofContextItemKind,
    dependencies: tuple[str, ...] = (),
    **payload: object,
) -> ProofContextItem:
    return ProofContextItem(
        item_id=item_id,
        kind=kind,
        payload=payload or {"symbol": item_id},
        dependency_ids=dependencies,
        expansion_locator=f"record:{item_id}",
        referenced_content_id=f"cid:{item_id}",
    )


def _request(*extra: ProofContextItem, limits=None, graph_slice=None):
    items = (
        _item("obl", ProofContextItemKind.OBLIGATION, ("contract", "rule")),
        _item("contract", ProofContextItemKind.CONTRACT, ("call", "effect")),
        _item("call", ProofContextItemKind.CALL, ("definition",)),
        _item("definition", ProofContextItemKind.DEFINITION),
        _item("effect", ProofContextItemKind.EFFECT, ("assumption",)),
        _item("assumption", ProofContextItemKind.ASSUMPTION),
        _item("rule", ProofContextItemKind.RULE),
        _item("unrelated", ProofContextItemKind.DEFINITION),
        *extra,
    )
    return ProofContextRequest(
        obligation_id="obl",
        items=items,
        limits=limits or ProofContextLimits(max_bytes=100_000, max_items=50),
        program_graph_slice=graph_slice,
    )


def test_selects_smallest_dependency_complete_closure_with_audit_reasons():
    result = CodeContractProofContextCompiler().compile(_request())

    assert result.status is ProofContextStatus.COMPLETE
    assert result.included_item_ids == (
        "assumption",
        "call",
        "contract",
        "definition",
        "effect",
        "obl",
        "rule",
    )
    assert "unrelated" not in result.included_item_ids
    decisions = {item.item_id: item for item in result.decisions}
    assert decisions["obl"].reason == "selected_obligation"
    assert decisions["definition"].reason == "transitive_required_dependency"
    assert decisions["unrelated"].reason == "not_in_obligation_dependency_closure"
    assert not decisions["unrelated"].included
    handles = {item.item_id: item for item in result.expansion_handles}
    assert handles["unrelated"].target_content_id == "cid:unrelated"
    assert handles["unrelated"].handle_id == handles["unrelated"].handle_id
    assert result.to_dict()["embeds_source_bodies"] is False
    assert result.to_dict()["embeds_full_graph"] is False


def test_missing_dependency_and_incomplete_graph_slice_fail_closed():
    request = ProofContextRequest(
        obligation_id="obl",
        items=(
            _item("obl", ProofContextItemKind.OBLIGATION, ("missing",)),
        ),
        program_graph_slice={
            "slice_id": "slice:1",
            "complete": False,
            "dependency_complete": False,
            "truncated": True,
            "node_ids": ["n1"],
            "edge_ids": [],
            "omitted_dependencies": ["n2"],
        },
    )

    result = CodeContractProofContextCompiler().compile(request)

    assert result.status is ProofContextStatus.INCOMPLETE
    assert result.included_item_ids == ("obl",)
    assert "missing_required_item:missing" in result.incomplete_reasons
    assert "program_graph_slice_dependency_incomplete" in result.incomplete_reasons
    assert "program_graph_slice_truncated" in result.incomplete_reasons
    assert result.program_graph_slice.node_count == 1
    assert "node_ids" not in result.program_graph_slice.to_dict()


def test_required_items_are_never_truncated_when_limits_are_exceeded():
    request = _request(limits=ProofContextLimits(max_bytes=1, max_items=1))

    result = CodeContractProofContextCompiler().compile(request)

    assert result.status is ProofContextStatus.INCOMPLETE
    assert len(result.items) == 7
    assert result.metrics.item_count == 7
    assert result.metrics.byte_count > 1
    assert result.metrics.llm_invocations == 0
    assert "required_item_limit_exceeded" in result.incomplete_reasons
    assert "required_byte_limit_exceeded" in result.incomplete_reasons
    assert result.to_dict()["required_inputs_truncated"] is False


def test_identical_context_reuses_exact_object_and_receipt():
    compiler = CodeContractProofContextCompiler()
    request = _request()

    first = compiler.compile(request)
    second = compiler.compile(request)

    assert first is second
    assert first.context_id == second.context_id
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert compiler.cache_size == 1
    assert compiler.receipt_is_valid(first.receipt, request)


def test_changed_dependency_invalidates_old_receipt():
    compiler = CodeContractProofContextCompiler()
    original = _request()
    first = compiler.compile(original)
    changed_items = tuple(
        replace(item, payload={"symbol": "definition-v2"})
        if item.item_id == "definition"
        else item
        for item in original.items
    )
    changed = replace(original, items=changed_items)

    second = compiler.compile(changed, previous_receipt=first.receipt)
    retried = compiler.compile(changed, previous_receipt=second.receipt)

    assert second.dependency_fingerprint != first.dependency_fingerprint
    assert not compiler.receipt_is_valid(first.receipt, changed)
    assert second.invalidated_receipt_ids == (first.receipt.receipt_id,)
    assert second.receipt.receipt_id != first.receipt.receipt_id
    assert retried is second
    assert compiler.receipt_is_valid(second.receipt, changed)


def test_delta_retry_only_sends_new_counterexample_and_requested_evidence():
    counterexample = _item(
        "cex", ProofContextItemKind.COUNTEREXAMPLE, ("cex-rule",)
    )
    cex_rule = _item("cex-rule", ProofContextItemKind.RULE)
    evidence = _item("evidence", ProofContextItemKind.EVIDENCE)
    compiler = CodeContractProofContextCompiler()
    base = compiler.compile(_request())
    request = _request(counterexample, cex_rule, evidence)

    delta = compile_proof_context_delta(
        request,
        base_receipt=base.receipt,
        counterexample_item_ids=("cex",),
        requested_evidence_item_ids=("evidence",),
        compiler=compiler,
    )

    assert delta.status is ProofContextStatus.COMPLETE
    assert delta.transmitted_item_ids == ("cex", "cex-rule", "evidence")
    assert not set(delta.transmitted_item_ids) & set(base.included_item_ids)
    assert {decision.reason for decision in delta.decisions} == {
        "new_counterexample",
        "new_required_dependency",
        "requested_evidence",
    }
    assert delta.metrics.llm_invocations == 0


def test_forged_receipt_is_invalid_for_reuse_and_delta():
    compiler = CodeContractProofContextCompiler()
    request = _request()
    base = compiler.compile(request)
    forged = replace(base.receipt, included_item_ids=("obl",))

    assert not compiler.receipt_is_valid(forged, request)
    recompiled = compiler.compile(request, previous_receipt=forged)
    assert recompiled.invalidated_receipt_ids == (forged.receipt_id,)

    delta = compiler.compile_delta(request, base_receipt=forged)
    assert delta.status is ProofContextStatus.INVALIDATED
    assert delta.incomplete_reasons == ("base_receipt_mismatch",)


def test_delta_rejects_stale_base_after_required_dependency_changes():
    compiler = CodeContractProofContextCompiler()
    original = _request(_item("cex", ProofContextItemKind.COUNTEREXAMPLE))
    base = compiler.compile(original)
    changed = replace(
        original,
        items=tuple(
            replace(item, payload={"changed": True})
            if item.item_id == "rule"
            else item
            for item in original.items
        ),
    )

    delta = compiler.compile_delta(
        changed,
        base_receipt=base.receipt,
        counterexample_item_ids=("cex",),
    )

    assert delta.status is ProofContextStatus.INVALIDATED
    assert delta.transmitted_item_ids == ()
    assert delta.incomplete_reasons == ("base_dependencies_changed",)


def test_rejects_embedded_source_or_full_graph_payloads():
    with pytest.raises(CodeContractProofContextError, match="expansion handle"):
        _item("bad", ProofContextItemKind.DEFINITION, source_text="secret")
    with pytest.raises(CodeContractProofContextError, match="expansion handle"):
        _item(
            "bad",
            ProofContextItemKind.DEFINITION,
            nested={"full_graph": {"nodes": []}},
        )


def test_payloads_are_recursively_immutable_and_serialize_detached_copies():
    item = _item(
        "immutable",
        ProofContextItemKind.RULE,
        nested={"sequence": [{"value": 1}]},
    )

    with pytest.raises(TypeError):
        item.payload["nested"]["new"] = True
    with pytest.raises(TypeError):
        item.payload["nested"]["sequence"][0]["value"] = 2

    serialized = item.to_dict()
    serialized["payload"]["nested"]["sequence"][0]["value"] = 3
    assert item.payload["nested"]["sequence"][0]["value"] == 1


def test_identifier_limit_and_graph_types_are_not_silently_coerced():
    with pytest.raises(CodeContractProofContextError, match="positive integer"):
        ProofContextLimits.from_value({"max_bytes": "12"})
    with pytest.raises(CodeContractProofContextError, match="iterable"):
        ProofContextItem(
            item_id="bad",
            kind=ProofContextItemKind.RULE,
            payload={},
            dependency_ids="abc",
        )
    with pytest.raises(CodeContractProofContextError, match="boolean"):
        _request(
            graph_slice={
                "slice_id": "slice:bad",
                "complete": "false",
                "dependency_complete": False,
            }
        )


def test_obligation_kind_is_enforced_but_cycles_are_closed():
    cyclic = ProofContextRequest(
        obligation_id="obl",
        items=(
            _item("obl", ProofContextItemKind.OBLIGATION, ("a",)),
            _item("a", ProofContextItemKind.RULE, ("b",)),
            _item("b", ProofContextItemKind.RULE, ("a",)),
        ),
        limits=ProofContextLimits(max_bytes=100_000, max_items=10),
    )
    assert CodeContractProofContextCompiler().compile(cyclic).complete

    invalid = replace(
        cyclic,
        items=(
            _item("obl", ProofContextItemKind.RULE),
        ),
    )
    with pytest.raises(CodeContractProofContextError, match="obligation item"):
        CodeContractProofContextCompiler().compile(invalid)
