from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.code_contract_proof_context import (
    CodeContractProofContextCompiler,
    CodeContractProofContextError,
    MINIMAL_PROOF_CONTEXT_EVIDENCE as PROOF_CONTEXT_MODULE_EVIDENCE,
    ProofContextItem,
    ProofContextItemKind,
    ProofContextLimits,
    ProofContextRequest,
    ProofContextStatus,
    compile_proof_context_delta,
)
from ipfs_accelerate_py.agent_supervisor.code_contract_prover import (
    MINIMAL_PROOF_CONTEXT_CLAIM_SCHEMA,
    MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS,
    MINIMAL_PROOF_CONTEXT_EVIDENCE,
    MINIMAL_PROOF_CONTEXT_GOAL_ID,
    MINIMAL_PROOF_CONTEXT_INVARIANTS,
    MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID,
    MINIMAL_PROOF_CONTEXT_TASK_ID,
    context_obeys_minimal_proof_context,
    context_satisfies_minimal_proof_context,
    default_minimal_proof_context_request,
    minimal_proof_context_evidence,
    minimal_proof_context_evidence_terms,
    prove_minimal_proof_context,
    prove_minimal_proof_context_evidence,
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


# ---------------------------------------------------------------------------
# VFS-G157 / VFS-092 objective evidence surface (vfs/minimal-proof-context@1)
# ---------------------------------------------------------------------------


def test_minimal_proof_context_evidence_terms_bind_vfs_g157() -> None:
    """Discovery scanners must observe the exact objective evidence term."""

    assert minimal_proof_context_evidence() == "vfs/minimal-proof-context@1"
    assert MINIMAL_PROOF_CONTEXT_EVIDENCE == "vfs/minimal-proof-context@1"
    assert PROOF_CONTEXT_MODULE_EVIDENCE == MINIMAL_PROOF_CONTEXT_EVIDENCE
    assert minimal_proof_context_evidence_terms() == (
        MINIMAL_PROOF_CONTEXT_EVIDENCE,
    )
    assert MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS == (
        "vfs/minimal-proof-context@1",
    )
    assert MINIMAL_PROOF_CONTEXT_GOAL_ID == "VFS-G157"
    assert MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID == "VFS-G071"
    assert MINIMAL_PROOF_CONTEXT_TASK_ID == "VFS-092"
    assert any(
        "never truncated" in item for item in MINIMAL_PROOF_CONTEXT_INVARIANTS
    )
    assert any(
        "inclusion reasons" in item for item in MINIMAL_PROOF_CONTEXT_INVARIANTS
    )
    assert any(
        "reuse exact receipts" in item
        for item in MINIMAL_PROOF_CONTEXT_INVARIANTS
    )
    assert any(
        "invalidate" in item for item in MINIMAL_PROOF_CONTEXT_INVARIANTS
    )


def test_context_satisfies_minimal_proof_context_fail_closed() -> None:
    """Required kinds stay in the closure; limits never silently promote."""

    request = default_minimal_proof_context_request()
    compiler = CodeContractProofContextCompiler()
    complete = compiler.compile(request)

    assert context_obeys_minimal_proof_context(complete)
    assert context_satisfies_minimal_proof_context(
        complete,
        required_item_ids=(
            "obl",
            "contract",
            "call",
            "effect",
            "axiom",
            "definition",
            "rule",
        ),
        required_kinds=(
            ProofContextItemKind.OBLIGATION,
            ProofContextItemKind.CONTRACT,
            ProofContextItemKind.CALL,
            ProofContextItemKind.EFFECT,
            ProofContextItemKind.ASSUMPTION,
        ),
        forbidden_item_ids=("optional-premise", "unrelated"),
        require_complete=True,
    )
    # Portable mapping path.
    assert context_satisfies_minimal_proof_context(
        complete.to_dict(),
        required_item_ids=("contract", "call", "effect", "axiom"),
        forbidden_item_ids=("unrelated",),
    )
    # Missing required node fails.
    assert not context_satisfies_minimal_proof_context(
        complete,
        required_item_ids=("does-not-exist",),
    )
    # Forbidden optional premise fails when treated as required-absent only if present.
    assert not context_satisfies_minimal_proof_context(
        complete,
        forbidden_item_ids=("contract",),
    )

    limited = compiler.compile(
        replace(request, limits=ProofContextLimits(max_bytes=1, max_items=1))
    )
    assert limited.status is ProofContextStatus.INCOMPLETE
    assert context_obeys_minimal_proof_context(limited)
    assert limited.to_dict()["required_inputs_truncated"] is False
    # Incomplete context cannot satisfy require_complete.
    assert not context_satisfies_minimal_proof_context(
        limited,
        require_complete=True,
    )


def test_prove_minimal_proof_context_for_required_kinds_and_receipts() -> None:
    """VFS-G157 claim: required kinds retained, reasons audited, receipts reuse/invalidate."""

    claim = prove_minimal_proof_context()

    assert claim["schema"] == MINIMAL_PROOF_CONTEXT_CLAIM_SCHEMA
    assert claim["evidence"] == "vfs/minimal-proof-context@1"
    assert claim["evidence_terms"] == ["vfs/minimal-proof-context@1"]
    assert claim["all_evidence_terms"] == list(
        MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS
    )
    assert claim["goal_id"] == "VFS-G157"
    assert claim["parent_goal_id"] == "VFS-G071"
    assert claim["task_id"] == "VFS-092"
    assert claim["satisfied"] is True
    assert claim["failure_codes"] == []
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["promotion_authoritative"] is False
    assert claim["semantic_authority"] is False
    assert claim["checks"]["primary_acceptance"] is True
    assert claim["checks"]["optional_premises_have_inclusion_reasons"] is True
    assert claim["checks"]["required_kinds_retained"] is True
    assert claim["checks"]["required_never_truncated"] is True
    assert claim["checks"]["identical_request_reuses_receipt"] is True
    assert claim["checks"]["changed_dependency_invalidates"] is True
    assert claim["contexts"]["primary"]["evidence"] == (
        "vfs/minimal-proof-context@1"
    )
    assert claim["contexts"]["primary"]["embeds_source_bodies"] is False
    assert claim["contexts"]["primary"]["embeds_full_graph"] is False
    assert "optional-premise" not in claim["contexts"]["primary"][
        "included_item_ids"
    ]
    assert "unrelated" not in claim["contexts"]["primary"]["included_item_ids"]
    assert claim["optional_premise_reasons"]["optional-premise"] == (
        "not_in_obligation_dependency_closure"
    )
    assert claim["optional_premise_reasons"]["unrelated"] == (
        "not_in_obligation_dependency_closure"
    )
    assert claim["contexts"]["limit_exceeded"]["status"] == "incomplete"
    assert claim["contexts"]["limit_exceeded"]["required_inputs_truncated"] is False
    assert claim["contexts"]["reuse"]["same_object"] is True
    assert claim["contexts"]["invalidation"]["old_receipt_valid"] is False
    assert claim["invariants"] == list(MINIMAL_PROOF_CONTEXT_INVARIANTS)

    # Discovery alias path.
    alias = prove_minimal_proof_context_evidence()
    assert alias["satisfied"] is True
    assert alias["evidence"] == MINIMAL_PROOF_CONTEXT_EVIDENCE
    assert "vfs/minimal-proof-context@1" in alias["evidence_terms"]


def test_prove_minimal_proof_context_fails_when_required_item_missing() -> None:
    """Required-in-scope items that are absent fail the claim closed."""

    claim = prove_minimal_proof_context(
        required_item_ids=("never-present",),
        probe_limit_truncation=False,
        probe_receipt_reuse=False,
        probe_dependency_invalidation=False,
    )
    assert claim["satisfied"] is False
    assert "primary-acceptance" in claim["failure_codes"]
    assert claim["checks"]["primary_acceptance"] is False
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False


def test_compiled_contexts_pin_minimal_proof_context_evidence() -> None:
    """Compiler receipts and contexts carry the closed evidence pin."""

    result = CodeContractProofContextCompiler().compile(
        default_minimal_proof_context_request()
    )
    assert result.to_dict()["evidence"] == "vfs/minimal-proof-context@1"
    assert result.receipt.to_dict()["evidence"] == "vfs/minimal-proof-context@1"
    assert context_obeys_minimal_proof_context(result)
    assert context_obeys_minimal_proof_context(result.to_dict())
