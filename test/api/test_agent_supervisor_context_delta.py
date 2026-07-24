from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    DELTA_RETRY_EVIDENCE_ID,
    ContextCompiler,
    ContextDeltaError,
    ContextDeltaReceipt,
    DeltaRetryContextEvidence,
    ExclusionReason,
    InclusionReason,
    compile_context_delta,
    expand_context,
    reconstruct_context,
)
from ipfs_accelerate_py.agent_supervisor.context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextContractError,
    ContextDeltaCapsule,
    ContextReference,
    ContextTier,
)


BINDING = {
    "repository_id": "repo:delta",
    "tree_id": "tree:current",
    "objective_id": "ASI-G092",
    "objective_revision": "sha256:objective",
    "policy_id": "policy:supervisor",
    "policy_revision": "sha256:policy",
    "caller": "supervisor:test",
    "stage": "implementation",
}
CORE = {
    "goal": {"id": "ASI-G092", "summary": "Use retry deltas"},
    "authority": {"mode": "proposal", "allowed_paths": ["src"]},
    "scope": {"paths": ["src/context.py"]},
    "acceptance": {"criteria": ["coverage remains complete"]},
}


def _budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=600,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=32,
        max_serialized_bytes=262_144,
    )


def _reference(
    reference_id: str,
    content: str,
    tokens: int,
    *,
    required: bool = False,
) -> ContextReference:
    return ContextReference(
        reference_id=reference_id,
        kind="test-evidence",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id=f"sha256:{content}",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        token_count=tokens,
        metadata={
            "required": required,
            "coverage_ids": (f"coverage:{reference_id}",),
        },
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 32)


def _compiler() -> ContextCompiler:
    return ContextCompiler(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=720,
    )


def _parent() -> tuple[
    ContextCompiler,
    ContextCapsule,
    ContextReference,
    ContextReference,
]:
    compiler = _compiler()
    required = _reference("required", "old-required", 80, required=True)
    optional = _reference("diagnostic", "old-diagnostic", 160)
    result = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, optional),
    )
    return compiler, result.capsule, required, optional


def test_delta_transmits_changes_and_preserves_required_coverage() -> None:
    compiler, parent, required, _ = _parent()
    changed = _reference("diagnostic", "new-diagnostic", 30)

    result = compiler.compile_delta(
        parent,
        evidence=(required, changed),
    )

    assert result.delta_capsule.is_delta
    assert ContextDeltaCapsule.from_json(
        result.delta_capsule.to_json()
    ) == result.delta_capsule
    assert result.delta_capsule.parent_capsule_id == parent.capsule_id
    assert tuple(
        item.reference_id for item in result.delta_capsule.evidence
    ) == ("diagnostic",)
    assert {
        item.reference_id for item in result.reconstructed_capsule.evidence
    } == {"required", "diagnostic"}
    assert result.receipt.delta_tokens < result.receipt.full_replay_tokens
    assert result.receipt.delta_tokens == compiler.estimator.estimate(
        result.delta_capsule.to_record()
    )
    assert result.receipt.full_replay_tokens == max(
        result.reconstructed_capsule.input_tokens,
        compiler.estimator.estimate(
            result.reconstructed_capsule.provider_input_payload
        ),
    )
    assert result.receipt.evidence is not None
    assert result.receipt.evidence.requirement_id == DELTA_RETRY_EVIDENCE_ID
    assert result.receipt.evidence_claim_references == (
        DELTA_RETRY_EVIDENCE_ID,
    )
    assert set(result.receipt.evidence.required_coverage_ids).issubset(
        result.receipt.evidence.reconstructed_coverage_ids
    )
    decisions = {item.reference_id: item for item in result.decisions}
    assert decisions["diagnostic"].reason is InclusionReason.CHANGED
    assert decisions["required"].reason is ExclusionReason.UNCHANGED


def test_delta_receipt_and_witness_round_trip_and_reject_forged_claims() -> None:
    compiler, parent, required, _ = _parent()
    result = compiler.compile_delta(
        parent,
        evidence=(
            required,
            _reference("diagnostic", "fixed", 20),
        ),
    )

    assert ContextDeltaReceipt.from_json(
        result.receipt.to_json()
    ) == result.receipt
    assert DeltaRetryContextEvidence.from_json(
        result.receipt.evidence.to_json()  # type: ignore[union-attr]
    ) == result.receipt.evidence

    forged = result.receipt.to_record()
    forged["delta_tokens"] += 1
    with pytest.raises(ContextDeltaError, match="bound|identity"):
        ContextDeltaReceipt.from_dict(forged)

    forged = result.receipt.to_dict()
    forged["evidence_claim_references"] = ()
    with pytest.raises(ContextDeltaError, match="claim"):
        ContextDeltaReceipt.from_dict(forged)

    assert result.receipt.evidence is not None
    forged_evidence = replace(
        result.receipt.evidence,
        artifact_digest="sha256:" + "0" * 64,
    )
    forged_receipt = replace(result.receipt, evidence=forged_evidence)
    with pytest.raises(ContextDeltaError, match="artifact digest"):
        replace(result, receipt=forged_receipt)


def test_unchanged_retry_and_required_evidence_loss_fail_closed() -> None:
    compiler, parent, required, optional = _parent()

    with pytest.raises(ContextDeltaError, match="changed or explicitly requested"):
        compiler.compile_delta(
            parent,
            evidence=(required, optional),
        )

    with pytest.raises(ContextDeltaError, match="drops required"):
        compiler.compile_delta(
            parent,
            evidence=(_reference("diagnostic", "new", 20),),
        )


def test_requested_expansion_is_parent_bound_and_deterministic() -> None:
    compiler = _compiler()
    required = _reference("required", "required", 50, required=True)
    omitted = _reference("large", "large", 700)
    parent_result = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, omitted),
    )
    parent = parent_result.capsule
    assert parent.expansion_references

    smaller = _reference("large", "large-summary", 20)
    result = expand_context(compiler, parent, (smaller,))

    assert result.delta_capsule.parent_capsule_id == parent.capsule_id
    decision = {
        item.reference_id: item for item in result.decisions
    }["large"]
    assert decision.reason in {
        InclusionReason.CHANGED,
        InclusionReason.REQUESTED,
    }
    rebuilt = reconstruct_context(parent, result.delta_capsule)
    assert rebuilt == result.reconstructed_capsule
    assert {
        item.reference_id for item in rebuilt.evidence
    } == {"required", "large"}


def test_reconstruction_rejects_stale_parent_and_delta_omits_invariant_core() -> None:
    compiler, parent, required, _ = _parent()
    result = compiler.compile_delta(
        parent,
        evidence=(required, _reference("diagnostic", "new", 20)),
    )

    stale_parent = replace(parent, objective_revision="sha256:other")
    with pytest.raises(ContextDeltaError, match="not bound"):
        reconstruct_context(stale_parent, result.delta_capsule)

    wire = result.delta_capsule.to_dict()
    assert {"goal", "authority", "scope", "acceptance"}.isdisjoint(wire)
    wire["goal"] = {"id": "ASI-G092", "summary": "smuggled replay"}
    with pytest.raises(ContextContractError, match="unsupported fields"):
        ContextDeltaCapsule.from_dict(wire)


def test_requested_unchanged_reference_is_not_masqueraded_as_changed() -> None:
    compiler, parent, required, optional = _parent()

    result = compiler.compile_delta(
        parent,
        evidence=(required, optional),
        requested_reference_ids=("diagnostic",),
    )

    assert result.receipt.evidence is not None
    assert result.receipt.evidence.changed_reference_ids == ()
    assert result.receipt.evidence.requested_reference_ids == ("diagnostic",)
    assert result.delta_capsule.requested_reference_ids == ("diagnostic",)
    decision = {item.reference_id: item for item in result.decisions}
    assert decision["diagnostic"].reason is InclusionReason.REQUESTED


def test_delta_rejects_requiredness_downgrade_and_full_context_overflow() -> None:
    compiler, parent, required, optional = _parent()
    downgraded = ContextReference(
        reference_id=required.reference_id,
        kind=required.kind,
        tier=ContextTier.EVIDENCE,
        referenced_content_id=required.referenced_content_id,
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        token_count=required.token_count,
        metadata={
            "required": False,
            "coverage_ids": required.coverage_ids,
        },
    )
    with pytest.raises(ContextDeltaError, match="downgrades"):
        compiler.compile_delta(
            parent,
            evidence=(downgraded, optional),
            requested_reference_ids=("diagnostic",),
        )
    coverage_losing = ContextReference(
        reference_id=required.reference_id,
        kind=required.kind,
        tier=ContextTier.INVARIANT,
        referenced_content_id="sha256:coverage-losing",
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        token_count=required.token_count,
        metadata={"required": True},
    )
    with pytest.raises(ContextDeltaError, match="loses required coverage"):
        compiler.compile_delta(
            parent,
            evidence=(coverage_losing, optional),
        )

    tight_budget = ContextBudget(
        max_input_tokens=100,
        reserved_output_tokens=0,
        reserved_tool_tokens=0,
    )
    tight = ContextCompiler(tight_budget, tokenizer=_tokenizer)
    base_only = tight.compile(**BINDING, **CORE).capsule.input_tokens
    base_required = _reference("required", "required", 1, required=True)
    tight_parent = tight.compile(
        **BINDING, **CORE, evidence=(base_required,)
    ).capsule
    overflowing = _reference(
        "new-required",
        "new-required",
        100 - base_only + 1,
        required=True,
    )
    with pytest.raises(ContextDeltaError, match="full context exceeds"):
        tight.compile_delta(
            tight_parent,
            evidence=(base_required, overflowing),
        )


def test_reconstruction_preserves_expansion_handles_and_rejects_token_forgery() -> None:
    compiler = _compiler()
    required = _reference("required", "required", 50, required=True)
    selected_later = _reference("selected-later", "large-a", 700)
    still_deferred = _reference("still-deferred", "large-b", 700)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, selected_later, still_deferred),
    ).capsule
    result = expand_context(
        compiler,
        parent,
        (_reference("selected-later", "summary-a", 20),),
    )

    assert tuple(
        item.reference_id
        for item in result.reconstructed_capsule.expansion_references
    ) == ("still-deferred",)
    forged = replace(
        result.delta_capsule,
        reconstructed_input_tokens=sum(
            item.token_count for item in result.reconstructed_capsule.evidence
        ),
    )
    with pytest.raises(ContextDeltaError, match="omits inherited core"):
        reconstruct_context(parent, forged)


def test_new_required_candidate_is_included_in_witness_coverage() -> None:
    compiler, parent, required, _ = _parent()
    newly_required = _reference("new-required", "new", 20, required=True)

    result = compiler.compile_delta(
        parent,
        evidence=(required, newly_required),
    )

    assert result.receipt.evidence is not None
    assert set(result.receipt.evidence.required_coverage_ids) == {
        "coverage:required",
        "coverage:new-required",
    }


def test_delta_must_be_smaller_than_full_replay() -> None:
    compiler = ContextCompiler(
        _budget(),
        tokenizer=lambda text: (
            100 if "context-delta-capsule@1" in text else 1
        ),
    )
    required = _reference("required", "old", 1, required=True)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required,),
    ).capsule

    with pytest.raises(ContextDeltaError, match="fewer tokens"):
        compiler.compile_delta(
            parent,
            evidence=(_reference("required", "new", 10, required=True),),
        )


def test_top_level_delta_wrapper_binds_the_same_contract() -> None:
    _, parent, required, _ = _parent()
    result = compile_context_delta(
        _budget(),
        parent,
        tokenizer=_tokenizer,
        provider_context_window=720,
        evidence=(required, _reference("diagnostic", "new", 20)),
    )

    assert result.receipt.parent_capsule_id == parent.capsule_id
    assert result.receipt.delta_capsule_id == result.delta_capsule.capsule_id
