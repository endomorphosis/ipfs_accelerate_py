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
    assert result.delta_capsule.parent_capsule_id == parent.capsule_id
    assert tuple(
        item.reference_id for item in result.delta_capsule.evidence
    ) == ("diagnostic",)
    assert {
        item.reference_id for item in result.reconstructed_capsule.evidence
    } == {"required", "diagnostic"}
    assert result.receipt.delta_tokens < result.receipt.full_replay_tokens
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


def test_reconstruction_rejects_stale_parent_and_mutated_invariant_core() -> None:
    compiler, parent, required, _ = _parent()
    result = compiler.compile_delta(
        parent,
        evidence=(required, _reference("diagnostic", "new", 20)),
    )

    stale_parent = replace(parent, objective_revision="sha256:other")
    with pytest.raises(ContextDeltaError, match="not bound"):
        reconstruct_context(stale_parent, result.delta_capsule)

    mutated_delta = replace(
        result.delta_capsule,
        goal={"id": "ASI-G092", "summary": "mutated"},
    )
    with pytest.raises(ContextDeltaError, match="immutable goal"):
        reconstruct_context(parent, mutated_delta)


def test_delta_must_be_smaller_than_full_replay() -> None:
    compiler = ContextCompiler(
        _budget(),
        tokenizer=lambda text: 1,
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
