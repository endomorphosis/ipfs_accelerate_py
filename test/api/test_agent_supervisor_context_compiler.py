from __future__ import annotations

from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,
    CalibratedTokenEstimator,
    ContextCompilationError,
    ContextCompilationReceipt,
    ContextCompileResult,
    ContextCompiler,
    ExclusionReason,
    InclusionReason,
    RequiredContextBudgetEvidence,
    RequiredContextOverflowError,
    compile_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context_contracts import (
    ContextBudget,
    ContextContractError,
    ContextReference,
    ContextTier,
)


BINDING = {
    "repository_id": "repo:example",
    "tree_id": "tree:abc",
    "objective_id": "ASI-G091",
    "objective_revision": "sha256:objective",
    "policy_id": "policy:supervisor",
    "policy_revision": "sha256:policy",
    "caller": "supervisor:test",
    "stage": "planning",
}
CORE = {
    "goal": {"id": "ASI-G091", "summary": "Preserve required context"},
    "authority": {"mode": "proposal", "allowed_paths": ["src"]},
    "scope": {"paths": ["src/context.py"], "symbols": ["compile"]},
    "acceptance": {"criteria": ["required fields remain complete"]},
}


def _budget(max_input_tokens: int = 220) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=40,
        reserved_tool_tokens=10,
        max_items=16,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )


def _reference(
    reference_id: str,
    tokens: int,
    *,
    required: bool = False,
    priority: int = 0,
    summary: str = "",
) -> ContextReference:
    return ContextReference(
        reference_id=reference_id,
        kind="test-evidence",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id=f"sha256:{reference_id}",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        summary=summary,
        token_count=tokens,
        metadata={
            "required": required,
            "priority": priority,
            "coverage_ids": (f"coverage:{reference_id}",),
        },
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 24)


def _compile(
    *,
    budget: ContextBudget | None = None,
    evidence: tuple[ContextReference, ...] = (),
    provider_context_window: int = 270,
):
    compiler = ContextCompiler(
        budget or _budget(),
        tokenizer=_tokenizer,
        provider_context_window=provider_context_window,
    )
    return compiler.compile(**BINDING, **CORE, evidence=evidence)


def test_required_fields_and_references_survive_effective_provider_budget() -> None:
    required = _reference("required", 16, required=True)
    optional = _reference("optional", 18, priority=10)

    result = _compile(evidence=(optional, required))

    assert result.capsule.required_field_names == (
        "goal",
        "authority",
        "scope",
        "acceptance",
    )
    assert result.capsule.invariant_core["goal"] == CORE["goal"]
    assert tuple(item.reference_id for item in result.capsule.evidence) == (
        "optional",
        "required",
    )
    assert result.capsule.budget.max_input_tokens == 220
    assert result.capsule.input_tokens <= 220
    assert result.receipt.evidence is not None
    assert (
        result.receipt.evidence.requirement_id
        == REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID
    )
    assert result.receipt.evidence_claim_references == (
        REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,
    )
    assert result.receipt.evidence.required_reference_ids == ("required",)
    assert result.receipt.evidence.artifact_digest.startswith("sha256:")


def test_provider_window_subtracts_output_and_tool_reserves() -> None:
    budget = _budget(max_input_tokens=500)
    compiler = ContextCompiler(
        budget,
        tokenizer=_tokenizer,
        provider_context_window=180,
    )

    assert compiler.effective_input_limit == 130
    assert compiler.effective_budget.max_input_tokens == 130

    direct_ceiling = ContextCompiler(
        budget,
        tokenizer=_tokenizer,
        provider_context_window=900,
        provider_max_input_tokens=111,
    )
    assert direct_ceiling.effective_input_limit == 111


def test_required_context_fails_closed_instead_of_truncating() -> None:
    tiny = ContextCompiler(
        _budget(max_input_tokens=1),
        tokenizer=lambda text: len(text),
    )
    with pytest.raises(
        RequiredContextOverflowError, match="goal/authority/scope/acceptance"
    ):
        tiny.compile(**BINDING, **CORE)

    base = _compile().capsule.input_tokens
    required = _reference("too-large", base + 500, required=True)
    with pytest.raises(
        RequiredContextOverflowError, match="required evidence"
    ):
        _compile(evidence=(required,))


def test_canonical_provider_input_defeats_forged_reference_token_count() -> None:
    compiler = ContextCompiler(
        _budget(max_input_tokens=100),
        tokenizer=lambda text: max(1, len(text.encode("utf-8")) // 16),
    )
    understated = _reference(
        "understated",
        1,
        required=True,
        summary="x" * 8_000,
    )

    with pytest.raises(
        RequiredContextOverflowError, match="required evidence"
    ):
        compiler.compile(**BINDING, **CORE, evidence=(understated,))

    valid = ContextCompiler(
        _budget(max_input_tokens=500),
        tokenizer=_tokenizer,
    ).compile(
        **BINDING,
        **CORE,
        evidence=(_reference("required", 1, required=True),),
    )
    verifier = ContextCompiler(
        _budget(max_input_tokens=500),
        tokenizer=_tokenizer,
    )
    assert verifier.estimate_capsule_input(valid.capsule) == (
        valid.capsule.input_tokens
    )
    assert dict(valid.capsule.provider_input_payload)["evidence"]


def test_optional_evidence_has_deterministic_ranking_and_decisions() -> None:
    references = (
        _reference("low", 200, priority=1),
        _reference("high-b", 40, priority=10),
        _reference("high-a", 40, priority=10),
    )
    forward = _compile(evidence=references)
    reverse = _compile(evidence=tuple(reversed(references)))

    assert forward.capsule == reverse.capsule
    assert forward.receipt == reverse.receipt
    assert forward.decisions == reverse.decisions
    included = {
        item.reference_id: item
        for item in forward.decisions
        if item.included
    }
    assert included["high-a"].reason is InclusionReason.RANKED_FIT
    assert included["high-b"].reason is InclusionReason.RANKED_FIT
    omitted = {
        item.reference_id: item
        for item in forward.decisions
        if not item.included
    }
    assert omitted
    assert set(item.reason for item in omitted.values()) == {
        ExclusionReason.TOKEN_BUDGET
    }
    assert forward.capsule.truncated
    assert {
        item.reference_id for item in forward.capsule.expansion_references
    } == set(omitted)
    assert all(
        item.tier is ContextTier.EXPANSION
        for item in forward.capsule.expansion_references
    )


def test_compilation_receipt_is_canonical_bounded_and_tamper_evident() -> None:
    result = _compile(
        evidence=(
            _reference(
                "required",
                12,
                required=True,
                summary="bounded summary, not source content",
            ),
        )
    )
    receipt = result.receipt

    assert ContextCompilationReceipt.from_json(receipt.to_json()) == receipt
    assert (
        RequiredContextBudgetEvidence.from_json(
            receipt.evidence.to_json()  # type: ignore[union-attr]
        )
        == receipt.evidence
    )
    encoded = json.dumps(receipt.to_dict(), sort_keys=True)
    assert "raw_prompt" not in encoded
    assert "decoded_output" not in encoded
    assert "source_body" not in encoded
    assert len(receipt.canonical_bytes()) < 262_144

    forged = receipt.to_record()
    forged["input_tokens"] += 1
    with pytest.raises(ContextCompilationError, match="bound|identity"):
        ContextCompilationReceipt.from_dict(forged)

    forged = receipt.to_dict()
    forged["evidence_claim_references"] = ()
    with pytest.raises(ContextCompilationError, match="claim"):
        ContextCompilationReceipt.from_dict(forged)


def test_compilation_result_revalidates_capsule_witness_and_decisions() -> None:
    result = _compile(
        evidence=(
            _reference("required", 12, required=True),
            _reference("optional", 12),
        )
    )
    assert result.receipt.evidence is not None

    forged_digest = replace(
        result.receipt,
        evidence=replace(
            result.receipt.evidence,
            artifact_digest="sha256:" + "0" * 64,
        ),
    )
    with pytest.raises(ContextCompilationError, match="artifact digest"):
        ContextCompileResult(
            result.capsule,
            forged_digest,
            result.decisions,
        )

    forged_references = replace(
        result.receipt,
        evidence=replace(
            result.receipt.evidence,
            required_reference_ids=(),
        ),
    )
    with pytest.raises(ContextCompilationError, match="required references"):
        ContextCompileResult(
            result.capsule,
            forged_references,
            result.decisions,
        )

    with pytest.raises(ContextCompilationError, match="bound to its receipt"):
        replace(
            result.receipt,
            tree_id="tree:stale",
        )
    forged_objective = replace(result.receipt, objective_id="ASI-G999")
    with pytest.raises(ContextCompilationError, match="complete compiled"):
        ContextCompileResult(
            result.capsule,
            forged_objective,
            result.decisions,
        )

    forged_decisions = tuple(
        replace(item, reason=InclusionReason.RANKED_FIT)
        if item.reference_id == "required"
        else item
        for item in result.decisions
    )
    forged_receipt = replace(result.receipt, decisions=forged_decisions)
    with pytest.raises(ContextCompilationError, match="selected reference"):
        ContextCompileResult(
            result.capsule,
            forged_receipt,
            forged_decisions,
        )


def test_required_evidence_cannot_be_deferred_as_expansion_handle() -> None:
    required_expansion = ContextReference(
        reference_id="required-expansion",
        kind="test-evidence",
        tier=ContextTier.EXPANSION,
        referenced_content_id="sha256:required-expansion",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        token_count=10,
        metadata={"required": True},
    )
    capsule = _compile().capsule

    with pytest.raises(
        ContextContractError, match="required evidence cannot be deferred"
    ):
        replace(
            capsule,
            expansion_references=(required_expansion,),
            truncated=True,
            omissions=("required-expansion:token_budget",),
        )


def test_estimator_uses_provider_tokenizer_and_records_fallback_calibration() -> None:
    provider = CalibratedTokenEstimator(lambda text: text.split())
    assert provider.provider_aware
    assert provider.estimate("one two three") == 3
    assert provider.error_bps == 0

    fallback = CalibratedTokenEstimator(chars_per_token=4)
    assert fallback.estimate("abcdefgh") == 2
    assert fallback.error_bps == 10_000
    fallback.calibrate("abcdefgh", actual_tokens=4)
    assert fallback.calibration_samples == 1
    assert fallback.error_bps == 5_000
    assert fallback.estimate("abcdefgh") == 4


def test_top_level_compiler_wrapper_preserves_contract_and_rejects_bad_inputs() -> None:
    result = compile_context_capsule(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=270,
        **BINDING,
        **CORE,
        evidence=(_reference("required", 10, required=True),),
    )
    assert result.receipt.capsule_id == result.capsule.capsule_id

    with pytest.raises(ContextCompilationError, match="tokenizer or estimator"):
        ContextCompiler(
            _budget(),
            tokenizer=_tokenizer,
            estimator=CalibratedTokenEstimator(),
        )

    with pytest.raises(ContextCompilationError, match="conflicting duplicate"):
        _compile(
            evidence=(
                _reference("duplicate", 10),
                replace(_reference("duplicate", 10), summary="different"),
            )
        )
