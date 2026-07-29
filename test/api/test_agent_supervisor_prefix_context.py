from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    MIN_WARM_PREFIX_REUSE_BPS,
    PREFIX_REUSE_REQUIREMENT_ID,
    ContextCompiler,
    PrefixCacheBoundaryError,
    PrefixCacheDecision,
    PrefixCacheIdentity,
    PrefixCacheKind,
    PrefixContextError,
    PrefixContextResult,
    PrefixReuseReceipt,
    PrefixReuseSource,
    PrefixStableContextCapsule,
    RequiredContextOverflowError,
    compile_prefix_context,
    render_prefix_context,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)


BINDING = {
    "repository_id": "repository:prefix",
    "tree_id": "tree:stable",
    "objective_id": "ASI-G210",
    "objective_revision": "sha256:objective-v1",
    "policy_id": "policy:implementation",
    "policy_revision": "sha256:policy-v1",
    "caller": "supervisor:prefix-test",
    "stage": "implementation",
}
CORE = {
    "goal": {
        "id": "ASI-G210",
        "instruction": "Build prefix-stable stage context",
    },
    "authority": {
        "mode": "proposal",
        "allowed_paths": ["src/context.py"],
    },
    "scope": {
        "task_id": "ASI-095",
        "paths": ["src/context.py"],
    },
    "acceptance": {
        "criteria": [
            "preserve authority and acceptance",
            "reuse at least seventy percent",
        ],
    },
}


def _budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=1_200,
        reserved_output_tokens=160,
        reserved_tool_tokens=40,
        max_items=32,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 16)


def _compiler() -> ContextCompiler:
    return ContextCompiler(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=1_400,
    )


def _reference(
    reference_id: str,
    content_id: str,
    *,
    required: bool = False,
) -> ContextReference:
    return ContextReference(
        reference_id=reference_id,
        kind="prefix-fixture",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id=f"sha256:{content_id}",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        summary=f"Evidence descriptor for {content_id}",
        token_count=20,
        metadata={
            "required": required,
            "priority": 10 if required else 5,
            "coverage_ids": (f"coverage:{reference_id}",),
        },
    )


def _compile(
    compiler: ContextCompiler,
    *,
    evidence: tuple[ContextReference, ...] = (),
    previous: PrefixContextResult | PrefixStableContextCapsule | None = None,
    **overrides: object,
) -> PrefixContextResult:
    values = {
        **BINDING,
        **CORE,
        "provider_id": "provider:test",
        "model_id": "model:test-v1",
        "evidence": evidence,
        "previous": previous,
    }
    values.update(overrides)
    return compiler.compile_prefix_context(**values)


def test_stage_input_has_three_canonical_sections_and_preserves_core() -> None:
    required = _reference("required", "required-v1", required=True)
    optional = _reference("diagnostic", "diagnostic-v1")

    result = _compile(
        _compiler(),
        evidence=(optional, required),
    )
    capsule = result.capsule
    rendered = render_prefix_context(capsule)

    policy_offset = rendered.index('"stable_policy_objective_prefix"')
    task_offset = rendered.index('"stable_task_core"')
    evidence_offset = rendered.index('"volatile_evidence_delta"')
    assert policy_offset < task_offset < evidence_offset
    assert capsule.stable_policy_objective_prefix["goal"] == CORE["goal"]
    assert (
        capsule.stable_policy_objective_prefix["authority"]
        == capsule.context_capsule.authority
    )
    assert (
        capsule.stable_policy_objective_prefix["acceptance"]
        == capsule.context_capsule.acceptance
    )
    assert capsule.stable_task_core["scope"] == capsule.context_capsule.scope
    assert capsule.required_field_names == (
        "goal",
        "authority",
        "scope",
        "acceptance",
    )
    assert {item.reference_id for item in capsule.evidence_delta} == {
        "required",
        "diagnostic",
    }
    assert result.receipt.cache_decision is PrefixCacheDecision.COLD
    assert result.receipt.evidence_claim_references == ()


def test_warm_evidence_delta_reuses_prefix_without_stale_evidence() -> None:
    compiler = _compiler()
    required = _reference("required", "required-v1", required=True)
    cold_evidence = _reference("diagnostic", "diagnostic-old")
    warm_evidence = _reference("diagnostic", "diagnostic-current")
    cold = _compile(
        compiler,
        evidence=(required, cold_evidence),
    )

    warm = _compile(
        compiler,
        evidence=(required, warm_evidence),
        previous=cold,
    )

    assert warm.capsule.stable_prefix_bytes == cold.capsule.stable_prefix_bytes
    assert warm.capsule.semantic_prefix_id == cold.capsule.semantic_prefix_id
    assert warm.capsule.evidence_digest != cold.capsule.evidence_digest
    assert warm.receipt.cache_decision is PrefixCacheDecision.HIT
    assert (
        warm.receipt.reuse_source
        is PrefixReuseSource.CONSERVATIVE_ESTIMATE
    )
    assert warm.receipt.reuse_bps >= MIN_WARM_PREFIX_REUSE_BPS
    assert warm.receipt.evidence_claim_references == (
        PREFIX_REUSE_REQUIREMENT_ID,
    )
    assert warm.receipt.evidence_digest == warm.capsule.evidence_digest
    selected = {
        item.reference_id: item
        for item in warm.capsule.volatile_evidence_delta
    }
    assert (
        selected["diagnostic"].referenced_content_id
        == "sha256:diagnostic-current"
    )
    assert "diagnostic-current" in warm.provider_input
    assert "diagnostic-old" not in warm.provider_input


def test_fallback_tokenizer_derives_a_conservative_qualifying_estimate() -> None:
    compiler = ContextCompiler(
        _budget(),
        provider_context_window=1_400,
    )
    cold = _compile(compiler)
    warm = _compile(
        compiler,
        previous=cold,
        evidence=(_reference("diagnostic", "fallback-current"),),
    )

    assert not compiler.estimator.provider_aware
    assert warm.receipt.reuse_source is PrefixReuseSource.CONSERVATIVE_ESTIMATE
    assert (
        MIN_WARM_PREFIX_REUSE_BPS
        <= warm.receipt.reuse_bps
        < 10_000
    )
    assert warm.receipt.reused_prefix_tokens < (
        warm.receipt.eligible_stable_prefix_tokens
    )
    assert warm.receipt.evidence_claim_references == (
        PREFIX_REUSE_REQUIREMENT_ID,
    )


def test_segment_overhead_defers_optional_evidence_but_never_required_evidence() -> None:
    tight_budget = ContextBudget(
        max_input_tokens=80,
        reserved_output_tokens=10,
        reserved_tool_tokens=5,
        max_items=32,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )
    compiler = ContextCompiler(tight_budget, tokenizer=_tokenizer)
    optional = replace(
        _reference("near-limit", "near-limit"),
        token_count=40,
    )

    base = compiler.compile(**BINDING, **CORE, evidence=(optional,))
    assert tuple(item.reference_id for item in base.capsule.evidence) == (
        "near-limit",
    )
    prefix = _compile(compiler, evidence=(optional,))
    assert prefix.capsule.provider_input_tokens <= 80
    assert prefix.capsule.volatile_evidence_delta == ()
    assert prefix.base_capsule.omitted_reference_ids == ("near-limit",)
    assert (
        prefix.base_capsule.expansion_references[0].referenced_content_id
        == optional.referenced_content_id
    )

    required = replace(
        optional,
        tier=ContextTier.INVARIANT,
        metadata={
            **dict(optional.metadata),
            "required": True,
        },
    )
    with pytest.raises(RequiredContextOverflowError, match="required evidence"):
        _compile(compiler, evidence=(required,))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repository_id", "repository:other"),
        ("tree_id", "tree:other"),
        ("objective_id", "ASI-G211"),
        ("objective_revision", "sha256:objective-v2"),
        ("policy_id", "policy:other"),
        ("policy_revision", "sha256:policy-v2"),
        ("caller", "supervisor:other"),
        ("stage", "validation"),
        ("goal", {"id": "ASI-G210", "instruction": "changed"}),
        ("authority", {"mode": "diagnostic", "allowed_paths": ["src"]}),
        ("scope", {"task_id": "ASI-095", "paths": ["other.py"]}),
        ("acceptance", {"criteria": ["changed"]}),
        ("provider_id", "provider:other"),
        ("model_id", "model:other"),
    ),
)
def test_semantic_prefix_dependencies_invalidate_exactly(
    field: str, value: object
) -> None:
    compiler = _compiler()
    cold = _compile(compiler)

    warm = _compile(compiler, previous=cold, **{field: value})

    assert warm.receipt.cache_decision is PrefixCacheDecision.INVALIDATED
    assert warm.receipt.reused_prefix_tokens == 0
    assert warm.receipt.invalidated_dependencies == (field,)
    assert warm.receipt.evidence_claim_references == ()
    if field in {"provider_id", "model_id"}:
        assert warm.capsule.semantic_prefix_id == cold.capsule.semantic_prefix_id
    else:
        assert warm.capsule.semantic_prefix_id != cold.capsule.semantic_prefix_id


def test_native_prompt_and_kv_cache_usage_bind_actual_reuse() -> None:
    compiler = _compiler()
    cold = _compile(compiler)
    prompt_actual = cold.capsule.stable_prefix_tokens

    prompt = _compile(
        compiler,
        previous=cold,
        provider_cache_id="prompt-cache:abc",
        provider_cache_kind=PrefixCacheKind.PROMPT_CACHE,
        provider_reused_tokens=prompt_actual,
    )
    assert (
        prompt.receipt.reuse_source
        is PrefixReuseSource.PROVIDER_PROMPT_CACHE
    )
    assert prompt.receipt.provider_reused_tokens == prompt_actual
    assert (
        prompt.receipt.cache_identity.provider_cache_id
        == "prompt-cache:abc"
    )
    assert prompt.receipt.cache_identity.cache_kind is PrefixCacheKind.PROMPT_CACHE
    assert prompt.receipt.reuse_bps == 10_000

    kv = _compile(
        compiler,
        previous=prompt,
        provider_cache_id="kv-cache:xyz",
        provider_cache_kind="provider_kv_cache",
        provider_reused_tokens=max(
            1, prompt.capsule.stable_prefix_tokens * 3 // 4
        ),
    )
    assert kv.receipt.reuse_source is PrefixReuseSource.PROVIDER_KV_CACHE
    assert kv.receipt.cache_identity.cache_kind is PrefixCacheKind.KV_CACHE
    assert kv.receipt.provider_reused_tokens is not None
    assert kv.receipt.reuse_bps >= MIN_WARM_PREFIX_REUSE_BPS


def test_provider_cache_identity_uses_estimate_when_native_count_is_absent() -> None:
    compiler = _compiler()
    cold = _compile(compiler)

    warm = _compile(
        compiler,
        previous=cold,
        provider_cache_id="prompt-cache:no-counter",
        provider_cache_kind=PrefixCacheKind.PROMPT_CACHE,
    )

    assert (
        warm.receipt.cache_identity.cache_kind
        is PrefixCacheKind.PROMPT_CACHE
    )
    assert (
        warm.receipt.cache_identity.provider_cache_id
        == "prompt-cache:no-counter"
    )
    assert warm.receipt.provider_reused_tokens is None
    assert (
        warm.receipt.reuse_source
        is PrefixReuseSource.CONSERVATIVE_ESTIMATE
    )
    assert warm.receipt.reuse_bps >= MIN_WARM_PREFIX_REUSE_BPS


def test_provider_reuse_is_prohibited_across_authority_and_target_boundaries() -> None:
    compiler = _compiler()
    cold = _compile(compiler)
    cache_args = {
        "previous": cold,
        "provider_cache_id": "prompt-cache:stale",
        "provider_cache_kind": PrefixCacheKind.PROMPT_CACHE,
        "provider_reused_tokens": cold.capsule.stable_prefix_tokens,
    }

    with pytest.raises(PrefixCacheBoundaryError, match="dependency changed"):
        _compile(
            compiler,
            authority={"mode": "merge", "allowed_paths": ["src"]},
            **cache_args,
        )
    with pytest.raises(PrefixCacheBoundaryError, match="dependency changed"):
        _compile(
            compiler,
            scope={"task_id": "ASI-999", "paths": ["other.py"]},
            **cache_args,
        )
    with pytest.raises(PrefixCacheBoundaryError, match="warm predecessor"):
        _compile(
            compiler,
            provider_cache_id="prompt-cache:unbound",
            provider_cache_kind=PrefixCacheKind.PROMPT_CACHE,
            provider_reused_tokens=cold.capsule.stable_prefix_tokens,
        )


def test_prefix_contracts_round_trip_and_reject_forged_reuse() -> None:
    compiler = _compiler()
    cold = _compile(
        compiler,
        evidence=(_reference("diagnostic", "v1"),),
    )
    warm = _compile(
        compiler,
        evidence=(_reference("diagnostic", "v2"),),
        previous=cold,
    )

    assert (
        PrefixStableContextCapsule.from_json(warm.capsule.to_json())
        == warm.capsule
    )
    assert (
        PrefixCacheIdentity.from_json(
            warm.receipt.cache_identity.to_json()
        )
        == warm.receipt.cache_identity
    )
    assert PrefixReuseReceipt.from_json(warm.receipt.to_json()) == warm.receipt

    forged_wire = warm.receipt.to_dict()
    forged_wire["reused_prefix_tokens"] = (
        warm.receipt.eligible_stable_prefix_tokens + 1
    )
    with pytest.raises(PrefixContextError, match="exceed"):
        PrefixReuseReceipt.from_dict(forged_wire)

    forged_receipt = replace(
        warm.receipt,
        evidence_digest="sha256:" + "0" * 64,
    )
    with pytest.raises(PrefixContextError, match="detached"):
        replace(warm, receipt=forged_receipt)


def test_prefix_compilation_is_deterministic_and_wrapper_uses_same_contract() -> None:
    first = _reference("first", "one")
    second = _reference("second", "two")
    compiler = _compiler()
    forward = _compile(compiler, evidence=(first, second))
    reverse = _compile(compiler, evidence=(second, first))

    assert forward.capsule == reverse.capsule
    assert forward.receipt == reverse.receipt
    wrapped = compile_prefix_context(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=1_400,
        provider_id="provider:test",
        model_id="model:test-v1",
        **BINDING,
        **CORE,
        evidence=(second, first),
    )
    assert wrapped.capsule == forward.capsule
    assert wrapped.receipt == forward.receipt
