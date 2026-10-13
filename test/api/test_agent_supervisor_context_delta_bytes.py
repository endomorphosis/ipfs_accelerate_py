"""Provider byte ceilings apply to both representations of a retry delta."""

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler,
    ContextDeltaBudgetError,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
    canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    public_replay_without_metadata,
)


@pytest.fixture(autouse=True)
def _no_incidental_metadata():
    with public_replay_without_metadata():
        yield


def _compiler(limit=None):
    return ContextCompiler(
        ContextBudget(
            max_input_tokens=20_000,
            max_items=16,
            max_item_bytes=65_536,
            max_serialized_bytes=262_144,
        ),
        tokenizer=lambda text: max(1, len(text.encode("utf-8")) // 4),
        provider_max_input_bytes=limit,
    )


def _scenario():
    compiler = _compiler()
    required = ContextReference(
        "required-safety",
        "security-constraint",
        tier=ContextTier.INVARIANT,
        repository_id="repo:delta-bytes",
        tree_id="tree:current",
        summary="Preserve every access check and the complete safety constraint. " * 45,
        metadata={"coverage_ids": ("security:access-check",)},
    )
    previous = ContextReference(
        "diagnostic",
        "diagnostic",
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        summary="The current diagnostic needs expansion.",
    )
    changed = replace(
        previous,
        summary="Detailed repair evidence: café, 東京, and 🧭. " * 45,
    )
    parent = compiler.compile(
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        objective_id="objective:repair",
        objective_revision="revision:1",
        policy_id="policy:proposal",
        policy_revision="revision:1",
        caller="test:delta-bytes",
        stage="implementation",
        goal="Repair the source without weakening its constraints.",
        authority="Propose source changes only.",
        scope={"paths": ["src/example.py"]},
        acceptance="All required safety checks remain intact.",
        evidence=(required, previous),
    ).capsule
    candidates = (required, changed)
    result = compiler.compile_delta(parent, evidence=candidates)
    return parent, candidates, result


def _wire(result):
    return (
        canonical_context_json_bytes(result.delta_capsule.to_record()),
        canonical_context_json_bytes(result.reconstructed_capsule.provider_input_payload),
    )


@pytest.mark.parametrize("operation", ("compile", "verify"))
@pytest.mark.parametrize("representation", ("transmitted", "reconstructed"))
def test_delta_rejects_one_byte_over_each_provider_ceiling(operation, representation):
    parent, candidates, result = _scenario()
    transmitted, reconstructed = _wire(result)
    assert len(transmitted) < len(reconstructed)
    if representation == "transmitted":
        limit = len(transmitted) - 1
        message = "retry delta exceeds the provider input-byte budget"
    else:
        limit = len(reconstructed) - 1
        assert len(transmitted) <= limit
        message = "reconstructed full context exceeds the provider input-byte budget"
    constrained = _compiler(limit)

    # Reverification must enforce the current consumer's limit even though
    # the immutable receipt was produced by a compiler without a byte ceiling.
    with pytest.raises(ContextDeltaBudgetError, match=message):
        if operation == "compile":
            constrained.compile_delta(parent, evidence=candidates)
        else:
            constrained.verify_delta_result(result)


def test_delta_exact_byte_boundary_preserves_required_context():
    parent, candidates, original = _scenario()
    transmitted, reconstructed = _wire(original)
    compiler = _compiler(max(len(transmitted), len(reconstructed)))

    result = compiler.compile_delta(parent, evidence=candidates)

    assert compiler.verify_delta_result(result) is result
    assert compiler.verify_delta_result(original) is original
    assert result.reconstructed_capsule.invariant_core == parent.invariant_core
    assert result.receipt.evidence.required_coverage_ids == ("security:access-check",)
    assert _wire(result) == (transmitted, reconstructed)


def test_delta_provider_ceiling_counts_utf8_bytes_not_characters():
    parent, candidates, result = _scenario()
    transmitted, reconstructed = _wire(result)
    limit = max(len(transmitted), len(reconstructed.decode("utf-8")))
    assert limit < len(reconstructed)
    compiler = _compiler(limit)

    with pytest.raises(ContextDeltaBudgetError, match="reconstructed full context"):
        compiler.compile_delta(parent, evidence=candidates)
    with pytest.raises(ContextDeltaBudgetError, match="reconstructed full context"):
        compiler.verify_delta_result(result)


def test_delta_without_provider_byte_ceiling_remains_valid():
    parent, candidates, original = _scenario()
    compiler = _compiler()

    result = compiler.compile_delta(parent, evidence=candidates)

    assert compiler.verify_delta_result(result) is result
    assert compiler.verify_delta_result(original) is original
    assert _wire(result) == _wire(original)


def test_reverification_does_not_trust_replaced_producer_verifier():
    _parent, _candidates, result = _scenario()
    transmitted, reconstructed = _wire(result)
    assert len(transmitted) < len(reconstructed)
    # The producer field is not an authority for the consumer's own byte
    # allowance, including when a caller replaces that field on a valid result.
    claimed = replace(result, verifier=_compiler(len(reconstructed) + 100_000))
    constrained = _compiler(len(reconstructed) - 1)

    with pytest.raises(ContextDeltaBudgetError, match="reconstructed full context"):
        constrained.verify_delta_result(claimed)
