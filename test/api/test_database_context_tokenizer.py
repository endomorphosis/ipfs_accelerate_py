"""Caller-supplied counting at the real database-context/compiler boundary.

Counters here are authored diagnostics, not production tokenizer bindings.
Optional metadata mirrors are suppressed through the existing public scope;
manifest construction, context compilation and canonical rendering stay real.
"""

from __future__ import annotations

from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompilationError,
    RequiredContextOverflowError,
    render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.context.database_context import (
    ContextBudgetSpec,
    DatabaseContextManifest,
    MemberKind,
    MemberTier,
    TaskContextInput,
    build_database_context_manifest,
    compile_manifest_to_capsule,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    public_replay_without_metadata,
)


@pytest.fixture(autouse=True)
def no_ambient_metadata():
    with public_replay_without_metadata():
        yield


def manifest() -> DatabaseContextManifest:
    return build_database_context_manifest(TaskContextInput(
        task_cid="task:tokenizer-boundary",
        repository_id="repo:tokenizer-boundary",
        tree_id="tree:tokenizer-boundary",
        task_revision="revision:1",
        goal_cid="goal:preserve-required-context",
        task_summary="Keep the current mandatory context and count the prompt",
        unmet_dependencies=({"dependency_id": "dependency:ready", "summary": "dependency ready"},),
        open_obligations=({"obligation_id": "obligation:coverage", "summary": "preserve required coverage"},),
        validations=({"command": "python -m pytest test/api/test_database_context_tokenizer.py"},),
        evidence=({"evidence_id": "evidence:optional", "summary": "current supporting evidence"},),
        budget=ContextBudgetSpec(max_rows=16, max_members=16, max_bytes=16_000,
                                 max_tokens=8_192, page_size=16),
    ))


def required_reference_ids(value: DatabaseContextManifest) -> set[str]:
    return {member.member_id for member in value.included_members()
            if member.tier is MemberTier.INVARIANT and member.kind is not MemberKind.TASK}


def test_default_and_explicit_none_keep_the_existing_capsule_and_manifest():
    value = manifest()
    before = value.to_dict()
    default = compile_manifest_to_capsule(value)
    explicit_none = compile_manifest_to_capsule(value, tokenizer=None)
    assert isinstance(default, ContextCapsule)
    assert default.to_record() == explicit_none.to_record()
    assert value.to_dict() == before
    assert required_reference_ids(value) <= {item.reference_id for item in default.evidence}


def test_supplied_callable_counts_full_render_and_preserves_mandatory_context():
    value = manifest()
    before = value.to_dict()
    original = compile_manifest_to_capsule(value)
    counted: list[str] = []

    def counter(text: str) -> int:
        counted.append(text)
        return max(1, len(text.encode("utf-8")) // 16)

    capsule = compile_manifest_to_capsule(value, tokenizer=counter)
    rendered = render_context_capsule(capsule)
    assert rendered in counted
    assert isinstance(capsule, ContextCapsule)
    assert capsule.input_tokens >= counter(rendered)
    assert capsule.invariant_core == original.invariant_core
    assert capsule.repository_id == value.repository_id
    assert capsule.tree_id == value.tree_id
    assert capsule.policy_id == value.policy_id
    assert capsule.acceptance["manifest_cid"] == value.manifest_cid
    assert capsule.acceptance["cannot_include_secrets"] is True
    assert required_reference_ids(value) <= {item.reference_id for item in capsule.evidence}
    assert value.to_dict() == before
    hints = {member.member_id: member.token_count for member in value.included_members()}
    assert all(item.token_count == hints[item.reference_id] for item in capsule.evidence)


def test_nonadditive_counter_keeps_conservative_hints_separate_from_full_count():
    value = manifest()
    before = value.to_dict()
    capsule = compile_manifest_to_capsule(value, tokenizer=lambda _text: 1)
    assert render_context_capsule(capsule)
    assert capsule.input_tokens > 1
    assert capsule.input_tokens >= sum(item.token_count for item in capsule.evidence)
    assert required_reference_ids(value) <= {item.reference_id for item in capsule.evidence}
    assert value.to_dict() == before


@pytest.mark.parametrize("invalid", [True, -1, None], ids=["boolean", "negative", "uncountable"])
def test_invalid_counter_result_refuses_without_estimate_fallback(invalid: Any):
    value = manifest()
    before = value.to_dict()
    with pytest.raises(ContextCompilationError):
        compile_manifest_to_capsule(value, tokenizer=lambda _text: invalid)
    assert value.to_dict() == before


def test_counter_failure_propagates_without_estimate_fallback():
    value = manifest()
    before = value.to_dict()

    class CounterUnavailable(RuntimeError):
        pass

    def unavailable(_text: str) -> int:
        raise CounterUnavailable("authored counter unavailable")

    with pytest.raises(CounterUnavailable, match="authored counter unavailable"):
        compile_manifest_to_capsule(value, tokenizer=unavailable)
    assert value.to_dict() == before


def test_counter_measured_mandatory_core_overflow_refuses_without_truncation():
    value = manifest()
    before = value.to_dict()
    with pytest.raises(RequiredContextOverflowError, match="invariant"):
        compile_manifest_to_capsule(value, tokenizer=lambda _text: 8_193)
    assert value.to_dict() == before
