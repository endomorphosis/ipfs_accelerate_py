"""Independent current-tree checks for DOEP-063 accelerate freshness and selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ChangedTreeContextError,
    ContextCompiler,
    ExclusionReason,
    InclusionReason,
    compile_retry_context,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
COMPILER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-063.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
    "test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-063.json",
)
TASK_CID = "sha256:a117a64676549c4a29666f6b0d90e407c08cbaa362d8454ee50a0647094599e9"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BINDING = {
    "repository_id": "repo:doep-063",
    "tree_id": "tree:current",
    "objective_id": "DOEP-G070",
    "objective_revision": "sha256:objective",
    "policy_id": "policy:supervisor",
    "policy_revision": "sha256:policy",
    "caller": "supervisor:doep-063",
    "stage": "planning",
}
CORE = {
    "goal": {"id": "DOEP-G070.S2", "summary": "Select current evidence"},
    "authority": {"mode": "proposal", "allowed_paths": ["src"]},
    "scope": {"paths": ["src/context.py"], "symbols": ["compile"]},
    "acceptance": {"criteria": ["stale trees are rejected"]},
}


def _budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=220,
        reserved_output_tokens=40,
        reserved_tool_tokens=10,
        max_items=16,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 24)


def _reference(
    reference_id: str,
    tokens: int,
    *,
    required: bool = False,
    priority: int = 0,
) -> ContextReference:
    return ContextReference(
        reference_id=reference_id,
        kind="test-evidence",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id=f"sha256:{reference_id}",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        summary=reference_id,
        token_count=tokens,
        metadata={
            "required": required,
            "priority": priority,
            "coverage_ids": (f"coverage:{reference_id}",),
        },
    )


def _compiler() -> ContextCompiler:
    return ContextCompiler(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=270,
    )


def test_declared_primary_and_test_exist() -> None:
    assert COMPILER_PATH.is_file()
    assert TEST_PATH.is_file()
    assert "class ContextCompiler" in COMPILER_PATH.read_text(encoding="utf-8")
    assert "def compile_retry_context(" in COMPILER_PATH.read_text(encoding="utf-8")


def test_optional_evidence_is_selected_by_rank_without_a_second_compiler() -> None:
    compiler = _compiler()
    source = COMPILER_PATH.read_text(encoding="utf-8")
    assert "class CompetingContextSelector" not in source
    assert "class FreshnessSubsystem" not in source
    result = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(
            _reference("low", 200, priority=1),
            _reference("high-b", 40, priority=10),
            _reference("high-a", 40, priority=10),
        ),
    )
    included = {item.reference_id: item for item in result.decisions if item.included}
    omitted = {item.reference_id: item for item in result.decisions if not item.included}
    assert included["high-a"].reason is InclusionReason.RANKED_FIT
    assert included["high-b"].reason is InclusionReason.RANKED_FIT
    assert omitted
    assert set(item.reason for item in omitted.values()) == {ExclusionReason.TOKEN_BUDGET}


def test_stale_tree_identity_is_rejected_and_does_not_complete() -> None:
    compiler = _compiler()
    required = _reference("required", 16, required=True)
    optional = _reference("diagnostic", 18, priority=10)
    parent = compiler.compile(**BINDING, **CORE, evidence=(required, optional)).capsule
    with pytest.raises(ChangedTreeContextError, match="invalidated"):
        compile_retry_context(
            compiler,
            parent,
            prior_decision_id="decision:previous",
            diagnostic_receipt_id="diagnostic:stable",
            evidence=(required, optional),
            failure_evidence_ids=(),
            tree_id="tree:stale",
        )


def test_candidate_artifacts_are_not_completion_authority() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS[:2]:
        assert (ACCELERATE_ROOT / relative).is_file(), relative
    for path in (OUTPUT_PATH, RECEIPT_PATH):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("task_id") == "DOEP-063"
        assert payload.get("completion_authoritative") is not True
        assert payload.get("worker_completion_insufficient") is not False
    if OUTPUT_PATH.is_file():
        manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        assert manifest["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"
        assert manifest["task_cid"] == TASK_CID
        assert manifest["plan_cid"] == PLAN_CID
        assert manifest["no_competing_subsystem_created"] is True
