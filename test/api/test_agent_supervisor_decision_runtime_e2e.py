from __future__ import annotations

import ast
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    DecisionKind,
    DecisionStage,
)
from ipfs_accelerate_py.agent_supervisor.context.decision_runtime import (
    DecisionBoundary,
    DecisionOutcome,
    DecisionRuntime,
    DecisionRuntimeBypassError,
    DecisionRuntimeCancelled,
    DecisionRuntimeConfig,
    DecisionRuntimeDenied,
    DecisionRuntimeEffectMismatch,
    DecisionRuntimeInput,
    DecisionRuntimeReceipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES,
    DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES,
    MAX_IMPLEMENTATION_PROPOSAL_MATERIALIZED_BYTES,
    MAX_IMPLEMENTATION_PROPOSAL_SERIALIZED_BYTES,
    PortalImplementationDaemon,
    PortalTask,
)
from test.api.test_agent_supervisor_execution_permit import (
    NOW,
    _fixture,
)


def _runtime_fixture(
    boundary: DecisionBoundary = DecisionBoundary.FILE_MUTATION,
) -> tuple[DecisionRuntime, object]:
    admission, admission_receipt, witness = _fixture()
    request = admission.decision_request
    runtime_input = DecisionRuntimeInput(
        boundary=boundary,
        decision_request=request,
        context_compilation=SimpleNamespace(witness=witness),
        admission_request=admission,
        admission_receipt=admission_receipt,
    )
    config = DecisionRuntimeConfig(
        mode="enforce",
        semantic_roots=request.semantic_roots,
        applicability_facts=request.applicability_facts,
        generic_prompt_policy=("Use the admitted compact context.",),
        allowed_edit_paths=("src/0.py",),
        protected_edit_paths=("docs/operator-owned.todo.md",),
        caller="agent-supervisor:implementation-daemon",
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:policy-v1",
    )
    runtime = DecisionRuntime(
        config,
        resolver=lambda selected, _payload: replace(
            runtime_input, boundary=selected
        ),
        clock_ms=lambda: NOW,
    )
    return runtime, request


def test_config_round_trip_binds_roots_facts_prompt_policy_and_edit_scope() -> None:
    runtime, _request = _runtime_fixture()
    config = runtime.config

    assert DecisionRuntimeConfig.from_json(config.to_json()) == config
    assert config.config_id == DecisionRuntimeConfig.from_dict(
        config.to_dict()
    ).config_id
    assert len(config.semantic_roots) == 8
    assert config.applicability_facts
    assert config.allowed_edit_paths == ("src/0.py",)
    assert config.protected_edit_paths == ("docs/operator-owned.todo.md",)

    changed_policy = replace(
        config,
        generic_prompt_policy=("A different generic policy.",),
    )
    changed_scope = replace(
        config,
        allowed_edit_paths=("src/1.py",),
    )
    assert changed_policy.config_id != config.config_id
    assert changed_scope.config_id != config.config_id


def test_off_and_shadow_are_non_authoritative_and_deterministic() -> None:
    off = DecisionRuntime(DecisionRuntimeConfig(mode="off"))
    shadow = DecisionRuntime(DecisionRuntimeConfig(mode="shadow"))

    off_decision = off.route("tool_invocation", {"operation": "status"})
    first = shadow.route("tool_invocation", {"operation": "status"})
    second = shadow.route("tool_invocation", {"operation": "status"})

    assert off_decision.receipt.outcome is DecisionOutcome.OFF
    assert not off_decision.authoritative
    assert first.receipt.outcome is DecisionOutcome.SHADOW_WOULD_BLOCK
    assert first.receipt.reason_codes == second.receipt.reason_codes
    assert first.receipt.metadata == second.receipt.metadata
    assert not first.authoritative
    assert not first.receipt.completion_authoritative


def test_enforcement_rejects_direct_call_and_transport_bypass() -> None:
    runtime, _request = _runtime_fixture()
    missing_resolver = DecisionRuntime(runtime.config, clock_ms=lambda: NOW)
    calls: list[str] = []

    with pytest.raises(DecisionRuntimeBypassError):
        missing_resolver.route(
            DecisionBoundary.FILE_MUTATION,
            {"path": "src/0.py"},
        )
    with pytest.raises(DecisionRuntimeBypassError):
        runtime.authorize_mutation(
            runtime.route(
                DecisionBoundary.PLAN_SELECTION,
                {"operation": "select"},
            ),
            lambda: calls.append("dispatched"),
        )

    assert calls == []


def test_current_permit_is_checked_at_effect_and_effects_match_exactly() -> None:
    runtime, request = _runtime_fixture()
    decision = runtime.route(
        DecisionBoundary.FILE_MUTATION,
        {"path": "src/0.py"},
    )
    calls: list[str] = []

    execution = runtime.authorize_mutation(
        decision,
        lambda: {
            "value": calls.append("mutated") or "ok",
            "observed_effects": request.expected_effects,
        },
    )

    assert calls == ["mutated"]
    assert execution.value["value"] == "ok"
    assert execution.permit_use is not None
    assert execution.effect_observation is not None
    assert execution.effect_observation.matched
    assert DecisionRuntimeReceipt.from_dict(
        decision.receipt.to_dict()
    ) == decision.receipt

    with pytest.raises(DecisionRuntimeDenied):
        runtime.authorize_mutation(
            decision,
            lambda: calls.append("replayed"),
        )
    assert calls == ["mutated"]


def test_observed_effect_mismatch_is_durable_and_fails_closed() -> None:
    runtime, _request = _runtime_fixture()
    decision = runtime.route("file_mutation", {"path": "src/0.py"})

    with pytest.raises(DecisionRuntimeEffectMismatch) as caught:
        runtime.authorize_mutation(
            decision,
            lambda: {"value": "changed", "observed_effects": ()},
        )

    assert not caught.value.receipt.matched
    assert caught.value.receipt.reason_codes == ("missing_expected_effect",)
    assert runtime.effect_receipts[-1] == caught.value.receipt
    assert runtime.receipts[-1].outcome is DecisionOutcome.EFFECT_MISMATCH


def test_cancellation_prevents_provider_and_mutation_dispatch() -> None:
    cancelled = DecisionRuntime(
        DecisionRuntimeConfig(mode="off"),
        cancellation=lambda: True,
    )
    calls: list[str] = []

    with pytest.raises(DecisionRuntimeCancelled):
        cancelled.route("analysis_request", {"query": "anything"})
    assert calls == []


def test_completion_requires_a_fresh_decision_and_merged_tree_evidence() -> None:
    admission, _receipt, witness = _fixture()
    prior = admission.decision_request
    completion_request = replace(
        prior,
        decision_kind=DecisionKind.COMPLETE,
        stage=DecisionStage.COMPLETION,
    )
    completion_witness = replace(
        witness,
        decision_request_id=completion_request.request_id,
    )
    config = DecisionRuntimeConfig(
        mode="enforce",
        semantic_roots=completion_request.semantic_roots,
        applicability_facts=completion_request.applicability_facts,
        allowed_edit_paths=("src/0.py",),
        caller="agent-supervisor:implementation-daemon",
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:policy-v1",
    )
    good_input = DecisionRuntimeInput(
        boundary=DecisionBoundary.COMPLETION,
        decision_request=completion_request,
        context_compilation=SimpleNamespace(witness=completion_witness),
        completion_evidence={
            "repository_tree_id": completion_request.repository_root.cid_v1,
            "passed": True,
            "completion_authoritative": True,
        },
        prior_decision_request_id=prior.request_id,
    )
    runtime = DecisionRuntime(
        config,
        resolver=lambda _boundary, _payload: good_input,
        clock_ms=lambda: NOW,
    )

    completed = runtime.route(
        DecisionBoundary.COMPLETION,
        {"fresh_merged_tree_required": True},
    )

    assert completed.receipt.outcome is DecisionOutcome.COMPLETION_ADMITTED
    assert completed.receipt.completion_authoritative
    assert not completed.receipt.authoritative
    assert completed.permit is None

    stale = replace(
        good_input,
        completion_evidence={
            "repository_tree_id": "tree:stale",
            "passed": True,
            "completion_authoritative": True,
        },
    )
    with pytest.raises(DecisionRuntimeDenied):
        runtime.route(DecisionBoundary.COMPLETION, decision_input=stale)


def test_runtime_module_has_no_optional_provider_imports() -> None:
    source = Path(
        "ipfs_accelerate_py/agent_supervisor/decision_runtime.py"
    ).read_text(encoding="utf-8")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not {
        "transformers",
        "torch",
        "openai",
        "ipfs_datasets_py",
        "neo4j",
        "duckdb",
    }.intersection(imported)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_daemon_accepts_bounded_local_sources_when_raw_patch_is_small(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    baseline = ("# locally observed source\n" * 31_000) + "VALUE = 1\n"
    paths = (repo / "runtime_a.py", repo / "runtime_b.py")
    for path in paths:
        path.write_text(baseline, encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "supervisor@example.invalid")
    _git(repo, "config", "user.name", "Supervisor Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline_ref = _git(repo, "rev-parse", "HEAD")
    for index, path in enumerate(paths, start=2):
        path.write_text(
            baseline.replace("VALUE = 1", f"VALUE = {index}"),
            encoding="utf-8",
        )

    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="ASI-141",
        title="Repair the retry-budget validation blocker",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[path.name for path in paths],
        validation=["python -m pytest"],
        acceptance="Keep raw patches bounded while validating local sources.",
    )

    result = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline_ref,
    )

    assert result.accepted
    assert (
        result.policy.max_patch_bytes
        > DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES
    )
    assert (
        result.policy.max_output_bytes
        > DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES
    )
    assert (
        result.policy.max_patch_bytes
        <= MAX_IMPLEMENTATION_PROPOSAL_MATERIALIZED_BYTES
    )
    assert (
        result.policy.max_output_bytes
        <= MAX_IMPLEMENTATION_PROPOSAL_SERIALIZED_BYTES
    )


def test_daemon_does_not_expand_limits_for_an_oversized_raw_patch() -> None:
    proposal = SimpleNamespace(
        candidate_diff=(),
        patch_text="x" * (DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES + 1),
        to_dict=lambda: {},
    )

    assert PortalImplementationDaemon._proposal_local_envelope_limits(
        proposal
    ) == {
        "max_patch_bytes": DEFAULT_IMPLEMENTATION_PROPOSAL_PATCH_BYTES,
        "max_output_bytes": DEFAULT_IMPLEMENTATION_PROPOSAL_OUTPUT_BYTES,
    }
