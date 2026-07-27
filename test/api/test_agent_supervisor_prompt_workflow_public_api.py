"""Public Python/CLI/MCP control surface for prompt workflow rollout."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    build_frozen_prompt_workflow_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_rollout import (
    PROMPT_WORKFLOW_BEHAVIOR_ID,
    PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID,
    PromptWorkflowControlRequest,
    PromptWorkflowPublicAPI,
    PromptWorkflowRolloutEvaluation,
    PromptWorkflowRolloutMode,
    build_default_prompt_workflow_binding,
    build_default_prompt_workflow_policy,
)


def _api():
    qualification = PromptWorkflowRolloutEvaluation(
        "evaluation:qualification@1",
        "2026-01-01T00:00:00Z",
        build_frozen_prompt_workflow_benchmark(observation_label="qualification"),
    )
    current = PromptWorkflowRolloutEvaluation(
        "evaluation:current@1",
        "2026-01-02T00:00:00Z",
        build_frozen_prompt_workflow_benchmark(observation_label="current"),
    )
    binding = build_default_prompt_workflow_binding()
    policy = build_default_prompt_workflow_policy(approve_automatic=True)
    return PromptWorkflowPublicAPI(
        qualification,
        binding=binding,
        policy=policy,
        current_evaluation=current,
    )


def test_python_cli_and_mcp_use_the_same_canonical_control_contract():
    request = PromptWorkflowControlRequest(action="automatic")
    results = []
    for adapter in ("python", "cli", "mcp"):
        api = _api()
        results.append(getattr(api, adapter)(request.to_dict()).to_dict())
    assert results[0] == results[1] == results[2]
    assert results[0]["decision"]["effective_mode"] == "automatic"
    assert results[0]["decision"]["requirement_id"] == (
        PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID
    )

    api = _api()
    assert api.status().decision.effective_mode is PromptWorkflowRolloutMode.SHADOW
    api.execute("automatic")
    rolled_back = api.rollback()
    assert rolled_back.decision.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert rolled_back.decision.affected_behavior_ids == (
        PROMPT_WORKFLOW_BEHAVIOR_ID,
    )

    api = _api()
    api.execute("automatic")
    api.binding = replace(api.binding, tree_id="sha256:later-regressed-root")
    regressed = api.status()
    assert regressed.decision.effective_mode is PromptWorkflowRolloutMode.SHADOW
    assert regressed.decision.rollback_applied


def test_public_discovery_is_lazy_and_provider_free():
    discovery = PromptWorkflowPublicAPI.discovery()
    assert set(discovery["surfaces"]) == {"python", "cli", "mcp"}
    assert set(discovery["actions"]) == {
        "off",
        "shadow",
        "assist",
        "automatic",
        "status",
        "explanation",
        "rollback",
    }
    assert discovery["behavior_id"] == PROMPT_WORKFLOW_BEHAVIOR_ID
    assert discovery["requirement_id"] == PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID
    assert discovery["optional_providers_loaded"] is False
    assert discovery["processes_started"] is False

    script = """
import json, sys
import ipfs_accelerate_py.agent_supervisor
forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_rollout import (
    PromptWorkflowPublicAPI,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    build_frozen_prompt_workflow_benchmark,
    recompute_prompt_workflow_gate,
)
after = {name for name in sys.modules if name.split(".")[0] in forbidden}
report = recompute_prompt_workflow_gate(build_frozen_prompt_workflow_benchmark())
print(json.dumps({
    "discovery": PromptWorkflowPublicAPI.discovery(),
    "added": sorted(after - before),
    "passed": report.passed,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert not payload["discovery"]["optional_providers_loaded"]
    assert payload["passed"]


def test_exact_python_cli_mcp_examples_round_trip():
    """Documented control examples stay schema-stable across surfaces."""

    request = PromptWorkflowControlRequest(action="assist")
    api = _api()
    # Python (canonical request object / mapping)
    python_result = api.python(request.to_dict())
    # CLI string vocabulary
    cli_result = api.cli("assist")
    # MCP mapping
    mcp_result = api.mcp(request.to_dict())
    assert (
        python_result.decision.effective_mode
        == cli_result.decision.effective_mode
        == mcp_result.decision.effective_mode
        == PromptWorkflowRolloutMode.ASSIST
    )
    explanation = api.explanation()
    assert PROMPT_WORKFLOW_BEHAVIOR_ID in explanation.explanation
    assert "assist" in explanation.explanation
