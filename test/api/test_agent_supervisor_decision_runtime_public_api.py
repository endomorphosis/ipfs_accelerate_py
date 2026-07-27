import json
import subprocess
import sys
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.decision_runtime_benchmark import (
    build_frozen_decision_runtime_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.decision_runtime_rollout import (
    DecisionRuntimeControlRequest,
    DecisionRuntimePublicAPI,
    DecisionRuntimeRolloutBinding,
    DecisionRuntimeRolloutEvaluation,
    DecisionRuntimeRolloutMode,
    DecisionRuntimeRolloutPolicy,
)


def _api():
    qualification = DecisionRuntimeRolloutEvaluation(
        "evaluation:qualification@1",
        "2026-01-01T00:00:00Z",
        build_frozen_decision_runtime_benchmark(observation_label="qualification"),
    )
    current = DecisionRuntimeRolloutEvaluation(
        "evaluation:current@1",
        "2026-01-02T00:00:00Z",
        build_frozen_decision_runtime_benchmark(observation_label="current"),
    )
    binding = DecisionRuntimeRolloutBinding(
        repository_id="repository:proof-runtime-benchmark@1",
        tree_id="sha256:frozen-proof-runtime-tree",
        behavior_id="behavior:proof-directed-decision-runtime",
        objective_id="ASI-G360",
        objective_revision="sha256:frozen-objective",
        policy_id="policy:proof-runtime-rollout@1",
        policy_revision="sha256:frozen-policy",
        capability_id="capability:proof-runtime-local@1",
        capability_revision="sha256:frozen-capability",
    )
    policy = DecisionRuntimeRolloutPolicy(
        policy_id=binding.policy_id,
        policy_revision=binding.policy_revision,
        approved_behavior_ids=(binding.behavior_id,),
        approved_modes=tuple(DecisionRuntimeRolloutMode),
    )
    return DecisionRuntimePublicAPI(
        qualification,
        binding=binding,
        policy=policy,
        current_evaluation=current,
    )


def test_python_cli_and_mcp_use_the_same_canonical_control_contract():
    request = DecisionRuntimeControlRequest(action="automatic")
    results = []
    for adapter in ("python", "cli", "mcp"):
        api = _api()
        results.append(getattr(api, adapter)(request.to_dict()).to_dict())
    assert results[0] == results[1] == results[2]
    assert results[0]["decision"]["effective_mode"] == "automatic"

    api = _api()
    assert api.status().decision.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    api.execute("automatic")
    rolled_back = api.rollback()
    assert rolled_back.decision.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    assert rolled_back.decision.affected_behavior_ids == (
        "behavior:proof-directed-decision-runtime",
    )

    api = _api()
    api.execute("automatic")
    api.binding = replace(api.binding, tree_id="sha256:later-regressed-root")
    regressed = api.status()
    assert regressed.decision.effective_mode is DecisionRuntimeRolloutMode.SHADOW
    assert regressed.decision.rollback_applied


def test_public_discovery_is_lazy_and_provider_free():
    discovery = DecisionRuntimePublicAPI.discovery()
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
    script = """
import json, sys
import ipfs_accelerate_py.agent_supervisor
forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.decision_runtime_rollout import DecisionRuntimePublicAPI
after = {name for name in sys.modules if name.split(".")[0] in forbidden}
print(json.dumps({
    "discovery": DecisionRuntimePublicAPI.discovery(),
    "added": sorted(after - before),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    # The legacy package root has eager optional exports; this new discovery
    # surface must not resolve any additional optional provider.
    assert payload["added"] == []
    assert not payload["discovery"]["optional_providers_loaded"]
