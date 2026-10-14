"""Native Doctor/router receipts distinguish declared budgets from hard caps.

The provider is replaced; native Doctor, capsule, source checks, transport and
historical replay run normally. Synthetic usage is deliberately above the
declared proposal-token budget and does not become permission to continue.
"""
import copy
import hashlib
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_residual_context import retained  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit


def _prompt(request, scenario):
    root = request["repository"]
    task_id = scenario["intent"].get_task(request["task_cid"])["task_alias"]
    output = root / ".runtime/semantic"
    prepare_semantic_context(repository=root, paths=["answer.py"], required_raw_paths=["answer.py"],
        objective="Repair the admitted answer", task_id=task_id, output=output)
    artifact = output / "worker-context.json"
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(), repository_id="repo:test",
        tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id=task_id,
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": task_id},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["answer.py"]},
        acceptance={"criteria": ["pending native validation"]}, evidence=refs)
    return render_context_capsule(compiled.capsule)


def _invoke(retained, provider, monkeypatch, scenario, *, schema=2):
    _, _, context, original = retained
    request = dict(original)
    if schema == 1:
        # Reproduce a retained historical artifact without inventing a budget
        # observation that did not exist in its signed input.
        payload = json.loads(request["artifact"].read_text())
        payload["schema"] = "supervisor-doctor-residual-context@1"
        payload.pop("budget_observation")
        payload.pop("context_cid")
        payload["context_cid"] = content_identity(payload)
        raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
        request["artifact"].chmod(0o644)
        request["artifact"].write_bytes(raw)
        request["artifact"].chmod(0o444)
        request["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    prompt = _prompt(request, scenario)
    state, observed = provider
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    state["after_generate"] = lambda: set_last_cli_observation("codex_cli", {
        "exit_code": 0, "prompt_tokens": 12000, "completion_tokens": 8192, "cached_tokens": 1000})
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(request["repository"].parent / "empty-worker-home"))
    text, receipt = runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, semantic_repository=request["repository"],
        doctor_residual_artifact=request["artifact"],
        doctor_residual_sha256=request["expected_sha256"], doctor_residual_task_cid=request["task_cid"])
    assert text == "literal summary" and len(observed) == 1
    return request, prompt, receipt, observed[0]


@pytest.mark.parametrize("schema", [1, 2])
def test_actual_router_reports_limits_without_claiming_enforcement_and_replays_history(
        retained, provider, monkeypatch, scenario, schema):
    request, prompt, receipt, (model_input, options) = _invoke(
        retained, provider, monkeypatch, scenario, schema=schema)
    assert options["max_tokens"] == options["max_new_tokens"] == 128
    assert receipt["provider_output_token_cap_enforced"] is False
    assert receipt["usage"]["completion_tokens"] == 8192
    if schema == 2:
        residual = receipt["doctor_residual_context"]
        assert residual["residual_budget"]["max_tokens"] == 4096
        capability = receipt["doctor_budget_capability"]
        assert capability["requested_output_tokens"] == 128
        assert capability["provider"] == "codex_cli"
        assert capability["native_output_token_cap_enforced"] is False
        assert capability["native_total_token_cap_enforced"] is False
        assert capability["provider_cost_cap_enforced"] is False
        assert capability["provider_usage_converted_to_residual_budget"] is False
        assert capability["additional_router_rounds_admitted"] is False
        assert "doctor-native-coding-budget-capability" not in model_input
        assert "coding_route_enforcement" not in model_input
    else:
        assert receipt["doctor_residual_context"]["schema"] == "doctor-residual-router-context@1"
        assert "doctor_budget_capability" not in receipt
        assert "budget_observation" not in receipt["doctor_residual_context"]
    parsed = audit._receipt({**receipt, "phase": "coding"}, request["workspace"].parent)
    actual, _, checks = audit._model_projection(rendered=prompt, receipt=parsed,
        repository=request["repository"])
    assert actual == model_input and all(checks.values())
    # Legitimate edits after dispatch do not invalidate historical input replay.
    (request["repository"] / "answer.py").write_text("# later published source\n")
    actual, _, checks = audit._model_projection(rendered=prompt, receipt=parsed,
        repository=request["repository"])
    assert actual == model_input and all(checks.values())


def test_audit_refuses_forged_enforcement_and_crossjoins_budget_to_artifact(
        retained, provider, monkeypatch, scenario):
    request, prompt, receipt, _ = _invoke(retained, provider, monkeypatch, scenario)
    for key in ("native_output_token_cap_enforced", "native_total_token_cap_enforced",
                "provider_cost_cap_enforced", "provider_usage_converted_to_residual_budget",
                "additional_router_rounds_admitted", "completion_authority"):
        for value in (True, 0):
            forged = copy.deepcopy(receipt)
            forged["doctor_budget_capability"][key] = value
            with pytest.raises(ValueError, match="native token caps"):
                audit._receipt({**forged, "phase": "coding"}, request["workspace"].parent)
    for key in ("doctor_budget_capability", "requested_output_tokens", "provider"):
        forged = copy.deepcopy(receipt)
        del forged[key]
        with pytest.raises(ValueError):
            audit._receipt({**forged, "phase": "coding"}, request["workspace"].parent)
    forged = copy.deepcopy(receipt)
    forged["doctor_residual_context"]["residual_budget"]["max_tokens"] = 8192
    forged["doctor_residual_context"]["budget_observation"]["limits"]["max_tokens"] = 8192
    forged["doctor_budget_capability"] = runner.doctor_budget_capability(provider="codex_cli",
        requested_output_tokens=128,
        budget_observation=forged["doctor_residual_context"]["budget_observation"])
    parsed = audit._receipt({**forged, "phase": "coding"}, request["workspace"].parent)
    _, _, checks = audit._model_projection(rendered=prompt, receipt=parsed,
        repository=request["repository"])
    assert checks["doctor_residual_residual_budget"] is False
    assert checks["doctor_residual_budget_observation"] is False
    assert checks["doctor_budget_capability"] is False


@pytest.mark.parametrize("provider_name", ["codex_cli", "grok_cli"])
def test_both_native_cli_routes_explicitly_lack_hard_token_caps(provider_name):
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_budget import residual_budget_observation
    limits = dict(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1)
    observation = residual_budget_observation(dict(residual_budget=limits,
        token_budget={"residual_" + key: value for key, value in limits.items()}))
    capability = runner.doctor_budget_capability(provider=provider_name,
        requested_output_tokens=4096, budget_observation=observation)
    assert capability["native_output_token_cap_enforced"] is False
    assert capability["native_total_token_cap_enforced"] is False
    assert capability["additional_router_rounds_admitted"] is False


def test_candidate_receipt_cannot_inject_doctor_budget_capability(tmp_path):
    receipt = {key: "a" * 64 for key in audit.RECEIPT_FIELDS}
    receipt.update(schema="router-implementation-invocation@1", purpose="coding", phase="coding",
        workspace=str(tmp_path / "allocated"), candidate_round_dispatch={}, doctor_budget_capability={})
    for key in ("prompt_bytes", "native_prompt_bytes", "model_prompt_bytes", "workspace_advisory_bytes"):
        receipt[key] = 1
    with pytest.raises(ValueError, match="candidate Doctor budget capability is not admitted"):
        audit._receipt(receipt, tmp_path)
