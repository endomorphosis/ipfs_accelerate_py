"""Actual native capsules and final-input audits, with only mocked providers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled  # noqa: F401


@pytest.fixture
def dispatch_case(native, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES,
    )
    repository, artifact, prompt, source = native
    for argv in (["init", "-q"], ["add", "mod.py"], ["-c", "user.name=Metadata fixture",
                 "-c", "user.email=metadata@example.invalid", "commit", "-qm", "authored fixture"]):
        subprocess.run(["git", "-C", str(repository), *argv], check=True, capture_output=True)
    workspace = repository.parent / "allocated-worktree"
    subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", str(workspace)],
        check=True, capture_output=True)
    monkeypatch.chdir(workspace)
    for name in (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
                 "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="authored-model", catalog_revision="authored"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **kw: object())
    calls = []
    def generate(model_prompt, **kwargs):
        calls.append((model_prompt, kwargs))
        schema = kwargs.get("codex_output_schema")
        observation = {"exit_code": 0, "usage": {"input_tokens": 5, "output_tokens": 2}}
        if schema is not None:
            raw = json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
            observation.update(codex_output_schema_sha256=hashlib.sha256(raw).hexdigest(),
                               codex_output_schema_bytes=len(raw))
        set_last_cli_observation("codex_cli", observation)
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    monkeypatch.setattr(llm_router, "generate_text", generate)
    return {"repository": repository, "workspace": workspace, "prompt": prompt, "calls": calls}


def invoke(case, **kwargs):
    return runner.run(prompt=case["prompt"], provider="codex_cli", model="authored-model",
        timeout=30, max_output_tokens=4096, semantic_repository=case["repository"],
        coding_reply_mode="ordinary-completion@1", **kwargs)


def test_opt_in_preserves_default_input_and_binds_actual_selected_reply(dispatch_case):
    case = dispatch_case
    _, legacy = invoke(case)
    legacy_input = case["calls"][-1][0]
    _, selected = invoke(case, semantic_metadata_view="common-bindings@1")
    selected_input = case["calls"][-1][0]
    assert "semantic_metadata_view" not in legacy
    assert "semantic_metadata_native_coding_reply_contract" not in legacy
    assert selected["semantic_metadata_native_coding_reply_contract"]["model_prompt_after_sha256"] == hashlib.sha256(legacy_input.encode()).hexdigest()
    assert selected["coding_reply_contract"]["model_prompt_after_sha256"] == hashlib.sha256(selected_input.encode()).hexdigest()
    assert len(selected_input.encode()) <= len(legacy_input.encode())
    assert selected["coding_reply_contract"]["instruction_sha256"] == legacy["coding_reply_contract"]["instruction_sha256"]
    assert case["calls"][0][1]["codex_output_schema"] == case["calls"][1][1]["codex_output_schema"]
    assert len(case["calls"]) == 2  # Mocked coding calls only; no provider network.


def test_independent_historical_reconstruction_checks_both_reply_layers(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    receipt["phase"] = "coding"
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, header, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
                                                   repository=case["repository"])
    assert model == case["calls"][-1][0] and all(checks.values())
    assert checks["semantic_metadata_selection_receipt"]
    assert checks["semantic_metadata_native_reply_contract"]
    tampered = deepcopy(projected)
    tampered["semantic_metadata_native_coding_reply_contract"]["model_prompt_after_sha256"] = "0" * 64
    _, _, bad_checks = audit._model_projection(rendered=case["prompt"], receipt=tampered,
                                              repository=case["repository"])
    assert not bad_checks["semantic_metadata_native_reply_contract"]


@pytest.mark.parametrize("mode,schema,purpose", [("foreign", "supervisor-semantic-router-input@1", "coding"),
    ("common-bindings@1", "supervisor-semantic-router-input@2", "coding"),
    ("common-bindings@1", "supervisor-semantic-router-input@1", "planning")])
def test_unsupported_selection_refuses_before_provider(dispatch_case, mode, schema, purpose):
    case = dispatch_case
    with pytest.raises(ValueError):
        invoke(case, semantic_metadata_view=mode, semantic_transport_schema=schema, purpose=purpose)
    assert case["calls"] == []


def test_implementation_argv_selects_only_explicit_semantic_model_route():
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    base = dict(router=Path("/router"), model="authored-model", reasoning="high", timeout=30,
                semantic_repository=Path("/repository"), doctor=None)
    legacy = implementation_argv(**base)
    assert "--semantic-metadata-view" not in legacy
    chosen = implementation_argv(**base, semantic_metadata_view="common-bindings@1")
    assert chosen[-2:] == ["--semantic-metadata-view", "common-bindings@1"]
    with pytest.raises(ValueError):
        implementation_argv(**{**base, "semantic_repository": None}, semantic_metadata_view="common-bindings@1")


@pytest.mark.parametrize("layer", ["metadata", "baseline_reply"])
def test_boolean_integer_receipt_tampering_is_rejected(dispatch_case, layer):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    receipt["phase"] = "coding"
    altered = deepcopy(receipt)
    if layer == "metadata":
        altered["semantic_metadata_view"]["proof_authority"] = 0
        with pytest.raises(ValueError):
            audit._receipt(altered, case["workspace"].parent)
    else:
        altered["semantic_metadata_native_coding_reply_contract"]["completion_authority"] = 0
        parsed = audit._receipt(altered, case["workspace"].parent)
        _, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
                                               repository=case["repository"])
        assert not checks["semantic_metadata_native_reply_contract"]


def test_actual_audit_subprocess_retains_metadata_and_both_reply_layers(dispatch_case, compiled):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import (
        PROFILE, instruction_sha256, task_profile_bytes,
    )
    case = dispatch_case
    root = case["repository"]
    artifact = root / ".runtime/semantic/worker-context.json"
    semantic = json.loads(artifact.read_text())
    prepared_path = compiled["state"] / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    expected_path = compiled["state"] / "context-result.json"
    expected = json.loads(expected_path.read_text())
    source_hash = hashlib.sha256((root / "mod.py").read_bytes()).hexdigest()
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": instruction_sha256(prepared["query"]), "input_paths": ["mod.py"],
        "outputs": [{"path": "mod.py", "effect": "modify", "media_type": "text/x-python"}]}
    prepared.update(repository=str(root), spec={"task_key": "TASK-1"}, worker_inputs=["mod.py"], task_profile=profile)
    prepared["manifest"]["payload"]["sources"] = {"mod.py": {"sha256": source_hash},
        PROFILE: {"sha256": hashlib.sha256(task_profile_bytes(profile)).hexdigest()}}
    expected["semantic_root_cid"] = semantic["semantic_root_cid"]
    retrieval = {"task_id": "TASK-1", "index_id": expected["index_id"], "result_id": "cid:result",
        "query_id": "cid:query", "query_text": prepared["query"], "source_sha256": {"mod.py": source_hash},
        "status": "current", "nomination_only": True, "semantic_authority": False,
        "execution_authority": False, "completion_authority": False}
    world = {"semantic_root_cid": semantic["semantic_root_cid"], "world_snapshot_cid": expected["world_snapshot_cid"],
        "plan_projection_cid": "cid:projection", "intent_freshness_checked": False,
        "tasks": [{"task_cid": expected["task_cid"], "body": {"local_planning_contract_cid": "cid:contract"}}],
        "execution_authority": False, "completion_authority": False}
    refs = []
    for kind, payload in [("semantic-context", semantic), ("code-retrieval-context", retrieval),
                          ("intent-world-context", world)]:
        text = artifact.read_text() if kind == "semantic-context" else json.dumps(payload, ensure_ascii=False)
        refs.extend(build_text_context_references(text,
            reference_prefix=kind, kind=kind, path=artifact.relative_to(root).as_posix() if kind == "semantic-context" else "",
            repository_id="repo:test", tree_id="tree:test", required=True, chunk_bytes=1201))
    result = ContextCompiler(ContextBudget(max_input_tokens=65536, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(repository_id="repo:test",
        tree_id="tree:test", objective_id="TASK-1", objective_revision="sha256:task", policy_id="policy:test",
        policy_revision="sha256:policy", caller="supervisor:test", stage="implementation", goal={"id": "TASK-1"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["mod.py"]}, acceptance={"criteria": ["pending native validation"]}, evidence=refs)
    case["prompt"] = render_context_capsule(result.capsule)
    compiled["capsule"].write_text(result.capsule.to_json())
    prepared_path.write_text(json.dumps(prepared))
    expected_path.write_text(json.dumps(expected))
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    receipt["phase"] = "coding"
    observed = audit.collect_terminal_context_audit(state=compiled["state"], receipts=[receipt],
        workspace_root=case["workspace"].parent, timeout_seconds=10)
    assert observed["status"] == "verified_all_observed_model_inputs", observed
    assert observed["all_observed_coding_inputs_verified"] is True
    assert observed["matches"][0]["model_input_checks"]["semantic_metadata_selection_receipt"]
    assert observed["matches"][0]["model_input_checks"]["semantic_metadata_native_reply_contract"]
    assert observed["provider_calls"] == 0 and observed["raw_prompts_exported"] is False
