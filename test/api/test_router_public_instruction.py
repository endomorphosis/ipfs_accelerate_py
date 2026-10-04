"""Signed public requirements reach real router composition without a provider call on refusal."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def selected(scenario, tmp_path):
    root = scenario["repository"]
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    # This authored fixture's exact read-only public requirement is executable
    # text. The loader neither parses it nor guesses an instruction filename.
    source_path = "test_answer.py"
    source = (root / source_path).read_bytes()
    context = instruction.prepare_public_instruction_context(repository=root, admission=admission,
        task_cid=scenario["task_cid"], source_path=source_path, expected_source_sha256=sha(source))
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    request = dict(artifact=Path(context["artifact"]), expected_sha256=context["sha256"],
        task_cid=scenario["task_cid"], prompt=json.dumps({"objective_id": "LOCAL-TASK"}), workspace=workspace)
    return root, admission, source, context, request


def _invoke(request, **extra):
    return runner.run(prompt=request["prompt"], provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, public_instruction_artifact=request["artifact"],
        public_instruction_sha256=request["expected_sha256"], public_instruction_task_cid=request["task_cid"], **extra)


@pytest.mark.parametrize("semantic", [False, True])
def test_actual_router_keeps_verbatim_requirements_and_auditable_native_hash(selected, provider, monkeypatch, semantic):
    root, _, source, context, request = selected
    refs = ()
    if semantic:
        output = root / ".runtime/semantic"
        prepare_semantic_context(repository=root, paths=["answer.py"], required_raw_paths=["answer.py"],
            objective="Repair the answer", task_id="LOCAL-TASK", output=output)
        artifact = output / "worker-context.json"
        refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
            kind="semantic-context", path=artifact.relative_to(root).as_posix(), repository_id="repo:test",
            tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="LOCAL-TASK",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "LOCAL-TASK"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["answer.py"]}, acceptance={"criteria": ["public validation pending"]}, evidence=refs)
    request["prompt"] = render_context_capsule(compiled.capsule)
    assert source.decode() not in request["prompt"]
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(root.parent / "empty-codex-home"))
    text, receipt = _invoke(request, semantic_repository=root if semantic else None)
    _, observed = provider
    assert text == "literal summary" and len(observed) == 1
    actual = observed[0][0]
    assert source.decode() in actual
    assert "grants no additional write scope" in actual
    assert receipt["native_prompt_sha256"] == receipt["prompt_sha256"] == sha(request["prompt"].encode())
    assert receipt["model_prompt_sha256"] == sha(actual.encode())
    assert receipt["public_instruction"]["source_sha256"] == sha(source)
    assert receipt["public_instruction"]["semantic_minification_applied"] is False
    assert receipt["public_instruction"]["manifest_signature_verified"] is True
    assert receipt["public_instruction"]["extra_provider_calls"] == 0
    parsed = audit._receipt({**receipt, "phase": "coding"}, request["workspace"].parent)
    # The completed source can change and the allocated checkout can disappear.
    (root / "answer.py").write_text("def answer():\n    return 2\n")
    subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(request["workspace"])], check=True)
    monkeypatch.chdir(root.parent)
    replayed, _, checks = audit._model_projection(rendered=request["prompt"], receipt=parsed, repository=root)
    assert replayed == actual and all(checks.values())
    assert context["completion_authority"] is False
    assert context["scope_expansion_authority"] is False


@pytest.mark.parametrize("change", ["source", "worker_source", "other_source", "task", "prompt",
    "artifact", "writable", "symlink", "hardlink", "foreign", "canonical", "partial"])
def test_router_refuses_drift_or_bad_binding_before_provider(selected, provider, monkeypatch, tmp_path, change):
    root, _, _, _, original = selected
    request = dict(original)
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(root.parent / "empty-codex-home"))
    if change == "source":
        (root / "test_answer.py").write_text("different requirements")
    elif change == "worker_source":
        (request["workspace"] / "test_answer.py").write_text("different requirements")
    elif change == "other_source":
        (request["workspace"] / "answer.py").write_text("different implementation baseline")
    elif change == "task":
        request["task_cid"] = "foreign-task"
    elif change == "prompt":
        request["prompt"] = json.dumps({"objective_id": "FOREIGN-TASK"})
    elif change in {"artifact", "writable"}:
        request["artifact"].chmod(0o644)
        if change == "artifact":
            request["artifact"].write_bytes(request["artifact"].read_bytes() + b" ")
            request["artifact"].chmod(0o444)
    elif change == "symlink":
        target = request["artifact"].with_name("linked.json")
        target.symlink_to(request["artifact"])
        request["artifact"] = target
    elif change == "hardlink":
        request["artifact"].with_name("linked.json").hardlink_to(request["artifact"])
    elif change == "foreign":
        foreign = tmp_path / "foreign"
        subprocess.run(["git", "clone", "-q", str(root), str(foreign)], check=True)
        monkeypatch.chdir(foreign)
    elif change == "canonical":
        monkeypatch.chdir(root)
    else:
        request["expected_sha256"] = ""
    with pytest.raises((ValueError, OSError)):
        _invoke(request)
    assert provider[1] == []


@pytest.mark.parametrize("mutation", ["signature", "scope", "authority", "wrong_selected_hash"])
def test_repinned_artifact_cannot_escape_signed_read_only_source(selected, mutation):
    _, _, _, _, request = selected
    payload = json.loads(request["artifact"].read_text())
    if mutation == "signature":
        payload["manifest"]["payload"]["sources"]["test_answer.py"]["sha256"] = "a" * 64
        payload["manifest_cid"] = content_identity(payload["manifest"])
    elif mutation == "scope":
        payload["source_path"] = "answer.py"
        payload["source_sha256"] = payload["manifest"]["payload"]["sources"]["answer.py"]["sha256"]
    elif mutation == "authority":
        payload["completion_authority"] = True
    else:
        payload["source_sha256"] = "a" * 64
    payload["context_cid"] = content_identity({k: v for k, v in payload.items() if k != "context_cid"})
    raw = json.dumps(payload).encode()
    request["artifact"].chmod(0o644)
    request["artifact"].write_bytes(raw)
    request["artifact"].chmod(0o444)
    request["expected_sha256"] = sha(raw)
    with pytest.raises(ValueError):
        instruction.load_public_instruction(**request)


def test_owner_requires_explicit_exact_read_only_source_and_digest(scenario):
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    args = dict(repository=scenario["repository"], admission=admission, task_cid=scenario["task_cid"])
    with pytest.raises(ValueError, match="selected source bytes"):
        instruction.prepare_public_instruction_context(**args, source_path="test_answer.py", expected_source_sha256="a" * 64)
    with pytest.raises(ValueError, match="read-only"):
        instruction.prepare_public_instruction_context(**args, source_path="answer.py",
            expected_source_sha256=sha((scenario["repository"] / "answer.py").read_bytes()))
    assert not (scenario["repository"] / instruction.DIRECTORY).exists()


def test_historical_instruction_replay_is_not_fresh_dispatch(selected):
    root, _, source, _, request = selected
    before, live = instruction.load_public_instruction(**request)
    (root / "test_answer.py").write_text("subsequently changed requirements")
    historical, receipt = instruction.load_public_instruction(**{**request, "workspace": None,
        "require_current_source": False})
    assert historical == before and source.decode() in historical
    assert receipt["historical_replay"] is True and receipt["source_freshness_verified"] is False
    assert live["source_freshness_verified"] is True
    with pytest.raises(ValueError, match="stale"):
        instruction.load_public_instruction(**request)


def test_instruction_absence_is_backward_compatible_and_not_an_ambient_file_lookup(selected, provider, monkeypatch):
    root, _, source, _, request = selected
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(root.parent / "empty-codex-home"))
    _, receipt = runner.run(prompt=request["prompt"], provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128)
    assert source.decode() not in provider[1][0][0]
    assert receipt["public_instruction"] is None
    assert receipt["router_prompt_sha256"] == receipt["native_prompt_sha256"]
