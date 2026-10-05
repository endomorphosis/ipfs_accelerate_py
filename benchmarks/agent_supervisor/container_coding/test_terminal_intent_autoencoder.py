"""Actual optional Intent inference before an authored, admitted planning fixture."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import shlex
import sys
import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent, intent_asset_arguments
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import build_prompt_goal_provider_request
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest, DirectoryScanReceipt
from ipfs_datasets_py.logic.intent_ir.formalize import preplanning as native


@pytest.fixture(scope="module")
def intent_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("intent-model")
    train = [native.prepare_instruction_feature_targets(text) for text in (
        "Inspect the parser module and repair its checks.", "Add a regression test for formatting.")]
    tuning = [native.prepare_instruction_feature_targets("Validate task outputs before completion.")]
    return native.train_intent_feature_checkpoint(train, tuning, root / "model", epochs=3)


def _descriptor(tmp_path, checkpoint):
    path = tmp_path / "intent-descriptor.json"
    path.write_text(json.dumps(checkpoint))
    return path


def _plan(state, prepared, monkeypatch):
    original_run = subprocess.run
    def version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.160.0\n", "")
        return original_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", version)
    captured = []
    def provider(prompt, **kwargs):
        captured.append(prompt)
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    result = prep.plan(state=state, provider_callable=provider)
    assert result["qualified"] is True, result
    assert result["provider_calls"] == 1 and len(captured) == 1
    assert result["intent_preplanning"]["execution_authority"] is False
    return result, captured[0]


def _original_prompt(prepared):
    return build_prompt_goal_provider_request(PromptWorkflowRequest.from_dict(prepared["request"]),
        DirectoryScanReceipt.from_dict(prepared["scan"]), config=prep._config(Path(prepared["repository"])),
        constraint_summaries=prepared["constraints"])


@pytest.mark.parametrize("disabled", [False, True])
def test_unconfigured_and_disabled_preserve_exact_provider_request(original, monkeypatch, disabled):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        disable_intent_autoencoder=disabled)
    expected = "disabled" if disabled else "fail_open_no_checkpoint"
    assert prepared["intent_preplanning"]["status"] == expected
    result, prompt = _plan(state, prepared, monkeypatch)
    assert prompt == _original_prompt(prepared)
    assert result["intent_preplanning"]["status"] == expected
    assert result["intent_preplanning"]["supplied_to_router"] is False
    assert (root / prep.INSTRUCTION).read_text() == instruction.read_text() == prepared["query"]


def test_stage_runs_before_domain_task_and_goal_declarations(original, monkeypatch):
    events = []
    native_prepare = advisor.prepare_intent_advice
    domain_declarations = prep.local.local_planning_domain_declarations
    def recorded(**kwargs):
        events.append("intent")
        return native_prepare(**kwargs)
    def domains(**kwargs):
        assert events == ["intent"]
        events.append("domains")
        return domain_declarations(**kwargs)
    monkeypatch.setattr(advisor, "prepare_intent_advice", recorded)
    monkeypatch.setattr(prep.local, "local_planning_domain_declarations", domains)
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2])
    assert events == ["intent", "domains"]
    assert prepared["intent_preplanning"]["before_goal_decomposition"] is True
    assert prepared["request"]["intent_ir_root"] == prep.local.content_identity(
        prepared["manifest"]["payload"]["planning_inputs"]["domain_declarations"]["intent"])


@pytest.mark.parametrize("kind", ["missing", "malformed", "oversized", "invalid_checkpoint", "frontend_error"])
def test_optional_failures_keep_native_admission_and_raw_prompt(original, tmp_path, monkeypatch, kind):
    path = tmp_path / "selected-intent.json"
    if kind == "malformed": path.write_text("{")
    if kind == "oversized": path.write_text(" " * (advisor.MAX_DESCRIPTOR_BYTES + 1))
    if kind == "invalid_checkpoint": path.write_text(json.dumps({"schema": "wrong"}))
    if kind == "frontend_error":
        def unavailable(*args, **kwargs): raise RuntimeError("optional inference unavailable")
        monkeypatch.setattr(native, "prepare_intent_instruction", unavailable)
        path = None
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2],
        intent_checkpoint_descriptor=path)
    assert prepared["intent_preplanning"]["status"].startswith("fail_open_")
    result, prompt = _plan(original[2], prepared, monkeypatch)
    assert prompt == _original_prompt(prepared)
    assert result["intent_preplanning"]["supplied_to_router"] is False


def test_real_shared_weights_execute_before_planning_and_reach_advisory_context(original, tmp_path, monkeypatch, intent_checkpoint):
    descriptor_path = _descriptor(tmp_path, intent_checkpoint)
    before = Path(intent_checkpoint["path"]).read_bytes()
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2],
        intent_checkpoint_descriptor=descriptor_path)
    assert prepared["intent_preplanning"]["status"] == "feature_advice"
    advice = json.loads((original[2] / "intent-advice.json").read_text())
    report = advice["report"]
    assert report["learned"]["inference"]["rows"][0]["latent"]
    assert report["learned"]["checkpoint_sha256"] == intent_checkpoint["sha256"]
    assert report["learned"]["decoded_formulas_generated"] is False
    assert report["provider_calls"] == report["training_steps"] == report["download_calls"] == 0
    summary, selected = advisor.intent_planner_summary(advice, instruction=prepared["query"])
    assert summary and len(summary.encode()) <= 8192 and selected == advice
    assert prepared["query"] not in summary and "latent" not in summary
    assert "reconstructed_projection_features" not in summary
    result, prompt = _plan(original[2], prepared, monkeypatch)
    assert result["intent_preplanning"]["supplied_to_router"] is True
    assert result["intent_preplanning"]["status"] == "feature_advice"
    assert summary in json.loads(prompt)["constraints"]["constraint_summaries"]
    assert Path(intent_checkpoint["path"]).read_bytes() == before
    assert prepared["query"] == original[1].read_text()


@pytest.mark.parametrize("kind", ["sidecar", "authority", "source", "checkpoint", "numerics"])
def test_changed_optional_advice_is_omitted_without_bypassing_admission(original, tmp_path, monkeypatch, intent_checkpoint, kind):
    copied = tmp_path / "candidate.json"
    copied.write_bytes(Path(intent_checkpoint["path"]).read_bytes())
    descriptor = {**intent_checkpoint, "path": str(copied)}
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2],
        intent_checkpoint_descriptor=_descriptor(tmp_path, descriptor))
    if kind == "checkpoint":
        copied.write_bytes(copied.read_bytes() + b" ")
    else:
        path = original[2] / "intent-advice.json"
        advice = json.loads(path.read_text())
        if kind == "numerics":
            advice["report"]["learned"]["inference"]["rows"][0]["latent"][0] += 1
            report = advice["report"]
            report["report_sha256"] = hashlib.sha256(native._raw({key: value for key, value in report.items()
                if key != "report_sha256"})).hexdigest()
        elif kind == "authority": advice["proof_authority"] = True
        elif kind == "source": advice["instruction_sha256"] = "0" * 64
        else: advice["status"] = "forged"
        # Re-signing untrusted local JSON cannot make changed authority valid.
        advice["advice_sha256"] = hashlib.sha256(advisor._encoded({k: v for k, v in advice.items() if k != "advice_sha256"})).hexdigest()
        prep._write(path, advice)
        prepared["intent_preplanning"]["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        prep._write(original[2] / "prepared.json", prepared)
    result, prompt = _plan(original[2], prepared, monkeypatch)
    assert prompt == _original_prompt(prepared)
    assert result["intent_preplanning"]["status"].startswith("fail_open_")
    assert result["intent_preplanning"]["supplied_to_router"] is False


def test_advisory_disk_failure_does_not_drop_the_task(original, monkeypatch):
    original_write = prep._write
    def reject_advice(path, value):
        if Path(path).name in {"intent-advice.json", "planner-intent-advice.json"}:
            raise OSError("injected optional artifact failure")
        return original_write(path, value)
    monkeypatch.setattr(prep, "_write", reject_advice)
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2])
    assert prepared["intent_preplanning"]["status"] == "fail_open_persistence_error"
    result, prompt = _plan(original[2], prepared, monkeypatch)
    assert prompt == _original_prompt(prepared)
    assert result["intent_preplanning"]["artifact_persisted"] is False


def test_portable_intent_candidate_infers_after_archive_relocation(tmp_path, intent_checkpoint):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", **_inputs(tmp_path),
        intent_checkpoint=intent_checkpoint)
    assert built["torch_cpu_requirement"] == "torch==2.13.0+cpu"
    assert built["security_training_requirements"] == []
    assert built["security_inference_requirements"] == []
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        raw = archive.extractfile(deployment.INTENT_CHECKPOINT_PATH).read()
        descriptor = json.load(archive.extractfile(deployment.INTENT_CHECKPOINT_DESCRIPTOR))
    assert hashlib.sha256(raw).hexdigest() == intent_checkpoint["sha256"]
    target = tmp_path / "offline-candidate.json"; target.write_bytes(raw)
    descriptor["path"] = str(target)
    advice = advisor.prepare_intent_advice(instruction="Inspect parser behavior.", checkpoint_descriptor=descriptor)
    assert advice["status"] == "feature_advice"
    args, observation = intent_asset_arguments(built)
    assert args == ["--intent-checkpoint-descriptor", deployment.ROOT + "/" + deployment.INTENT_CHECKPOINT_DESCRIPTOR]
    assert observation["sha256"] == intent_checkpoint["sha256"]
    assert intent_asset_arguments(built, enabled=False)[0] == ["--disable-intent-autoencoder"]


@pytest.mark.parametrize("change", ["path", "descriptor_path", "sha256", "mode", "training", "source_data"])
def test_intent_transport_retargeting_is_rejected(tmp_path, intent_checkpoint, change):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", **_inputs(tmp_path), intent_checkpoint=intent_checkpoint)
    forged = deepcopy(built)
    binding = forged["intent_checkpoint"]
    if change == "path": binding["descriptor"]["path"] = "/unselected.json"
    elif change == "descriptor_path": binding["descriptor_path"] = "../unselected.json"
    elif change == "sha256": binding["sha256"] = "0" * 64
    elif change == "mode": binding["mode"] = "training"
    elif change == "training": binding["runtime_training_steps"] = True
    else: binding["source_training_data_included"] = True
    with pytest.raises(ValueError): intent_asset_arguments(forged)


def test_ablation_selection_requires_boolean():
    with pytest.raises(ValueError): intent_asset_arguments({}, enabled="false")
    assert intent_asset_arguments({}) == ([], {"enabled": True, "status": "fail_open_no_checkpoint"})


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("disabled", [False, True])
def test_harbor_forwards_intent_selection_in_both_arms(tmp_path, intent_checkpoint, arm, disabled):
    from harbor.models.agent.context import AgentContext
    built = deployment.build_runtime_archive(output=tmp_path / "archive", **_inputs(tmp_path),
        intent_checkpoint=intent_checkpoint)
    commands = []
    report = {"task_completed": False, "seconds": 0, "phases": {}, "provider_invocations": []}
    class Environment:
        async def upload_file(self, *args): pass
        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(tmp_path / "archive"), arm=arm, disable_intent_autoencoder=disabled)
    context = AgentContext()
    asyncio.run(agent.run("Authored public instruction", Environment(), context))
    command = next(row for row in commands if "--arm" in row)
    assert command[command.index("--arm") + 1] == arm
    assert ("--disable-intent-autoencoder" in command) is disabled
    assert ("--intent-checkpoint-descriptor" in command) is not disabled
    assert context.metadata["intent_model_assets"]["enabled"] is not disabled


def test_bundle_cli_selects_pinned_intent_descriptor(tmp_path, intent_checkpoint, monkeypatch, capsys):
    inputs = _inputs(tmp_path)
    output = tmp_path / "archive"
    descriptor_path = _descriptor(tmp_path, intent_checkpoint)
    argv = ["terminal_deployment", "bundle", "--output", str(output)]
    for key, value in inputs.items(): argv += ["--" + key.replace("_", "-"), str(value)]
    argv += ["--intent-checkpoint-descriptor", str(descriptor_path)]
    monkeypatch.setattr(sys, "argv", argv)
    deployment.main()
    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    assert printed["archive_sha256"] == manifest["archive_sha256"]
    assert manifest["intent_checkpoint"]["sha256"] == intent_checkpoint["sha256"]


@pytest.mark.parametrize("disabled", ["false", 0, 1, None])
@pytest.mark.parametrize("archive_exists", [False, True])
def test_harbor_constructor_rejects_non_boolean_ablation_before_archive_io(tmp_path, monkeypatch, disabled, archive_exists):
    from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as cache

    def unexpected_cache_validation(*args, **kwargs):
        pytest.fail("invalid ablation switch reached archive cache validation")

    monkeypatch.setattr(cache, "validate_setup_cache_selection", unexpected_cache_validation)
    archive = tmp_path / "archive"
    if archive_exists:
        archive.mkdir()
    with pytest.raises(ValueError, match="ablation switch"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
            runtime_archive=str(archive), disable_intent_autoencoder=disabled)
