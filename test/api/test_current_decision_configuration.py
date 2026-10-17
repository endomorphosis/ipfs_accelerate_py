"""Current launch controls and authored transport; no provider or Docker calls.

Real source-member archives and capability declarations exercise preparation
and dispatch gates. Harbor dry-run and remote publisher results are explicit
process doubles, without benchmark acceptance, delivery, or savings claims.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import stat
import sys
import tarfile
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import semantic_decision_nomination_delivery as delivery
from benchmarks.agent_supervisor.container_coding import terminal_decision_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as resources
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE

DECISION_FIELDS = ('semantic_decision_policy_artifact', 'semantic_decision_policy_sha256', 'semantic_decision_nomination_artifact', 'semantic_decision_nomination_source', 'semantic_decision_nomination_sha256', 'disable_semantic_typed_nomination', 'semantic_retrieval_query_reference', 'semantic_decision_max_calls', 'semantic_decision_max_total_input_tokens')

POLICY = {"semantic_decision_policy_artifact": "/opt/ipfs-supervisor/authored/policy.json",
          "semantic_decision_policy_sha256": "d" * 64}
NOMINATION = {"semantic_decision_nomination_artifact": "/opt/ipfs-supervisor/authored/nomination.json",
              "semantic_decision_nomination_sha256": "e" * 64}
RAW = ('  {\r\n "schema":"supervisor-semantic-decision-nomination@1",\r\n'
       ' "policy":{"required_capsule_ids":["required-λ"],"optional_capsule_ids":["optional"],'
       '"selected_optional_capsule_ids":[]}\r\n}\n\t').encode()


def _source(tmp_path):
    source = tmp_path / "authored nomination café.json"
    source.write_bytes(RAW)
    return source, {"semantic_decision_nomination_source": str(source),
                    "semantic_decision_nomination_sha256": hashlib.sha256(RAW).hexdigest()}


def _archive(tmp_path, selected, *, omit=None):
    """Pack genuine current selected and capability bytes, never model assets."""
    archive = tmp_path / "archive"
    archive.mkdir()
    root = Path(adapter.__file__).resolve().parents[3]
    sources = {"source/" + name: root / name for name in capability.SOURCE_PATHS}
    sources.update({benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name: Path(benchmark.__file__).with_name(name)
                    for name in benchmark.TIMEOUT_RUNTIME_SOURCE_NAMES})
    if selected:
        sources.update(benchmark._decision_runtime_sources(
            retrieval_query_reference=selected.get("semantic_retrieval_query_reference", False)))
    sources.update(policy.decision_runtime_sources(policy.decision_selection(**selected)))
    rows = []
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name, source in sources.items():
            if name == omit:
                continue
            raw = source.read_bytes()
            mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
            rows.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw), "mode": mode})
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), mode
            bundle.addfile(member, io.BytesIO(raw))
    manifest = {"schema": "terminal-supervisor-runtime-archive@1",
                "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"),
                "codex_version": CLI_VERSION, "learned_requirements": [], "files": rows,
                capability.KEY: capability._current_capability()}
    benchmark._json(archive / "manifest.json", manifest)
    return archive, manifest


def _dataset(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Authored public preparation fixture.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    return dataset


def _prepare(tmp_path, monkeypatch, selected):
    archive, manifest = _archive(tmp_path, selected)
    calls = []

    def dry(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    output = tmp_path / "prepared"
    prepared = benchmark.prepare(dataset=_dataset(tmp_path), output=output, archive=archive,
                                 arm="full", **selected)
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    return output, prepared, manifest


@pytest.mark.parametrize("profile", [None, *resources.PROFILES])
def test_config_adds_only_explicit_decision_controls_and_keeps_current_model_resources(tmp_path, profile):
    paths = [tmp_path / name for name in ("dataset", "output", "archive")]
    ordinary = benchmark.config_for(*paths, "full", resource_profile=profile)
    selected = benchmark.config_for(*paths, "full", resource_profile=profile, **POLICY,
        semantic_decision_max_calls=2, semantic_decision_max_total_input_tokens=30000)
    expected = policy.decision_selection(**POLICY, semantic_decision_max_calls=2,
                                         semantic_decision_max_total_input_tokens=30000)
    assert not {key for key in DECISION_FIELDS if key.startswith("semantic_decision_")} & ordinary["agents"][0]["kwargs"].keys()
    assert selected["agents"][0]["kwargs"] == {**ordinary["agents"][0]["kwargs"], **expected}
    assert selected["agents"][0]["model_name"] == adapter.MODEL
    assert selected["agents"][0]["override_timeout_sec"] == resources.execution_budget(profile)["harbor_seconds"]
    ordinary_agent, selected_agent = deepcopy(ordinary["agents"][0]), deepcopy(selected["agents"][0])
    selected_agent["kwargs"] = {key: value for key, value in selected_agent["kwargs"].items()
                                if key not in DECISION_FIELDS}
    assert ordinary_agent == selected_agent
    assert {key: value for key, value in selected.items() if key != "agents"} == {
        key: value for key, value in ordinary.items() if key != "agents"}
    JobConfig.model_validate(selected, extra="forbid")


@pytest.mark.parametrize("options", [
    {"semantic_decision_policy_artifact": POLICY["semantic_decision_policy_artifact"]},
    {**POLICY, "semantic_decision_max_calls": True},
    {**POLICY, "semantic_decision_max_total_input_tokens": 0},
    {**POLICY, "semantic_retrieval_query_reference": True},
    {"semantic_retrieval_query_reference": True},
    {**POLICY, "planning_mode": "declared"},
    {**POLICY, "semantic_metadata_view": "common-bindings@1"},
    {**POLICY, "provider_profile": GROK_PROFILE},
])
def test_invalid_selected_configuration_refuses_before_any_filesystem_or_adapter_setup(tmp_path, options):
    def unread(*args, **kwargs):
        pytest.fail("invalid selected route reached filesystem or process")

    missing = SimpleNamespace(resolve=unread)
    with pytest.raises(ValueError):
        benchmark.config_for(missing, tmp_path / "output", missing, "full", **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=missing, output=tmp_path / "output", archive=missing, arm="full", **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=adapter.MODEL,
            runtime_archive=str(tmp_path / "missing"), arm="full", **options)
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_disabled_preparation_keeps_existing_shape(tmp_path, monkeypatch, arm):
    archive, _ = _archive(tmp_path, {})
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    result = benchmark.prepare(dataset=_dataset(tmp_path), output=tmp_path / "prepared", archive=archive, arm=arm)
    assert not {"semantic_decision_selection", "runtime_decision_source_sha256",
                "semantic_decision_nomination_snapshot"} & result.keys()


@pytest.mark.parametrize("selected", [POLICY, NOMINATION,
    {**NOMINATION, "semantic_retrieval_query_reference": True},
    {**NOMINATION, "semantic_retrieval_query_reference": True, "disable_semantic_typed_nomination": True}])
def test_child_local_preparation_pins_sources_without_reading_or_uploading_child_artifact(tmp_path, monkeypatch, selected):
    output, prepared, _ = _prepare(tmp_path, monkeypatch, selected)
    configured = json.loads((output / "config.json").read_text())
    expected = policy.decision_selection(**selected)
    held = prepared["prompt_comparison_options"]
    assert policy.decision_selection(**{key: value for key, value in held.items() if key in DECISION_FIELDS}) == expected
    benchmark._require_current_decision_runtime_pins(prepared["runtime_decision_source_sha256"],
        retrieval_query_reference=selected.get("semantic_retrieval_query_reference", False))
    assert benchmark._prompt_option_observation(prepared, configured["agents"][0]["kwargs"])["prompt_comparison_config_unchanged"] is True
    assert "semantic_decision_nomination_snapshot" not in prepared
    assert prepared["provider_calls"] == 0 and prepared["benchmark_advantage_claimed"] is False
    assert not list(output.glob("semantic-decision-nomination-*.json"))


def test_host_preparation_freezes_exact_bytes_and_uses_only_snapshot_path(tmp_path, monkeypatch):
    source, selected = _source(tmp_path)
    output, prepared, _ = _prepare(tmp_path, monkeypatch, selected)
    snapshot = prepared["semantic_decision_nomination_snapshot"]
    assert delivery.verify_nomination_snapshot(snapshot, directory=output) == RAW
    assert stat.S_IMODE(Path(snapshot["artifact"]).stat().st_mode) == 0o444
    configured = json.loads((output / "config.json").read_text())["agents"][0]["kwargs"]
    assert configured["semantic_decision_nomination_source"] == snapshot["artifact"] != str(source)
    source.write_bytes(b"authored source changed after immutable preparation")
    assert delivery.verify_nomination_snapshot(snapshot, directory=output) == RAW


@pytest.mark.parametrize("fault", ["missing", "population", "digest", "selection", "snapshot"])
def test_execute_rejects_selected_preparation_drift_before_invocation(tmp_path, monkeypatch, fault):
    _, selected = _source(tmp_path)
    output, prepared, _ = _prepare(tmp_path, monkeypatch, selected)
    if fault == "missing":
        prepared.pop("runtime_decision_source_sha256")
    elif fault == "population":
        prepared["runtime_decision_source_sha256"].pop(next(iter(prepared["runtime_decision_source_sha256"])))
    elif fault == "digest":
        prepared["runtime_decision_source_sha256"][next(iter(prepared["runtime_decision_source_sha256"]))] = "0" * 64
    elif fault == "selection":
        prepared["prompt_comparison_options"]["semantic_decision_max_calls"] = 2
    else:
        prepared["semantic_decision_nomination_snapshot"]["bytes"] += 1
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("changed selected preparation reached native invocation"))
    with pytest.raises(ValueError):
        benchmark._execute_once(output)
    assert not (output / "invocation.json").exists()


def test_execute_uses_retained_snapshot_after_original_host_source_is_removed(tmp_path, monkeypatch):
    source, selected = _source(tmp_path)
    output, _, _ = _prepare(tmp_path, monkeypatch, selected)
    source.unlink()

    class InvocationReached(Exception):
        pass

    def intercepted(*args, **kwargs):
        raise InvocationReached

    monkeypatch.setattr(benchmark.subprocess, "run", intercepted)
    with pytest.raises(InvocationReached):
        benchmark._execute_once(output)
    assert (output / "invocation.json").is_file()


def test_self_consistent_missing_selected_archive_member_refuses_before_output_or_process(tmp_path, monkeypatch):
    selected = policy.decision_selection(**POLICY)
    # This member is outside the preexisting generic worker-capability list.
    target = "source/benchmarks/agent_supervisor/container_coding/terminal_decision_policy.py"
    archive, _ = _archive(tmp_path, selected, omit=target)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("incomplete selected archive reached Harbor dry run"))
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=_dataset(tmp_path), output=tmp_path / "prepared", archive=archive,
                          arm="full", **selected)
    assert not (tmp_path / "prepared").exists()


def test_cli_forwards_query_reference_once_and_execute_cannot_reselect(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"provider_calls": 0})
    arguments = ["benchmark", "prepare", "--dataset", str(tmp_path / "dataset"),
        "--output", str(tmp_path / "output"), "--archive", str(tmp_path / "archive"),
        "--semantic-decision-nomination-artifact", NOMINATION["semantic_decision_nomination_artifact"],
        "--semantic-decision-nomination-sha256", NOMINATION["semantic_decision_nomination_sha256"],
        "--disable-semantic-typed-nomination", "--semantic-retrieval-query-reference",
        "--semantic-decision-max-calls", "2", "--semantic-decision-max-total-input-tokens", "30000"]
    monkeypatch.setattr(sys, "argv", arguments)
    benchmark.main()
    assert json.loads(capsys.readouterr().out) == {"provider_calls": 0}
    selected = {key: value for key, value in calls[0].items() if key in DECISION_FIELDS}
    argv = policy.semantic_decision_arguments(**selected)
    assert argv.count("--semantic-retrieval-query-reference") == 1
    assert argv[argv.index("--semantic-decision-max-calls") + 1] == "2"
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path / "output"),
        "--semantic-decision-max-calls", "2"])
    monkeypatch.setattr(benchmark, "execute", lambda *_args, **_kwargs:
                        pytest.fail("execute cannot reselect the decision route"))
    # The retained peer validator can reject the stray budget before argparse
    # reaches its operation-selection check. Both refusals must stop execution.
    with pytest.raises((SystemExit, ValueError)) as error:
        benchmark.main()
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
    assert len(calls) == 1


class _Environment:
    """Authored publisher receipts; no remote code, containers or providers."""
    def __init__(self, raw=RAW, *, failure=None):
        self.raw, self.failure = raw, failure
        self.events, self.commands, self.uploads, self.downloads = [], [], [], []

    async def upload_file(self, source, target):
        self.uploads.append(target)
        if target.endswith("/.upload.json"):
            self.events.append("nomination_upload")
            assert Path(source).read_bytes() == self.raw
            assert stat.S_IMODE(Path(source).stat().st_mode) == 0o444
            if self.failure == "upload":
                raise RuntimeError("authored upload failure")
        else:
            self.events.append("instruction_upload")

    async def exec(self, **kwargs):
        args = shlex.split(kwargs["command"])
        self.commands.append((args, kwargs))
        if kwargs["user"] != "root":
            self.events.append("driver")
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        assert args[:3] == [adapter.PYTHON, "-P", "-c"] and args[3] == delivery.NOMINATION_DELIVERY_CODE
        operation, root, run_id, digest, count = args[4:]
        self.events.append("nomination_" + operation)
        locations = delivery.nomination_delivery_arguments(root=root, run_id=run_id,
            expected_sha256=digest, expected_bytes=int(count))
        receipt = {"schema": delivery.DELIVERY_SCHEMA, "operation": operation,
            "published": operation == "publish", "artifact": locations["artifact"],
            "sha256": digest, "bytes": int(count), "provider_called": False, "completion_authority": False}
        if self.failure == "receipt" and operation == "publish":
            receipt["bytes"] = True
        return SimpleNamespace(return_code=0, stdout=json.dumps(receipt), stderr="")

    async def download_file(self, source, target):
        self.downloads.append(source)
        raise FileNotFoundError(source)


def _agent(tmp_path, monkeypatch, *, selected, environment=None):
    archive, _ = _archive(tmp_path, selected)
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=adapter.MODEL,
        runtime_archive=str(archive), arm="full", **selected)
    environment = environment or _Environment()

    async def capture(*args, **kwargs):
        environment.events.append("public_input_capture")
        return {"status": "authored"}

    async def export(*args, **kwargs):
        environment.events.append("public_output_export")
        return {"status": "authored"}

    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", export)
    return agent, environment, AgentContext()


@pytest.mark.parametrize("disabled", [False, True])
def test_adapter_delivers_snapshot_before_instruction_and_forwards_only_derived_child_controls(tmp_path, monkeypatch, disabled):
    source, selected = _source(tmp_path)
    selected.update(disable_semantic_typed_nomination=disabled, semantic_retrieval_query_reference=True,
                    semantic_decision_max_calls=2, semantic_decision_max_total_input_tokens=30000)
    agent, environment, context = _agent(tmp_path, monkeypatch, selected=selected)
    observed = {}
    original = adapter._AdapterRunDeadline.stage

    async def stage(self, name, operation, *, maximum=None, reserve=adapter.ADAPTER_RECEIPT_MARGIN_SECONDS):
        observed[name] = maximum, reserve
        return await original(self, name, operation, maximum=maximum, reserve=reserve)

    monkeypatch.setattr(adapter._AdapterRunDeadline, "stage", stage)
    asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert environment.events[:4] == ["nomination_prepare", "nomination_upload", "nomination_publish", "instruction_upload"]
    assert all(observed[name] == (15, 25) for name in
               ("nomination_prepare", "nomination_upload", "nomination_publish"))
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    assert transfer["published"] is True and transfer["completion_authority"] is False
    argv, _ = next((args, kwargs) for args, kwargs in environment.commands if kwargs["user"] == "supervisor")
    assert argv[argv.index("--semantic-decision-nomination-artifact") + 1] == transfer["artifact"]
    assert str(source) not in argv and "--semantic-decision-nomination-source" not in argv
    assert argv.count("--semantic-retrieval-query-reference") == 1
    assert ("--disable-semantic-typed-nomination" in argv) is disabled
    assert argv[argv.index("--semantic-decision-max-total-input-tokens") + 1] == "30000"
    assert context.metadata["adapter_deadline"]["finalization_reserve_seconds"] == 25
    assert len(environment.downloads) == 2 and environment.events[-1] == "public_output_export"
    assert context.n_input_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("failure", ["upload", "receipt"])
def test_nomination_failure_prevents_driver_but_preserves_existing_finalization(tmp_path, monkeypatch, failure):
    _, selected = _source(tmp_path)
    agent, environment, context = _agent(tmp_path, monkeypatch, selected=selected,
                                         environment=_Environment(failure=failure))
    with pytest.raises(RuntimeError if failure == "upload" else ValueError):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert "driver" not in environment.events and "instruction_upload" not in environment.events
    assert context.metadata["semantic_decision_nomination_delivery"]["status"] == "failed"
    assert len(environment.downloads) == 2 and environment.events[-1] == "public_output_export"
    assert context.n_input_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("phase", ["setup", "run"])
def test_adapter_selection_drift_refuses_before_environment_io(tmp_path, monkeypatch, phase):
    agent, environment, context = _agent(tmp_path, monkeypatch, selected=POLICY)
    agent.semantic_decision_max_calls = 2
    with pytest.raises(ValueError, match="selection changed"):
        asyncio.run(agent.setup(environment) if phase == "setup" else agent.run("Authored", environment, context))
    assert environment.events == [] and environment.downloads == []


def test_adapter_changed_archive_refuses_before_setup_environment(tmp_path, monkeypatch):
    agent, environment, _ = _agent(tmp_path, monkeypatch, selected=POLICY)
    (agent.runtime_archive / "runtime.tar.gz").write_bytes(b"authored invalid archive")
    with pytest.raises(ValueError, match="selected decision"):
        asyncio.run(agent.setup(environment))
    assert environment.events == []


@pytest.mark.parametrize("selected", [POLICY, NOMINATION])
def test_adapter_child_local_controls_are_forwarded_without_host_delivery(tmp_path, monkeypatch, selected):
    controls = {**selected, "semantic_decision_max_calls": 2}
    agent, environment, context = _agent(tmp_path, monkeypatch, selected=controls)
    asyncio.run(agent.run("Authored public instruction.", environment, context))
    args, _ = next((args, kwargs) for args, kwargs in environment.commands if kwargs["user"] == "supervisor")
    expected = policy.semantic_decision_arguments(**controls)
    assert args[-len(expected):] == expected
    assert not any(event.startswith("nomination_") for event in environment.events)
    assert "semantic_decision_nomination_delivery" not in context.metadata
    assert len(environment.uploads) == 1 and environment.uploads[0].endswith("/instruction.md")


def test_nomination_upload_cancellation_keeps_original_error_and_finalizes(tmp_path, monkeypatch):
    _, selected = _source(tmp_path)
    error = asyncio.CancelledError("authored cancellation")

    class Cancelled(_Environment):
        async def upload_file(self, source, target):
            if target.endswith("/.upload.json"):
                self.events.append("nomination_upload")
                raise error
            return await super().upload_file(source, target)

    agent, environment, context = _agent(tmp_path, monkeypatch, selected=selected, environment=Cancelled())
    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert caught.value is error
    assert "driver" not in environment.events
    assert len(environment.downloads) == 2 and environment.events[-1] == "public_output_export"
    assert context.metadata["semantic_decision_nomination_delivery"]["error_type"] == "CancelledError"


def _controls(config):
    from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import resolve_provider_profile
    selected = resolve_provider_profile(config["agents"][0]["kwargs"].get("provider_profile"))
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, **{key: selected[key] for key in ("model", "reasoning_effort", "cli_version")})


@pytest.mark.parametrize("selected", [
    POLICY, NOMINATION,
    {**NOMINATION, "semantic_retrieval_query_reference": True},
    {"semantic_decision_nomination_source": "/authored-host/nomination.json",
     "semantic_decision_nomination_sha256": "c" * 64},
    {**NOMINATION, "disable_semantic_typed_nomination": True,
     "semantic_retrieval_query_reference": True, "semantic_decision_max_calls": 2,
     "semantic_decision_max_total_input_tokens": 30000},
])
def test_declared_controls_validate_selected_group_and_seal_complete_config_without_new_schema(tmp_path, selected):
    from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
    args = (tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    ordinary = benchmark.config_for(*args)
    configured = benchmark.config_for(*args, **selected)
    # Exercise the same genuine validated serialization used during prepare.
    wire = JobConfig.model_validate(configured, extra="forbid").model_dump(mode="json")
    declared = _controls(wire)
    ordinary_declared = _controls(JobConfig.model_validate(ordinary, extra="forbid").model_dump(mode="json"))
    assert declared["schema"] == ordinary_declared["schema"] == controls.SCHEMA
    assert declared.keys() == ordinary_declared.keys()
    assert controls.validate_controls(declared) is True
    assert declared["configuration_sha256"] == controls._digest(wire)
    assert declared["configuration_sha256"] != ordinary_declared["configuration_sha256"]
    assert declared["job_controls_sha256"] == ordinary_declared["job_controls_sha256"]
    assert declared["agent_controls_sha256"] == ordinary_declared["agent_controls_sha256"]
    changed = deepcopy(wire)
    changed["agents"][0]["kwargs"]["semantic_decision_max_calls"] = 1
    assert _controls(changed)["configuration_sha256"] != declared["configuration_sha256"]


@pytest.mark.parametrize("changes", [
    {"semantic_decision_policy_sha256": None},
    {"semantic_decision_policy_sha256": "not-a-pin"},
    {"semantic_decision_max_calls": True},
    {"semantic_decision_max_total_input_tokens": 0},
    {"semantic_decision_nomination_source": "/authored-host/nomination.json",
     "semantic_decision_nomination_sha256": "c" * 64},
    {"semantic_retrieval_query_reference": True},
    {"disable_semantic_typed_nomination": True},
    {"arm": "no-index"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_context_profile": "digest-query@1"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"coding_reply_mode": "ordinary-completion@1"},
    {"source_view_policy": {"schema": "authored-unsupported"}},
    {"unknown_control": True},
])
def test_closed_control_allowlist_does_not_admit_malformed_or_unsupported_selected_groups(tmp_path, changes):
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full", **POLICY)
    config["agents"][0]["kwargs"].update(changes)
    with pytest.raises(ValueError):
        _controls(config)


@pytest.mark.parametrize("where", ["mounts", "agent_env", "extra_instruction_paths"])
def test_selected_controls_keep_external_input_refusal(tmp_path, where):
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full", **POLICY)
    if where == "mounts":
        config["environment"]["mounts"] = ["/authored/external"]
    elif where == "agent_env":
        config["agents"][0]["env"] = {"AUTHORED_UNSEALED": "value"}
    else:
        config["extra_instruction_paths"] = ["/authored/external.md"]
    with pytest.raises(ValueError):
        _controls(config)


def _retained_report(mode="typed"):
    if mode == "policy":
        from test.api.test_terminal_decision_usage import _native, _attempt, _dialogue
        native = _native()
        dialogue = _dialogue(_attempt(native))
        dialogue["worker_launch"] = {"launch_policy_artifact": POLICY["semantic_decision_policy_artifact"],
            "launch_policy_sha256": POLICY["semantic_decision_policy_sha256"]}
        return {"provider_invocations": [native], "semantic_decision_dialogues": [dialogue],
                "semantic_decision_dialogue_expected": True}, POLICY
    from test.api.test_semantic_typed_nomination_usage import _case
    report = _case()
    selected = {"semantic_decision_nomination_artifact": report["semantic_decision_owner_bindings"][0]["nomination_artifact"],
                "semantic_decision_nomination_sha256": report["semantic_decision_nomination_sha256"]}
    if mode == "untyped":
        report["semantic_decision_owner_bindings"][0].pop("typed_nomination")
        report["semantic_decision_typed_nomination_expected"] = False
        report.pop("context_input_audit")
        selected["disable_semantic_typed_nomination"] = True
    return report, selected


def _collected_fixture(tmp_path, monkeypatch, report, selected):
    output, prepared, _ = _prepare(tmp_path, monkeypatch, selected)
    config = json.loads((output / "config.json").read_text())
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored-trial"
    (trial / "agent").mkdir(parents=True)
    benchmark._json(job / "result.json", {"stats": {}})
    benchmark._json(trial / "result.json", {"task_name": benchmark.TASK,
        "config": {"task": {"path": str(Path(prepared["dataset"]) / benchmark.TASK)}},
        "verifier_result": {"rewards": {"reward": 1}}})
    if report is not None:
        benchmark._json(trial / "agent/supervisor-result.json", report)
    return output, prepared, trial


@pytest.mark.parametrize("mode", ["policy", "typed", "untyped"])
def test_collect_binds_valid_selected_report_to_request_and_recomputes_original_native_receipts(tmp_path, monkeypatch, mode):
    report, selected = _retained_report(mode)
    report["provider_usage_accounting"] = {"report_provider_total_tokens": 999999, "authoritative_usage_coverage_complete": False}
    before = deepcopy(report)
    output, _, trial = _collected_fixture(tmp_path, monkeypatch, report, selected)
    observed = benchmark.collect(output)
    binding = observed["trials"][0]["semantic_decision_dispatch"]
    assert binding["verified"] is True and binding["problems"] == []
    assert binding["authoritative_usage_coverage_complete"] is binding["native_usage_complete"] is True
    assert binding["retained_receipt_usage_recomputed"] is True and binding["stored_usage_summary_trusted"] is False
    assert observed["semantic_decision_runtime_verified"] is True
    assert observed["semantic_decision_config_unchanged"] is True
    assert observed["complete_single_trial_receipt"] is True
    assert observed["benchmark_advantage_claimed"] is False
    assert json.loads((trial / "agent/supervisor-result.json").read_text()) == before == report


@pytest.mark.parametrize("fault", ["report_missing", "all_fields_missing", "dialogue_flag", "owner_flag", "nomination", "typed_missing", "typed_wrong", "owner_task", "native_incomplete", "stored_summary_only"])
def test_selected_collect_cannot_complete_from_reward_or_partial_foreign_stored_evidence(tmp_path, monkeypatch, fault):
    report, selected = _retained_report()
    if fault == "report_missing":
        report = None
    elif fault == "all_fields_missing":
        report = {"provider_invocations": report["provider_invocations"]}
    elif fault == "dialogue_flag":
        report.pop("semantic_decision_dialogue_expected")
    elif fault == "owner_flag":
        report.pop("semantic_decision_owner_binding_expected")
    elif fault == "nomination":
        report["semantic_decision_nomination_sha256"] = "f" * 64
    elif fault == "typed_missing":
        report.pop("semantic_decision_typed_nomination_expected")
    elif fault == "typed_wrong":
        report["semantic_decision_typed_nomination_expected"] = False
    elif fault == "owner_task":
        report["semantic_decision_expected_task_id"] = "foreign-task"
    elif fault == "native_incomplete":
        report["provider_invocations"][0]["native_rollout_usage"]["usage_lifecycle_complete"] = False
        report["semantic_decision_dialogues"][0]["attempts"][0]["provider_receipt"]["native_invocation"]["native_rollout_usage"]["usage_lifecycle_complete"] = False
    else:
        report = {"provider_usage_accounting": {"report_provider_total_tokens": 120,
            "semantic_decision_coverage": {"authoritative_usage_coverage_complete": True}}}
    before = deepcopy(report)
    output, _, _ = _collected_fixture(tmp_path, monkeypatch, report, selected)
    observed = benchmark.collect(output)
    binding = observed["trials"][0]["semantic_decision_dispatch"]
    assert binding["verified"] is False and binding["problems"]
    assert observed["semantic_decision_runtime_verified"] is False
    assert observed["complete_single_trial_receipt"] is False
    assert report == before


@pytest.mark.parametrize("field", ["launch_policy_sha256", "launch_policy_artifact"])
def test_collect_prebound_policy_must_match_independent_requested_launch_pair(tmp_path, monkeypatch, field):
    report, selected = _retained_report("policy")
    report["semantic_decision_dialogues"][0]["worker_launch"][field] = "f" * 64 if field.endswith("sha256") else "/foreign/policy.json"
    output, _, _ = _collected_fixture(tmp_path, monkeypatch, report, selected)
    observed = benchmark.collect(output)
    binding = observed["trials"][0]["semantic_decision_dispatch"]
    assert "decision_launch_policy_mismatch" in binding["problems"]
    assert observed["complete_single_trial_receipt"] is False


def test_collect_selected_config_drift_refuses_without_rewriting_retained_report(tmp_path, monkeypatch):
    report, selected = _retained_report()
    output, prepared, trial = _collected_fixture(tmp_path, monkeypatch, report, selected)
    before = (trial / "agent/supervisor-result.json").read_bytes()
    prepared["prompt_comparison_options"]["semantic_decision_max_calls"] = 1
    benchmark._json(output / "preparation.json", prepared)
    with pytest.raises(ValueError, match="selection changed"):
        benchmark.collect(output)
    assert not (output / "receipt.json").exists()
    assert (trial / "agent/supervisor-result.json").read_bytes() == before


def test_collect_selected_query_cannot_borrow_ordinary_untyped_nomination_usage(tmp_path, monkeypatch):
    report, selected = _retained_report("untyped")
    selected["semantic_retrieval_query_reference"] = True
    output, _, _ = _collected_fixture(tmp_path, monkeypatch, report, selected)
    observed = benchmark.collect(output)
    assert "decision_query_reference_binding_unavailable_or_mismatched" in observed["trials"][0]["semantic_decision_dispatch"]["problems"]
    assert observed["complete_single_trial_receipt"] is False


def test_ordinary_collect_shape_remains_free_of_selected_decision_observations(tmp_path, monkeypatch):
    output, _, _ = _collected_fixture(tmp_path, monkeypatch, {"provider_invocations": []}, {})
    observed = benchmark.collect(output)
    assert not {"semantic_decision_selection", "semantic_decision_config_unchanged",
                "semantic_decision_runtime_verified", "semantic_decision_verification_scope"} & observed.keys()
    assert "semantic_decision_dispatch" not in observed["trials"][0]
    assert observed["complete_single_trial_receipt"] is True
