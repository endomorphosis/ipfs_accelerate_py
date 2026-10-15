"""Authored Harbor transport tests; remote isolation is qualified separately."""
import asyncio
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import terminal_metadata_tokenizer_bundle as bundle
from benchmarks.agent_supervisor.container_coding import terminal_metadata_tokenizer_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_semantic_metadata_policy as metadata
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def captured(tmp_path):
    asset = tmp_path / "original-tokenizer.json"
    asset.write_text('{"truncation":null,"padding":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')
    descriptor = tmp_path / "original-descriptor.json"
    descriptor.write_text(json.dumps(dict(schema="supervisor-local-prompt-tokenizer@1",
        provider="codex_cli", model="gpt-6.1-sol", tokenizer_id="authored-local",
        tokenizer_revision="tokenizers/0.1.0/sha256:" + _sha(asset.read_bytes()), special_token_policy="tokenizers.encode:add_special_tokens=false",
        backend="tokenizers-json@1", tokenizers_version="0.1.0", asset_path=str(asset), asset_sha256=_sha(asset.read_bytes()))))
    selected = bundle.stage_metadata_tokenizer_bundle(artifact=descriptor,
        expected_sha256=_sha(descriptor.read_bytes()), output=tmp_path / "bundle",
        provider="codex_cli", model="gpt-6.1-sol")
    snapshot = bundle.read_metadata_tokenizer_bundle(selected, provider="codex_cli", model="gpt-6.1-sol",
        semantic_metadata_view="common-bindings@1")
    return selected, snapshot


def _manifest():
    checkout = Path(__file__).resolve().parents[3]
    manifest = _supervisor_manifest(dict(archive_sha256="authored-archive", codex_version="0.160.0",
                                         learned_requirements=[]))
    sources = set(metadata.SOURCE_PATHS) | set(bundle.SOURCE_PATHS)
    by_path = {row["path"]: row for row in manifest["files"]}
    for name in sources:
        raw = (checkout / name).read_bytes()
        by_path["source/" + name] = dict(path="source/" + name, bytes=len(raw), sha256=_sha(raw), mode=0o644)
    manifest["files"] = list(by_path.values())
    return manifest


def _agent(tmp_path, selected, **changes):
    archive = tmp_path / "archive"
    archive.mkdir(exist_ok=True)
    (archive / "manifest.json").write_text(json.dumps(_manifest()))
    options = dict(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol", runtime_archive=str(archive),
                   arm="full", semantic_metadata_view="common-bindings@1", semantic_metadata_tokenizer=selected)
    options.update(changes)
    return adapter.FullSupervisorAgent(**options)


class AuthoredEnvironment:
    """Return explicit protocol responses; no remote filesystem is claimed."""
    def __init__(self, snapshot, fail=None):
        self.snapshot, self.fail = snapshot, fail
        self.events, self.uploads = [], []

    async def upload_file(self, source, target):
        self.events.append(("upload", target))
        self.uploads.append((target, Path(source).read_bytes()))

    async def exec(self, **kwargs):
        argv = shlex.split(kwargs["command"])
        if "-c" not in argv:
            self.events.append(("driver", argv))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        tree = ast.parse(argv[argv.index("-c") + 1])
        constants = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                break
            constants[node.targets[0].id] = ast.literal_eval(node.value)
        phase = constants.get("phase", "worker")
        self.events.append((phase, kwargs["user"]))
        if phase == self.fail:
            return SimpleNamespace(return_code=1, stdout="", stderr="authored refusal")
        if phase == "worker":
            observed = dict(schema="terminal-tokenizer-worker-readiness@1", verified=True,
                descriptor_sha256=self.snapshot.manifest()["descriptor_sha256"],
                asset_sha256=self.snapshot.manifest()["asset_sha256"], tokenizers_version="0.1.0", probe_tokens=4)
        else:
            observed = dict(schema="terminal-tokenizer-root-verification@1", phase=phase,
                manifest_sha256=self.snapshot.selection()["manifest_sha256"], verified=True)
        return SimpleNamespace(return_code=0, stdout=json.dumps(observed), stderr="")

    async def download_file(self, source, target):
        report = dict(task_completed=False, seconds=0, phases={}, provider_invocations=[])
        Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))


def test_deployment_uploads_only_captured_portable_bytes_and_checks_worker(tmp_path, captured):
    selected, snapshot = captured
    # Once captured, neither the original asset nor the private bundle is read.
    shutil.rmtree(selected["bundle_path"])
    (tmp_path / "original-tokenizer.json").unlink()
    environment = AuthoredEnvironment(snapshot)
    observed = asyncio.run(deployment.deploy_metadata_tokenizer_bundle(environment,
        snapshot=snapshot, output=tmp_path / "deployment"))
    descriptor, pin = deployment.metadata_tokenizer_remote_descriptor(snapshot)
    assert descriptor == adapter.ROOT + "/metadata-tokenizers/" + selected["manifest_sha256"] + "/descriptor.json"
    assert pin == snapshot.manifest()["descriptor_sha256"]
    assert [Path(target).name for target, _ in environment.uploads] == ["descriptor.json", "tokenizer.json"]
    expected = dict(snapshot.files())
    assert all(raw == expected[Path(target).name] for target, raw in environment.uploads)
    assert json.loads(environment.uploads[0][1])["asset_path"] == "tokenizer.json"
    assert environment.events[0] == ("create", "root")
    assert environment.events[-3:] == [("publish", "root"), ("verify", "root"), ("worker", "benchmarkworker")]
    assert observed["worker_runtime_verified"] is True and observed["provider_calls"] == 0
    assert observed["provider_tokenizer_equivalence_verified"] is False
    assert not any("original-descriptor" in target or target.startswith("/app/") for target, _ in environment.uploads)


@pytest.mark.parametrize("phase", ["create", "publish", "verify", "worker"])
def test_deployment_refusal_cannot_produce_a_ready_descriptor(tmp_path, captured, phase):
    _, snapshot = captured
    environment = AuthoredEnvironment(snapshot, fail=phase)
    with pytest.raises(RuntimeError, match="verification failed"):
        asyncio.run(deployment.deploy_metadata_tokenizer_bundle(environment,
            snapshot=snapshot, output=tmp_path / "deployment"))
    assert not (tmp_path / "deployment/observation.json").exists()
    assert environment.events[-1][0] == phase


@pytest.mark.parametrize("return_code", [None, False, ""])
def test_deployment_requires_explicit_integer_zero_settlement_before_upload(tmp_path, captured, return_code):
    _, snapshot = captured
    class UnsettledEnvironment(AuthoredEnvironment):
        async def exec(self, **kwargs):
            result = await super().exec(**kwargs)
            result.return_code = return_code
            return result
    environment = UnsettledEnvironment(snapshot)
    with pytest.raises(RuntimeError, match="verification failed"):
        asyncio.run(deployment.deploy_metadata_tokenizer_bundle(environment,
            snapshot=snapshot, output=tmp_path / "deployment"))
    assert environment.events == [("create", "root")]
    assert environment.uploads == []
    assert not (tmp_path / "deployment/observation.json").exists()


@pytest.mark.parametrize("field,value", [("verified", 1), ("manifest_sha256", []), ("phase", None)])
def test_deployment_requires_type_exact_root_observation_before_upload(tmp_path, captured, field, value):
    _, snapshot = captured
    class MalformedObservationEnvironment(AuthoredEnvironment):
        async def exec(self, **kwargs):
            result = await super().exec(**kwargs)
            observed = json.loads(result.stdout)
            observed[field] = value
            result.stdout = json.dumps(observed)
            return result
    environment = MalformedObservationEnvironment(snapshot)
    with pytest.raises(ValueError, match="observation differs"):
        asyncio.run(deployment.deploy_metadata_tokenizer_bundle(environment,
            snapshot=snapshot, output=tmp_path / "deployment"))
    assert environment.events == [("create", "root")]
    assert environment.uploads == []
    assert not (tmp_path / "deployment/observation.json").exists()


def test_generated_remote_scripts_are_bounded_and_do_not_import_application(captured):
    _, snapshot = captured
    for phase in ("create", "publish", "verify"):
        ast.parse(deployment._root_script(snapshot, phase))
    worker = deployment._worker_script(snapshot)
    tree = ast.parse(worker)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == []
    assert "load_owner_prompt_tokenizer" in worker and "OwnerPromptTokenizer.count" in worker
    assert "provider_calls" not in worker and "generate_text" not in worker


def test_adapter_capture_survives_deleted_bundle_and_refuses_mutable_selection(tmp_path, captured):
    selected, _ = captured
    agent = _agent(tmp_path, selected)
    captured_pin = agent._checked_metadata_tokenizer().selection()["manifest_sha256"]
    selected["manifest_sha256"] = "b" * 64  # Constructor made a detached selection.
    shutil.rmtree(agent.semantic_metadata_tokenizer["bundle_path"])
    assert agent._checked_metadata_tokenizer().selection()["manifest_sha256"] == captured_pin
    agent.semantic_metadata_tokenizer["manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="selection changed"):
        agent._checked_metadata_tokenizer()


@pytest.mark.parametrize("field,value", [("arm", "no-index"), ("semantic_metadata_view", "legacy"),
    ("semantic_transport_schema", "supervisor-semantic-router-input@2"), ("metadata_census_only", True),
    ("model_name", "different-model"), ("provider_profile", "grok-4.7-cli-1.0.46@1")])
def test_adapter_rechecks_frozen_route_before_environment_work(tmp_path, captured, field, value):
    agent = _agent(tmp_path, captured[0])
    setattr(agent, field, value)
    environment = AuthoredEnvironment(captured[1])
    with pytest.raises(ValueError, match="route changed"):
        asyncio.run(agent.setup(environment))
    assert environment.events == []


def test_adapter_requires_verified_setup_before_run(tmp_path, captured):
    agent = _agent(tmp_path, captured[0])
    environment = AuthoredEnvironment(captured[1])
    with pytest.raises(ValueError, match="deployment must be verified"):
        asyncio.run(agent.run("Public instruction", environment, AgentContext()))
    assert environment.events == [] and environment.uploads == []


def test_selected_adapter_requires_current_archive_before_deployment(tmp_path, captured):
    agent = _agent(tmp_path, captured[0])
    path = agent.runtime_archive / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = [row for row in manifest["files"] if not row["path"].endswith("terminal_metadata_tokenizer_deployment.py")]
    path.write_text(json.dumps(manifest))
    environment = AuthoredEnvironment(captured[1])
    with pytest.raises(ValueError, match="current_runtime_archive"):
        asyncio.run(agent.setup(environment))
    assert environment.events == []


def test_setup_and_run_verify_remote_assets_before_driver_and_forward_fixed_pair(tmp_path, captured, monkeypatch):
    agent = _agent(tmp_path, captured[0])
    environment = AuthoredEnvironment(captured[1])
    monkeypatch.setattr(adapter, "deploy_supervisor", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "deploy_worker_boundary", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={}))
    asyncio.run(agent.setup(environment))
    assert agent._metadata_tokenizer_remote_ready is True
    context = AgentContext()
    assert context.metadata is None  # Preserve the real Harbor default until run initializes its report.
    asyncio.run(agent.run("Public instruction", environment, context))
    driver = next(event[1] for event in environment.events if event[0] == "driver")
    descriptor, pin = deployment.metadata_tokenizer_remote_descriptor(captured[1])
    assert driver[driver.index("--semantic-metadata-tokenizer-artifact") + 1] == descriptor
    assert driver[driver.index("--semantic-metadata-tokenizer-sha256") + 1] == pin
    assert captured[0]["bundle_path"] not in driver
    driver_index = next(index for index, event in enumerate(environment.events) if event[0] == "driver")
    assert environment.events[driver_index - 2][0] == "worker"
    assert context.metadata["semantic_metadata_tokenizer_bundle"]["manifest_sha256"] == captured[0]["manifest_sha256"]
    assert context.metadata["semantic_metadata_tokenizer_deployment"]["worker_runtime_verified"] is True


def test_run_remote_refusal_occurs_before_instruction_upload_or_planning(tmp_path, captured):
    agent = _agent(tmp_path, captured[0])
    agent._metadata_tokenizer_remote_ready = True
    environment = AuthoredEnvironment(captured[1], fail="verify")
    with pytest.raises(RuntimeError, match="verification failed"):
        asyncio.run(agent.run("Public instruction", environment, AgentContext()))
    assert environment.events == [("verify", "root")] and environment.uploads == []


def test_unselected_adapter_performs_no_tokenizer_deployment(tmp_path, monkeypatch):
    agent = _agent(tmp_path, None, semantic_metadata_view="legacy", arm="no-index")
    monkeypatch.setattr(deployment, "deploy_metadata_tokenizer_bundle", AsyncMock(side_effect=AssertionError("unexpected tokenizer")))
    monkeypatch.setattr(deployment, "verify_deployed_metadata_tokenizer_bundle", AsyncMock(side_effect=AssertionError("unexpected tokenizer")))
    monkeypatch.setattr(adapter, "deploy_supervisor", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "deploy_worker_boundary", AsyncMock(return_value={}))
    assert agent._checked_metadata_tokenizer() is None
    asyncio.run(agent.setup(AuthoredEnvironment(None)))
