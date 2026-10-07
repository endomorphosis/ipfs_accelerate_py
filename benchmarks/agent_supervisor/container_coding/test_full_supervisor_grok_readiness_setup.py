"""Harbor setup routes through provider readiness before any coding invocation.

Runtime and worker installation are explicit async doubles. The readiness
consumer itself parses the authored transport response; no executable, secret,
container, network endpoint or model is accessed by these tests.
"""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
from benchmarks.agent_supervisor.container_coding import terminal_task_bootstrap as bootstrap
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import (
    CLI_VERSION, CODEX_PROFILE, GROK_PROFILE, resolve_provider_profile,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from ipfs_accelerate_py import llm_router


def _agent(tmp_path, provider_profile, arm):
    archive = tmp_path / "archive"
    archive.mkdir()
    manifest = _supervisor_manifest({"archive_sha256": "fixture-only-no-archive",
        "learned_requirements": [], "codex_version": CLI_VERSION})
    if provider_profile == GROK_PROFILE:
        manifest["grok_cli_assets"] = grok.grok_binding()
        manifest["files"].append({"path": grok.GROK_PATH, "bytes": grok.GROK_BYTES,
            "sha256": grok.GROK_SHA256, "mode": 0o755})
    (archive / "manifest.json").write_text(json.dumps(manifest))
    text = "Update the public module.py implementation."
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "input_paths": ["module.py"], "outputs": [
            {"path": "module.py", "effect": "modify", "media_type": "text/x-python"}]}
    return harbor.FullSupervisorAgent(logs_dir=tmp_path / "logs",
        model_name=resolve_provider_profile(provider_profile)["model"], runtime_archive=str(archive),
        arm=arm, auth_json=str(tmp_path / "unread-auth.json"),
        task_profile=profile, provider_profile=provider_profile)


def _installation(monkeypatch, agent, environment, events):
    async def task_bootstrap(observed_environment, **kwargs):
        assert observed_environment is environment
        assert kwargs == {"profile": agent.task_profile, "output": agent.logs_dir / "task-bootstrap"}
        assert events == []
        events.append("bootstrap")
        return {"authored_installation_double": True}

    async def deployment(observed_environment, **kwargs):
        assert observed_environment is environment
        assert events == ["bootstrap"]
        assert kwargs["auth_json"] == agent.auth_json
        assert not agent.auth_json.exists()
        assert kwargs["isolated_uv_bootstrap"] is True
        if agent.provider_profile == GROK_PROFILE:
            assert kwargs["provider"] == "grok_cli" and kwargs["install_codex"] is False
        else:
            assert "provider" not in kwargs and "install_codex" not in kwargs
        events.append("deployment")
        return {"authored_installation_double": True}

    async def boundary(observed_environment, **kwargs):
        assert observed_environment is environment
        assert events == ["bootstrap", "deployment"]
        assert kwargs["output"] == agent.logs_dir / "worker-boundary"
        assert kwargs.get("provider", "codex_cli") == resolve_provider_profile(agent.provider_profile)["provider"]
        events.append("boundary")
        return {"authored_installation_double": True}

    def no_model(*args, **kwargs):
        pytest.fail("Harbor setup must not dispatch a model")

    monkeypatch.setattr(bootstrap, "bootstrap_task_repository", task_bootstrap)
    monkeypatch.setattr(harbor, "deploy_supervisor", deployment)
    monkeypatch.setattr(harbor, "deploy_worker_boundary", boundary)
    monkeypatch.setattr(llm_router, "generate_text", no_model)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_codex_setup_does_not_probe_grok_credentials(tmp_path, monkeypatch, arm):
    agent = _agent(tmp_path, CODEX_PROFILE, arm)
    events = []

    class Environment:
        async def exec(self, **kwargs):
            pytest.fail("Codex setup must not invoke Grok readiness")

    environment = Environment()
    _installation(monkeypatch, agent, environment, events)
    asyncio.run(agent.setup(environment))
    assert events == ["bootstrap", "deployment", "boundary"]
    assert not (agent.logs_dir / "provider-readiness").exists()
    assert not agent.auth_json.exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_grok_negative_probe_refuses_setup_after_worker_boundary_without_model_calls(
        tmp_path, monkeypatch, arm):
    agent = _agent(tmp_path, GROK_PROFILE, arm)
    events = []

    class Environment:
        async def exec(self, **kwargs):
            assert events == ["bootstrap", "deployment", "boundary"]
            assert kwargs["user"] == "benchmarkworker"
            assert kwargs["cwd"] == "/" and kwargs["timeout_sec"] == 30
            assert kwargs["env"]["GROK_HOME"] == "/opt/ipfs-supervisor/worker-home/.grok"
            events.append("readiness")
            negative = grok._catalog_observation(
                b"You are not authenticated. PRIVATE_TRANSPORT_STDOUT\nDefault model: grok-4.7\n", b"", 0)
            return SimpleNamespace(return_code=0, stdout=json.dumps(negative), stderr="")

    environment = Environment()
    _installation(monkeypatch, agent, environment, events)
    with pytest.raises(grok.GrokCredentialReadinessError) as caught:
        asyncio.run(agent.setup(environment))
    assert events == ["bootstrap", "deployment", "boundary", "readiness"]
    receipt = json.loads((agent.logs_dir / "provider-readiness/readiness.json").read_bytes())
    assert receipt == caught.value.receipt
    assert receipt["status"] == "unauthenticated"
    assert receipt["reason"] == "authentication_refused"
    assert receipt["text_generation_calls"] == 0
    assert receipt["native_catalog_commands"] == 1
    assert "PRIVATE_TRANSPORT" not in json.dumps(receipt)
    assert "PRIVATE_TRANSPORT" not in str(caught.value)
    assert not agent.auth_json.exists()


def test_grok_setup_accepts_ready_catalog_only_after_isolated_boundary(tmp_path, monkeypatch):
    agent = _agent(tmp_path, GROK_PROFILE, "full")
    events = []

    class Environment:
        async def exec(self, **kwargs):
            assert events == ["bootstrap", "deployment", "boundary"]
            assert kwargs["user"] == "benchmarkworker"
            assert kwargs["env"]["HOME"] == "/opt/ipfs-supervisor/worker-home"
            events.append("readiness")
            ready = grok._catalog_observation(b"Available models:\n  grok-4.7\n", b"", 0)
            return SimpleNamespace(return_code=0, stdout=json.dumps(ready), stderr="")

    environment = Environment()
    _installation(monkeypatch, agent, environment, events)
    asyncio.run(agent.setup(environment))
    assert events == ["bootstrap", "deployment", "boundary", "readiness"]
    receipt = json.loads((agent.logs_dir / "provider-readiness/readiness.json").read_bytes())
    assert receipt["status"] == "ready"
    assert receipt["selected_model_observed"] is True
    assert receipt["text_generation_calls"] == 0
    assert all(receipt[name] is False for name in (
        "authentication_authority", "proof_authority", "execution_authority", "completion_authority"))
    assert not agent.auth_json.exists()
