"""Generated owner lifecycle with authored processes, without provider calls.

The production owner script executes directly. Real UID separation, the source
protocol and private-store permissions belong to the separate Docker probe.
"""
import signal
import subprocess
import sys
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY, OWNER_MONITOR
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from test.api.test_semantic_router_integration import allocated, native, provider  # noqa: F401
from test.api.test_semantic_router_translation import _reply


class AuthoredFailure(RuntimeError):
    pass


@pytest.mark.parametrize("failure", [None, "spawn", "source_register", "numerical_register",
    "wait", "cleanup", "guard_abort", "numerical_close", "source_close"])
def test_generated_owner_monitor_retires_source_and_numerical_services(monkeypatch, failure):
    events = []

    def event(name):
        events.append(name)
        if failure == name:
            raise AuthoredFailure(name)

    class Process:
        waits = 0

        def wait(self, **kwargs):
            self.waits += 1
            if self.waits == 1:
                event("wait")
            else:
                events.append("reap")
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    child = Process()

    def spawn(command):
        assert command == ["authored-worker"]
        event("spawn")
        return child

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup"
        event("cleanup")
        return SimpleNamespace(returncode=0)

    def service(name):
        def register(process):
            assert process is child
            event(name + "_register")
        return SimpleNamespace(note_spawned=register, close=lambda: event(name + "_close"))

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(subprocess, "run", cleanup)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    namespace = dict(command=["authored-worker"], worker_source_broker=service("source"),
        numerical_obligation_broker=service("numerical"),
        proof_query_guard=SimpleNamespace(note_spawned=lambda process: event("guard_register"),
                                        abort=lambda: event("guard_abort")))
    with pytest.raises(SystemExit if failure is None else AuthoredFailure) as raised:
        exec(compile(OWNER_MONITOR, "<installed-owner-monitor>", "exec"), namespace)
    if failure is None:
        assert raised.value.code == 0
    assert events.count("source_close") == events.count("numerical_close") == 1
    assert events.count("cleanup") == events.count("guard_abort") == 1
    if failure is None:
        assert events.index("source_register") < events.index("wait")
        assert events.index("numerical_register") < events.index("wait")
    if failure in {"source_register", "numerical_register", "wait"}:
        assert "terminate" in events
        assert events.index("terminate") < events.index("reap" if failure == "wait" else "wait")


@pytest.mark.parametrize("failure", ["source_factory", "source_extend", "source_extend_and_abort"])
def test_source_startup_failure_closes_previously_created_services(monkeypatch, failure):
    events = []

    class Command(list):
        def extend(self, extra):
            if extra[0] == "--source-view-descriptor":
                events.append("source_extend")
                raise AuthoredFailure("source_extend")
            events.append("numerical_extend")
            super().extend(extra)

    numerical = SimpleNamespace(extra_argv=["--numerical-obligation-descriptor", "/numerical"],
        close=lambda: events.append("numerical_close"))
    source = SimpleNamespace(extra_argv=["--source-view-descriptor", "/source"],
        close=lambda: events.append("source_close"))
    argv = ["--purpose", "coding", "--source-view-profile", "python-whitespace@1"]

    def source_factory(**kwargs):
        assert kwargs == dict(argv=argv, boundary=Path("/owner/container-boundary.json"),
                              workspace=Path("/work"), input_fd=0)
        events.append("source_factory")
        if failure == "source_factory":
            raise AuthoredFailure("source_factory")
        return source

    def numerical_factory(**kwargs):
        assert kwargs["argv"] == argv
        events.append("numerical_factory")
        return numerical

    def abort():
        events.append("guard_abort")
        if failure == "source_extend_and_abort":
            raise AuthoredFailure("guard_abort")

    monkeypatch.setitem(sys.modules,
        "ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker",
        SimpleNamespace(prepare_numerical_obligation_broker=numerical_factory))
    monkeypatch.setitem(sys.modules,
        "ipfs_accelerate_py.agent_supervisor.runtime.worker_source_broker",
        SimpleNamespace(prepare_worker_source_broker=source_factory))
    start = min(OWNER_ENTRY.index("numerical_obligation_broker=None"),
                OWNER_ENTRY.index("worker_source_broker=None"))
    stop = OWNER_ENTRY.index(OWNER_MONITOR)
    namespace = dict(root=Path("/owner"), cwd=Path("/work"), handoff={"argv": argv},
        command=Command(["worker"]), proof_query_guard=SimpleNamespace(abort=abort))
    with pytest.raises(AuthoredFailure):
        exec(compile(OWNER_ENTRY[start:stop], "<installed-owner-startup>", "exec"), namespace)
    assert events.count("guard_abort") == events.count("numerical_close") == 1
    assert events.count("source_close") == (0 if failure == "source_factory" else 1)


@pytest.fixture
def endpoint(allocated, monkeypatch, tmp_path):
    """Real semantic context; only isolated descriptor/UID admission is authored."""
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary

    repository, workspace, prompt, _ = allocated
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    profile = "python-whitespace@1"
    descriptor_path = tmp_path / "authored-source-descriptor.json"
    descriptor = dict(session_id="1" * 32, workspace=str(workspace),
        boundary_manifest_sha256="b" * 64, **client.semantic_binding(encoded, profile))
    checks = []

    def load(path, digest, *, workspace):
        assert path == descriptor_path and digest == "d" * 64 and workspace == allocated[1]
        checks.append(True)
        return deepcopy(descriptor)

    monkeypatch.setattr(client, "load_descriptor", load)
    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kwargs: {"authored_uid_boundary": True})
    kwargs = dict(prompt=prompt, semantic_repository=repository, provider="codex_cli", model="pinned",
        timeout=1, max_output_tokens=128, container_boundary=tmp_path / "boundary.json",
        container_boundary_sha256="b" * 64, source_view_profile=profile,
        source_view_descriptor=descriptor_path, source_view_descriptor_sha256="d" * 64)
    return encoded, descriptor, checks, kwargs


@pytest.mark.parametrize("failed", [False, True])
def test_current_semantic_source_route_reaches_actual_child_environment_and_restores_it(endpoint, provider, monkeypatch, failed):
    encoded, descriptor, checks, kwargs = endpoint
    state, observed = provider
    state["reply"] = json.dumps(_reply(encoded)[0])
    keys = (runner.SOURCE_DESCRIPTOR_ENV, runner.SOURCE_DESCRIPTOR_DIGEST_ENV)
    monkeypatch.setenv(keys[0], "previous-owner-descriptor")
    monkeypatch.delenv(keys[1], raising=False)
    child_environments = []

    def during_call():
        code = "import json,os;print(json.dumps({key:os.environ.get(key) for key in " + repr(keys) + "}))"
        child_environments.append(json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], text=True)))
        if failed:
            raise AuthoredFailure("provider failed")

    state["after_generate"] = during_call
    if failed:
        with pytest.raises(AuthoredFailure, match="provider failed"):
            runner.run(**kwargs)
    else:
        _, receipt = runner.run(**kwargs)
        observation = receipt["source_view_retrieval"]
        assert all(observation[key] == descriptor[key] for key in client.BINDING_FIELDS)
        advice = client.guidance(kwargs["source_view_profile"]).encode()
        assert observation["guidance_bytes"] == len(advice)
        assert observation["guidance_sha256"] == hashlib.sha256(advice).hexdigest()
        assert observation["requests_observed"] is observation["proof_authority"] is False
        assert observation["execution_authority"] is observation["completion_authority"] is False
        assert receipt["model_prompt_bytes"] == len(observed[0][0].encode())
        assert "source_formalization_context" not in receipt  # Source tooling does not require numerical/learned advice.
    assert child_environments == [{keys[0]: str(kwargs["source_view_descriptor"]), keys[1]: "d" * 64}]
    assert len(observed) == 1 and checks == [True, True]
    assert client.guidance(kwargs["source_view_profile"]) in observed[0][0]
    assert str(kwargs["source_view_descriptor"]) not in observed[0][0]
    assert os.environ[keys[0]] == "previous-owner-descriptor" and keys[1] not in os.environ


@pytest.mark.parametrize("field", [*client.BINDING_FIELDS, "boundary_manifest_sha256"])
def test_foreign_source_descriptor_never_reaches_provider(endpoint, provider, field):
    _, descriptor, checks, kwargs = endpoint
    _, observed = provider
    descriptor[field] = "foreign"
    with pytest.raises(ValueError, match="differs from validated semantic evidence"):
        runner.run(**kwargs)
    assert checks == [True] and observed == []


def test_source_descriptor_is_rechecked_after_provider_initialization(endpoint, provider, monkeypatch):
    from ipfs_accelerate_py import llm_router

    _, descriptor, checks, kwargs = endpoint
    _, observed = provider

    def initialized(*args, **kwargs):
        descriptor["session_id"] = "2" * 32
        return object()

    monkeypatch.setattr(llm_router, "get_llm_provider", initialized)
    with pytest.raises(ValueError, match="changed before provider dispatch"):
        runner.run(**kwargs)
    assert observed == [] and checks == [True, True]
    assert runner.SOURCE_DESCRIPTOR_ENV not in os.environ
    assert runner.SOURCE_DESCRIPTOR_DIGEST_ENV not in os.environ


@pytest.mark.parametrize("key", [runner.SOURCE_DESCRIPTOR_ENV, runner.SOURCE_DESCRIPTOR_DIGEST_ENV])
def test_ambient_source_endpoint_without_explicit_owner_selection_is_rejected(monkeypatch, key):
    monkeypatch.setenv(key, "unselected")
    with pytest.raises(ValueError, match="environment requires explicit owner selection"):
        runner.run(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1)


@pytest.mark.parametrize("missing", ["source_view_descriptor", "source_view_descriptor_sha256",
    "source_view_profile", "semantic_repository", "container_boundary", "container_boundary_sha256"])
def test_partial_source_route_refuses_before_boundary_or_provider(tmp_path, missing):
    kwargs = dict(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1,
        semantic_repository=tmp_path, container_boundary=tmp_path / "boundary.json",
        container_boundary_sha256="b" * 64, source_view_profile="python-whitespace@1",
        source_view_descriptor=tmp_path / "descriptor.json", source_view_descriptor_sha256="d" * 64)
    del kwargs[missing]
    with pytest.raises(ValueError, match="exact source descriptor"):
        runner.run(**kwargs)


def test_default_disabled_route_preserves_existing_complete_input(allocated, provider, monkeypatch):
    repository, workspace, prompt, _ = allocated
    state, observed = provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    state["reply"] = json.dumps(_reply(encoded)[0])
    monkeypatch.setattr(client, "load_descriptor", lambda *a, **k: pytest.fail("disabled source endpoint opened"))
    monkeypatch.setattr(client, "guidance", lambda *a, **k: pytest.fail("disabled source guidance rendered"))
    _, receipt = runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, semantic_repository=repository)
    expected, _ = runner.render_model_prompt(prompt=encoded.provider_prompt,
        purpose="coding", workspace=workspace, semantic_transport=True)
    assert len(observed) == 1 and observed[0][0] == expected
    assert "source_view_retrieval" not in receipt
    assert runner.SOURCE_DESCRIPTOR_ENV not in os.environ and runner.SOURCE_DESCRIPTOR_DIGEST_ENV not in os.environ
