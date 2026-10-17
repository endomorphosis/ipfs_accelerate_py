"""Execute generated owner wiring with explicitly authored process custody.

These tests execute the installed Python snippets, with inert child, service,
boundary and cleanup owners. They admit no native resources, change no UID,
launch no container/provider, and do not qualify automatic Terminal wiring.
The retained-cleanup loop and real native leases have separate qualifications.
"""
from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_owner_guard import (
    REFUSAL, owner_prefix, worker_parser,
)


DISPATCH = "ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_dispatch"
MONITOR = "ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_monitor"
NUMERICAL = "ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker"
SOURCE = "ipfs_accelerate_py.agent_supervisor.runtime.worker_source_broker"
PUBLIC_ARGUMENTS = ["--repository-source-owner-descriptor", "/owner/worker-observations/public.json",
                    "--repository-source-owner-descriptor-sha256", "d" * 64]


class AuthoredFailure(RuntimeError):
    pass


def _execute(source, namespace, modules):
    """Only the literal generated imports may reach these authored owners."""
    def imports(name, _globals=None, _locals=None, fromlist=(), level=0):
        assert level == 0 and name in modules, "unexpected generated import: " + name
        return modules[name]

    namespace["__builtins__"] = {**vars(builtins), "__import__": imports}
    exec(compile(source, "<installed-repository-source-owner>", "exec"), namespace)


@pytest.mark.parametrize("with_input", [False, True])
@pytest.mark.parametrize("failure", [None, "spawn", "numerical_register", "source_register",
    "repository_register", "guard_register", "settle", "cleanup", "guard_abort",
    "numerical_close", "source_close"])
def test_generated_monitor_retires_exact_source_owner_after_every_cleanup_path(with_input, failure):
    events, retired = [], []

    def event(name):
        events.append(name)
        if failure == name:
            raise AuthoredFailure(name)

    class Child:
        returncode = None
        terminated = False

        def wait(self, **kwargs):
            if kwargs or self.terminated:
                events.append("reap")
            else:
                event("settle")
            self.returncode = 0
            return 0

        def communicate(self, *, input):
            assert input == b"authored exact owner input"
            event("settle")
            self.returncode = 0
            return None, None

        def terminate(self):
            self.terminated = True
            events.append("terminate")

        def kill(self):
            events.append("kill")

    child = Child()

    def spawn(command, **kwargs):
        assert command == ["authored-worker"]
        assert kwargs == ({"stdin": subprocess.PIPE} if with_input else {})
        event("spawn")
        return child

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup"
        assert kwargs == dict(cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              start_new_session=True, timeout=15)
        event("cleanup")
        return SimpleNamespace(returncode=0)

    def service(name):
        def registered(process):
            assert process is child
            event(name + "_register")
        return SimpleNamespace(note_spawned=registered, close=lambda: event(name + "_close"))

    source_owner = service("repository")

    def retire(owner):
        assert owner is source_owner
        assert events[-1] == "source_close"
        retired.append(owner)
        events.append("repository_retired")
        return {"cleanup_complete": True}

    namespace = dict(command=["authored-worker"], repository_source_owner=source_owner,
        worker_source_broker=service("source"), numerical_obligation_broker=service("numerical"),
        proof_query_guard=SimpleNamespace(note_spawned=lambda process: event("guard_register"),
                                        abort=lambda: event("guard_abort")))
    if with_input:
        namespace["owner_prompt_bytes"] = b"authored exact owner input"
    modules = {"json": json, "time": SimpleNamespace(),
        "signal": SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT,
            SIGHUP=signal.SIGHUP, SIG_IGN=signal.SIG_IGN, signal=lambda *args: None),
        "subprocess": SimpleNamespace(Popen=spawn, run=cleanup, PIPE=subprocess.PIPE,
            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired),
        MONITOR: SimpleNamespace(retire_repository_source_owner=retire)}
    with pytest.raises(SystemExit if failure is None else AuthoredFailure) as error:
        _execute(deployment.OWNER_MONITOR, namespace, modules)
    if failure is None:
        assert error.value.code == 0
        assert events.index("repository_register") < events.index("settle")
    assert retired == [source_owner]
    assert events[-1] == "repository_retired"
    assert "repository_close" not in events  # The generated caller must use the retaining owner helper.
    assert all(events.count(name) == 1 for name in (
        "cleanup", "guard_abort", "numerical_close", "source_close"))
    assert events.index("cleanup") < events.index("guard_abort") < events.index("numerical_close")
    assert events.index("numerical_close") < events.index("source_close") < events.index("repository_retired")
    if failure in {"numerical_register", "source_register", "repository_register", "guard_register", "settle"}:
        assert events.index("terminate") < events.index("reap") < events.index("cleanup")
    if failure == "spawn":
        assert "repository_register" not in events and "terminate" not in events


@pytest.mark.parametrize("binding", ["missing", "none"])
def test_legacy_generated_monitor_keeps_original_process_and_cleanup_events(binding):
    events = []

    def record(name, value=None):
        events.append(name)
        return value

    child = SimpleNamespace(wait=lambda: record("wait", 0))
    namespace = {"command": ["authored-worker"]}
    if binding == "none":
        namespace["repository_source_owner"] = None

    def retire(owner):
        assert owner is None
        return None

    modules = {"json": json, "time": SimpleNamespace(),
        "signal": SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT,
            SIGHUP=signal.SIGHUP, SIG_IGN=signal.SIG_IGN, signal=lambda *args: None),
        "subprocess": SimpleNamespace(Popen=lambda argv: record("spawn", child),
            run=lambda *args, **kwargs: record("cleanup", SimpleNamespace(returncode=0)),
            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired),
        MONITOR: SimpleNamespace(retire_repository_source_owner=retire)}
    with pytest.raises(SystemExit) as error:
        _execute(deployment.OWNER_MONITOR, namespace, modules)
    assert error.value.code == 0 and events == ["spawn", "wait", "cleanup"]


def _startup_source():
    # The full installed entry still refuses nonlegacy policy earlier. Select
    # its actual service-construction block to qualify only future wiring.
    begin = deployment.OWNER_ENTRY.index("numerical_obligation_broker=None")
    return deployment.OWNER_ENTRY[begin:deployment.OWNER_ENTRY.index(deployment.OWNER_MONITOR)]


@pytest.mark.parametrize("failure,close_failure", [
    (None, None), ("configure", None), ("start", None), ("public_arguments", None),
    ("extend", None), ("start", "guard_abort"), ("start", "numerical_close"),
    ("start", "source_close"), ("public_arguments", "source_close"),
    ("start_sigterm", None), ("start_sigint", None), ("start_sighup", None),
])
def test_generated_startup_retains_owner_before_start_and_retires_on_failure(failure, close_failure):
    events, retired, forwarded = [], [], []
    signal_handlers, signal_calls = {}, []
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    argv = ["--purpose", "coding", "--semantic-repository", "/repository"]
    namespace = dict(root=Path("/owner"), cwd=Path("/work"), handoff={"argv": argv},
                     owner_candidate_round_deadline=120.5)

    def close(name):
        events.append(name)
        if failure != "configure":
            assert all(signal_handlers[number] is signal.SIG_IGN for number in signals)
        if close_failure == name:
            raise AuthoredFailure(name)

    def set_signal(number, handler):
        assert number in signals
        signal_handlers[number] = handler
        signal_calls.append((number, handler))

    class Owner:
        def start(self):
            assert namespace["repository_source_owner"] is self
            assert all(callable(signal_handlers[number]) for number in signals)
            assert len({id(signal_handlers[number]) for number in signals}) == 1
            events.append("start")
            if failure in {"start_sigterm", "start_sigint", "start_sighup"}:
                number = {"start_sigterm": signal.SIGTERM, "start_sigint": signal.SIGINT,
                          "start_sighup": signal.SIGHUP}[failure]
                signal_handlers[number](number, None)
            if failure == "start":
                raise AuthoredFailure("start")
            return self

        @property
        def extra_argv(self):
            events.append("public_arguments")
            if failure == "public_arguments":
                raise AuthoredFailure("public_arguments")
            return list(PUBLIC_ARGUMENTS)

    owner = Owner()

    class Command(list):
        def extend(self, extra):
            if extra == PUBLIC_ARGUMENTS:
                events.append("extend")
                if failure == "extend":
                    raise AuthoredFailure("extend")
                forwarded.extend(extra)
            super().extend(extra)

    def prepare(name):
        def factory(**kwargs):
            assert kwargs == dict(argv=argv, boundary=Path("/owner/container-boundary.json"),
                                  workspace=Path("/work"), input_fd=0)
            events.append(name + "_prepare")
            return SimpleNamespace(extra_argv=[], close=lambda: close(name + "_close"))
        return factory

    def configure(**kwargs):
        assert namespace["repository_source_owner"] is None
        assert kwargs == dict(argv=argv, boundary=Path("/owner/container-boundary.json"),
                              workspace=Path("/work"), deadline_monotonic=120.5)
        events.append("configure")
        if failure == "configure":
            raise AuthoredFailure("configure")
        return owner

    def retire(value):
        assert value is owner
        assert namespace["repository_source_owner"] is owner
        assert all(signal_handlers[number] is signal.SIG_IGN for number in signals)
        retired.append(value)
        events.append("retire")
        return {"cleanup_complete": True}

    namespace.update(command=Command(["worker"]),
        proof_query_guard=SimpleNamespace(abort=lambda: close("guard_abort")))
    modules = {NUMERICAL: SimpleNamespace(prepare_numerical_obligation_broker=prepare("numerical")),
        SOURCE: SimpleNamespace(prepare_worker_source_broker=prepare("source")),
        DISPATCH: SimpleNamespace(configure_repository_source_owner_dispatch=configure),
        MONITOR: SimpleNamespace(retire_repository_source_owner=retire),
        "signal": SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT,
            SIGHUP=signal.SIGHUP, SIG_IGN=signal.SIG_IGN, signal=set_signal)}
    if failure is None:
        _execute(_startup_source(), namespace, modules)
        assert events == ["numerical_prepare", "source_prepare", "configure", "start", "public_arguments", "extend"]
        assert forwarded == PUBLIC_ARGUMENTS and namespace["command"] == ["worker", *PUBLIC_ARGUMENTS]
        assert retired == [] and namespace["repository_source_owner"] is owner
        assert [number for number, _handler in signal_calls] == list(signals)
    else:
        interrupted = failure in {"start_sigterm", "start_sigint", "start_sighup"}
        with pytest.raises(InterruptedError if interrupted else AuthoredFailure) as error:
            _execute(_startup_source(), namespace, modules)
        if interrupted:
            assert str(error.value) == "repository source owner interrupted"
        assert all(events.count(name) == 1 for name in ("guard_abort", "numerical_close", "source_close"))
        assert events.index("guard_abort") < events.index("numerical_close") < events.index("source_close")
        assert retired == ([] if failure == "configure" else [owner])
        if failure != "configure":
            assert events[-1] == "retire"
            assert [number for number, _handler in signal_calls] == list(signals) * 2
        else:
            assert signal_calls == []
        assert forwarded == []


@pytest.mark.parametrize("mode", ["repository-source-raw@1", "repository-source-compact@1"])
def test_owner_lifecycle_wiring_does_not_enable_nonlegacy_deployment(mode):
    arguments = ["--semantic-repository", "/app", "--repository-source-policy", mode]
    code, events = owner_prefix(arguments)
    assert code == REFUSAL and "candidate_access" not in events
    assert worker_parser(arguments)[0] == REFUSAL


def test_generated_legacy_startup_can_leave_source_owner_unselected():
    selected = []
    argv = ["--purpose", "coding"]

    def configure(**kwargs):
        selected.append(kwargs)
        return None

    namespace = dict(root=Path("/owner"), cwd=Path("/work"), handoff={"argv": argv},
        command=["worker"], proof_query_guard=None, owner_candidate_round_deadline=None)
    modules = {NUMERICAL: SimpleNamespace(prepare_numerical_obligation_broker=lambda **kwargs: None),
        SOURCE: SimpleNamespace(prepare_worker_source_broker=lambda **kwargs: None),
        DISPATCH: SimpleNamespace(configure_repository_source_owner_dispatch=configure)}
    _execute(_startup_source(), namespace, modules)
    assert namespace["repository_source_owner"] is None and namespace["command"] == ["worker"]
    assert selected == [dict(argv=argv, boundary=Path("/owner/container-boundary.json"),
                             workspace=Path("/work"), deadline_monotonic=None)]


@pytest.mark.parametrize("resources", [True, {}, {"unknown": "authored"}])
def test_invalid_installer_resource_binding_refuses_before_io(tmp_path, resources):
    class NoEnvironment:
        def __getattribute__(self, name):
            pytest.fail("invalid resource binding reached environment IO: " + name)

    output = tmp_path / "installation"
    with pytest.raises(ValueError):
        asyncio.run(deployment.install_worker_boundary(NoEnvironment(), container_name="authored",
            output=output, repository_source_resources=resources))
    assert not output.exists()


@pytest.mark.parametrize("resources", [True, {}, {"unknown": "authored"}])
def test_invalid_deployment_resource_binding_refuses_before_io(tmp_path, resources):
    class NoEnvironment:
        def __getattribute__(self, name):
            pytest.fail("invalid resource binding reached deployment IO: " + name)

    output = tmp_path / "deployment"
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_worker_boundary(NoEnvironment(), output=output,
            repository_source_resources=resources))
    assert not output.exists()


@pytest.mark.parametrize("entrypoint", ["install", "deploy"])
@pytest.mark.parametrize("provider", ["unknown", "grok_cli"])
def test_unsupported_resource_provider_refuses_before_installer_or_deployment_io(tmp_path, entrypoint, provider):
    class NoEnvironment:
        def __getattribute__(self, name):
            pytest.fail("unsupported resource provider reached environment IO: " + name)

    resources = dict(schema="repository-source-owner-resources@1",
        scheduler_state_path="/opt/ipfs-supervisor/state/source-scheduler.json",
        scheduler_policy_sha256="a" * 64, admission_timeout_seconds=10,
        resource_scope="repository_source_service_only")
    output = tmp_path / entrypoint
    operation = deployment.install_worker_boundary if entrypoint == "install" else deployment.deploy_worker_boundary
    kwargs = dict(output=output, provider=provider, repository_source_resources=resources)
    if entrypoint == "install":
        kwargs["container_name"] = "authored"
    with pytest.raises(ValueError):
        asyncio.run(operation(NoEnvironment(), **kwargs))
    assert not output.exists()


@pytest.mark.parametrize("selected", [False, True])
def test_actual_installer_serializes_only_selected_detached_resource_configuration(tmp_path, monkeypatch, selected):
    """Container operations are recorded strings; local manifest bytes are real."""
    resources = dict(schema="repository-source-owner-resources@1",
        scheduler_state_path="/opt/ipfs-supervisor/state/source-scheduler.json",
        scheduler_policy_sha256="a" * 64, admission_timeout_seconds=10,
        resource_scope="repository_source_service_only") if selected else None
    expected = None if resources is None else dict(resources)
    scheduler_path = "/opt/ipfs-supervisor/state/source-scheduler.json"
    commands, uploads, inspections, namespace_reads = [], {}, [], []
    namespaces = {name: name + ":[authored]" for name in ("pid", "mnt", "net")}
    cid, image, pid = "1" * 64, "sha256:" + "2" * 64, 54321
    original_open, original_path_open, original_os_open = builtins.open, Path.open, os.open

    def reject_scheduler_path(path):
        if isinstance(path, (str, bytes, Path)):
            name = os.fsdecode(path)
            assert name != scheduler_path, "installer accessed native scheduler contents or created state"

    def opened(path, *args, **kwargs):
        reject_scheduler_path(path)
        return original_open(path, *args, **kwargs)

    def path_opened(path, *args, **kwargs):
        reject_scheduler_path(path)
        return original_path_open(path, *args, **kwargs)

    def os_opened(path, *args, **kwargs):
        reject_scheduler_path(path)
        return original_os_open(path, *args, **kwargs)

    def inspect(argv, *, text):
        assert text is True
        assert argv == ["docker", "inspect", "--format",
            "{{json .Id}} {{json .Image}} {{json .State.Pid}} {{json .State.Running}} {{json .HostConfig.Privileged}}",
            "authored-existing-task"]
        inspections.append(argv)
        return " ".join(json.dumps(value) for value in (cid, image, pid, True, False))

    def readlink(path):
        prefix = "/proc/" + str(pid) + "/ns/"
        assert str(path).startswith(prefix)
        name = str(path)[len(prefix):]
        assert name in namespaces
        namespace_reads.append(name)
        return namespaces[name]

    class Environment:
        async def exec(self, **kwargs):
            assert kwargs["cwd"] == "/app" and kwargs["user"] == "root"
            assert kwargs["env"] is None and kwargs["timeout_sec"] == 180
            command = kwargs["command"]
            assert scheduler_path not in command
            commands.append(command)
            return SimpleNamespace(return_code=0, stderr="", stdout=(
                json.dumps(namespaces) if "/proc/self/ns/" in command else ""))

        async def upload_file(self, source, target):
            assert target not in uploads
            uploads[target] = Path(source).read_bytes()

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(Path, "open", path_opened)
    monkeypatch.setattr(os, "open", os_opened)
    monkeypatch.setattr(os, "readlink", readlink)
    monkeypatch.setattr(subprocess, "check_output", inspect)
    # Only a command text is requested by the installer. No native CLI runs.
    monkeypatch.setattr(deployment, "native_codex_exposure_script", lambda: "authored_native_cli_exposure")
    output = tmp_path / "installation"
    result = asyncio.run(deployment.install_worker_boundary(Environment(),
        container_name="authored-existing-task", output=output, repository_source_resources=resources))
    target = deployment.ROOT + "/container-boundary.json"
    raw = uploads[target]
    manifest = json.loads(raw)
    assert result["boundary"] == manifest
    assert result["boundary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert raw == (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    assert json.loads((output / "installation.json").read_bytes()) == result
    assert result["provider_calls"] == 0 and result["qualified"] is False
    assert manifest["container_id"] == cid and manifest["image_id"] == image
    assert manifest["namespaces"] == namespaces
    assert len(inspections) == 1 and namespace_reads == ["pid", "mnt", "net"]
    assert commands.count("chown root:root " + target + " && chmod 0444 " + target) == 1
    assert uploads[deployment.ROOT + "/bin/router-worker"] == deployment.OWNER_ENTRY.encode()
    assert uploads[deployment.ROOT + "/bin/worker-entry"] == deployment.WORKER_ENTRY.encode()
    if selected:
        assert manifest["repository_source_resources"] == expected
        assert result["boundary"]["repository_source_resources"] is not resources
        resources["scheduler_policy_sha256"] = "b" * 64
        assert result["boundary"]["repository_source_resources"] == expected
        assert json.loads(raw)["repository_source_resources"] == expected
    else:
        assert "repository_source_resources" not in manifest


@pytest.mark.parametrize("selected", [False, True])
def test_actual_deployment_forwards_optional_detached_configuration(tmp_path, monkeypatch, selected):
    """Only the existing resolve/install/qualify interfaces are authored."""
    resources = dict(schema="repository-source-owner-resources@1",
        scheduler_state_path="/opt/ipfs-supervisor/state/source-scheduler.json",
        scheduler_policy_sha256="a" * 64, admission_timeout_seconds=10,
        resource_scope="repository_source_service_only") if selected else None
    expected = None if resources is None else dict(resources)
    events, installation_kwargs = [], []

    class Platform:
        async def _resolve_service_container(self, name):
            assert name == "main"
            events.append("resolve")
            return "authored-existing-task"

    environment = SimpleNamespace(_platform=Platform())

    async def install(actual, **kwargs):
        assert actual is environment
        events.append("install")
        installation_kwargs.append(kwargs)
        if selected:
            assert kwargs["repository_source_resources"] == expected
            assert kwargs["repository_source_resources"] is not resources
        else:
            assert "repository_source_resources" not in kwargs
        return {"schema": "authored-installation", "qualified": False}

    async def qualify(actual, *, output):
        assert actual is environment and output == destination / "qualification"
        events.append("qualify")
        return {"qualified": True, "authored_boundary_only": True}

    monkeypatch.setattr(deployment, "install_worker_boundary", install)
    monkeypatch.setattr(deployment, "qualify_worker_boundary", qualify)
    destination = tmp_path / "deployment"
    result = asyncio.run(deployment.deploy_worker_boundary(environment,
        output=destination, repository_source_resources=resources))
    assert events == ["resolve", "install", "qualify"]
    assert installation_kwargs[0]["container_name"] == "authored-existing-task"
    assert installation_kwargs[0]["output"] == destination / "installation"
    assert installation_kwargs[0]["provider"] == "codex_cli"
    assert result["qualification"]["authored_boundary_only"] is True
    assert json.loads((destination / "deployment.json").read_bytes()) == result
