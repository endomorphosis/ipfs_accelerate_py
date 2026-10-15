"""Signed owner routing and actual monitor stdin replay, without providers.

The entry test executes only production argument/staging statements; identity
and container qualification are outside scope. Monitor checks launch authored
local Python children and replace only the fixed sudo cleanup command.
"""
from __future__ import annotations

import ast
import builtins
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create

NOMINATION = {"semantic_decision_nomination_artifact": Path("/owner/explicit nomination.json"),
              "semantic_decision_nomination_sha256": "a" * 64}
FLAGS = ["--semantic-decision-nomination-artifact", str(NOMINATION["semantic_decision_nomination_artifact"]),
         "--semantic-decision-nomination-sha256", "a" * 64,
         "--semantic-decision-max-calls", "3"]
OWNER_FLAGS = FLAGS[:4] + ["--semantic-decision-typed-nomination", "enabled"] + FLAGS[4:]


def test_nomination_command_requires_independent_task_key_and_preserves_default():
    base = dict(router=Path("/router"), model="pinned", reasoning="high", timeout=30,
                semantic_repository=Path("/app"), doctor=None)
    assert deployment.semantic_decision_arguments() == []
    assert deployment.semantic_decision_arguments(**NOMINATION) == FLAGS
    assert deployment.semantic_decision_arguments(**NOMINATION, owner_binding=True) == OWNER_FLAGS
    default = doctor.implementation_argv(**base)
    selected = doctor.implementation_argv(**base, **NOMINATION, semantic_decision_task_id="admitted-task")
    assert selected == default + OWNER_FLAGS + ["--semantic-decision-task-id", "admitted-task"]
    disabled = doctor.implementation_argv(**base, **NOMINATION, semantic_decision_task_id="admitted-task",
                                         disable_semantic_typed_nomination=True)
    assert disabled == default + FLAGS[:4] + ["--semantic-decision-typed-nomination", "disabled"] + FLAGS[4:] + [
        "--semantic-decision-task-id", "admitted-task"]
    with pytest.raises(ValueError, match="independently admitted task"):
        doctor.implementation_argv(**base, **NOMINATION)
    with pytest.raises(ValueError, match="requires an owner nomination"):
        doctor.implementation_argv(**base, semantic_decision_task_id="unbound")


@pytest.mark.parametrize("extra", [
    {"semantic_decision_policy_artifact": Path("/owner/policy.json")},
    {"semantic_decision_policy_sha256": "b" * 64},
    {"semantic_decision_nomination_sha256": "invalid"},
])
def test_nomination_or_policy_bindings_refuse_ambiguous_inputs(extra):
    with pytest.raises(ValueError):
        deployment.semantic_decision_arguments(**(NOMINATION | extra))


def test_actual_signed_supervisor_binds_task_key_and_expected_nomination(supervisor):
    state, planned, signed = supervisor
    result = driver.run(instruction=Path("/instruction"), state=state, arm="full",
                        doctor_dispatch_policy="model-only", **NOMINATION)
    assert result["task_completed"] is True
    assert result["semantic_decision_dialogue_expected"] is True
    assert result["semantic_decision_owner_binding_expected"] is True
    assert result["semantic_decision_expected_task_id"] == "TASK"
    assert result["semantic_decision_nomination_sha256"] == "a" * 64
    assert len(planned) == len(signed) == 1
    assert not any(arg.startswith("--semantic-decision") for arg in planned[0])
    start = signed[0].index("--semantic-decision-nomination-artifact")
    assert signed[0][start:start + len(OWNER_FLAGS)] == OWNER_FLAGS
    assert signed[0][signed[0].index("--semantic-decision-task-id") + 1] == "TASK"


def _sealed_prompt(payload):
    """Actual immutable fd for the current inherited-stdin monitor contract."""
    assert type(payload) is bytes and len(payload) <= 256001
    fd = _validation_memfd_create("authored-owner-prompt",
        getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002))
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        os.lseek(fd, 0, os.SEEK_SET)
        seals = (getattr(fcntl, "F_SEAL_SEAL", 0x0001) | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
                 | getattr(fcntl, "F_SEAL_GROW", 0x0004) | getattr(fcntl, "F_SEAL_WRITE", 0x0008))
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
        assert fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) & seals == seals
        return fd
    except BaseException:
        os.close(fd)
        raise


@pytest.fixture
def owner_entry(monkeypatch, capsys):
    parsed = ast.parse(deployment.OWNER_ENTRY)
    assignments = {}
    for index, node in enumerate(parsed.body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            # The current entry rebinds worker_arguments after sealed capture.
            # This authored slice starts with its original literal argv input.
            assignments.setdefault(node.targets[0].id, index)
    code = compile(ast.Module(body=parsed.body[assignments["worker_arguments"]:
                               assignments["proof_query_guard"]], type_ignores=[]),
                   "<owner-staging-and-routing>", "exec")
    calls = []

    def execute(arguments, payload, *, refuse=False):
        calls.clear()
        def prepare(**kwargs):
            calls.append(kwargs)
            if refuse:
                raise ValueError("authored policy refusal")
            return SimpleNamespace(worker_arguments=("--semantic-decision-policy-artifact", "/canonical/policy.json"),
                                   prompt_bytes=kwargs["prompt_bytes"],
                                   receipt={"schema": "supervisor-semantic-decision-owner-binding@1"})
        def handoff(**kwargs):
            assert kwargs == {"argv": arguments, "boundary": Path("/opt/ipfs-supervisor/container-boundary.json"),
                              "workspace": Path("/allocated")}
            return {"argv": list(kwargs["argv"]), "stdin_fd": _sealed_prompt(payload)}
        def imported(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context":
                assert fromlist == ("prepare_container_source_formalization_dispatch",)
                return SimpleNamespace(prepare_container_source_formalization_dispatch=handoff)
            assert name == "ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner"
            return SimpleNamespace(prepare_decision_owner_dispatch=prepare)
        monkeypatch.setattr(sys, "argv", ["router-worker", *arguments])
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
        monkeypatch.setattr(sys, "path", list(sys.path))
        namespace = {"__builtins__": {**vars(builtins), "__import__": imported},
                     "sys": sys, "os": os, "json": json, "root": Path("/opt/ipfs-supervisor"),
                     "cwd": Path("/allocated")}
        original_stdin = os.dup(0)
        try:
            if refuse:
                with pytest.raises(SystemExit) as error:
                    exec(code, namespace)
                assert error.value.code == 1
            else:
                exec(code, namespace)
            assert os.pread(0, len(payload) + 1, 0) == payload
        finally:
            os.dup2(original_stdin, 0)
            os.close(original_stdin)
        return namespace, list(calls), capsys.readouterr().out
    return execute


@pytest.mark.parametrize("arguments", [[], ["--semantic-decision-policy-artifact", "/owner/prebound.json"]])
def test_default_and_prebound_owner_routes_keep_inherited_stdin(owner_entry, arguments):
    namespace, calls, output = owner_entry(arguments, b"unchanged inherited input")
    assert namespace["owner_prompt_bytes"] is None and calls == [] and output == ""
    assert namespace["command"] == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--",
                                     "/opt/ipfs-supervisor/bin/worker-entry", *arguments]


def test_owner_entry_stages_before_worker_and_retains_exact_native_bytes(owner_entry):
    payload = "{\"native\":\"λ\"}\r\nrequired literal suffix\r\n".encode()
    namespace, calls, output = owner_entry(FLAGS + ["--semantic-decision-task-id", "TASK"], payload)
    assert calls == [{"arguments": FLAGS + ["--semantic-decision-task-id", "TASK"],
                      "prompt_bytes": payload, "workspace": Path("/allocated")}]
    assert namespace["owner_prompt_bytes"] == payload
    assert namespace["command"][-2:] == ["--semantic-decision-policy-artifact", "/canonical/policy.json"]
    assert json.loads(output)["schema"] == "supervisor-semantic-decision-owner-binding@1"


def test_owner_entry_refuses_before_constructing_worker_command(owner_entry):
    namespace, calls, output = owner_entry(FLAGS, b"native", refuse=True)
    assert len(calls) == 1 and "command" not in namespace
    assert json.loads(output)["provider_called"] is False


def _monitor(tmp_path, *, child_code=0, cleanup_code=0, payload=None, wait=False):
    target = tmp_path / "delivered.bin"
    cleanup = tmp_path / "cleanup.json"
    popen_options = tmp_path / "popen.json"
    child = tmp_path / "child.py"
    child.write_text("import os,pathlib,sys,time\n"
        + ("pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())\n" if payload is not None else "")
        + ("print(os.getpid(),flush=True)\ntime.sleep(30)\n" if wait else "")
        + f"raise SystemExit({child_code})\n")
    harness = tmp_path / "monitor.py"
    harness.write_text("import json,pathlib,subprocess,sys,types\n"
        "original_popen=subprocess.Popen\n"
        "def observed_popen(command,**kwargs):\n"
        f" pathlib.Path({str(popen_options)!r}).write_text(json.dumps(kwargs))\n"
        " return original_popen(command,**kwargs)\n"
        "subprocess.Popen=observed_popen\n"
        "def cleanup(command,**kwargs):\n"
        " assert command==['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--cleanup']\n"
        f" pathlib.Path({str(cleanup)!r}).write_text(json.dumps(command))\n"
        f" return types.SimpleNamespace(returncode={cleanup_code})\n"
        "subprocess.run=cleanup\n"
        f"command=[sys.executable,{str(child)!r},{str(target)!r}]\n"
        + f"exec({deployment.OWNER_MONITOR!r})\n")
    inherited_stdin = _sealed_prompt(payload) if payload is not None else None
    try:
        process = subprocess.Popen([sys.executable, "-I", str(harness)], stdin=inherited_stdin,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if inherited_stdin is not None:
            os.close(inherited_stdin)
    return process, target, cleanup, popen_options


@pytest.mark.parametrize("child_code,cleanup_code,expected", [(0, 0, 0), (7, 0, 7), (0, 1, 1)])
def test_actual_monitor_inherits_sealed_bytes_and_preserves_exit_and_cleanup(tmp_path, child_code, cleanup_code, expected):
    payload = "λ\r\nverbatim required suffix\r\n".encode()
    process, target, cleanup, options = _monitor(tmp_path, child_code=child_code,
                                               cleanup_code=cleanup_code, payload=payload)
    process.communicate(timeout=10)
    assert process.returncode == expected and target.read_bytes() == payload
    assert cleanup.is_file() and "stdin" not in json.loads(options.read_text())


def test_actual_default_monitor_does_not_replace_stdin(tmp_path):
    process, _, cleanup, options = _monitor(tmp_path)
    process.communicate(timeout=10)
    assert process.returncode == 0 and cleanup.is_file()
    assert "stdin" not in json.loads(options.read_text())


def test_actual_monitor_terminates_its_child_and_cleans_up_on_interrupt(tmp_path):
    import select
    process, target, cleanup, _ = _monitor(tmp_path, payload=b"exact input", wait=True)
    try:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready
        child_pid = int(process.stdout.readline())
        process.terminate()
        process.communicate(timeout=15)
        assert process.returncode != 0 and cleanup.is_file() and target.read_bytes() == b"exact input"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
