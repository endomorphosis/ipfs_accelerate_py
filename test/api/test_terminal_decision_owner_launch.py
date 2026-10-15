"""Signed owner routing and actual monitor stdin replay, without providers.

The entry test executes only production argument/staging statements; identity
and container qualification are outside scope. Monitor checks launch authored
local Python children and replace only the fixed sudo cleanup command.
"""
from __future__ import annotations

import ast
import builtins
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


@pytest.fixture
def owner_entry(monkeypatch, capsys):
    parsed = ast.parse(deployment.OWNER_ENTRY)
    assignments = {node.targets[0].id: index for index, node in enumerate(parsed.body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
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
        def imported(name, globals=None, locals=None, fromlist=(), level=0):
            assert name == "ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner"
            return SimpleNamespace(prepare_decision_owner_dispatch=prepare)
        monkeypatch.setattr(sys, "argv", ["router-worker", *arguments])
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
        monkeypatch.setattr(sys, "path", list(sys.path))
        namespace = {"__builtins__": {**vars(builtins), "__import__": imported},
                     "sys": sys, "json": json, "root": Path("/opt/ipfs-supervisor"),
                     "cwd": Path("/allocated")}
        if refuse:
            with pytest.raises(SystemExit) as error:
                exec(code, namespace)
            assert error.value.code == 1
        else:
            exec(code, namespace)
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
        + (f"owner_prompt_bytes={payload!r}\n" if payload is not None else "")
        + f"exec({deployment.OWNER_MONITOR!r})\n")
    return subprocess.Popen([sys.executable, "-I", str(harness)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE), target, cleanup, popen_options


@pytest.mark.parametrize("child_code,cleanup_code,expected", [(0, 0, 0), (7, 0, 7), (0, 1, 1)])
def test_actual_monitor_replays_bytes_and_preserves_exit_and_cleanup(tmp_path, child_code, cleanup_code, expected):
    payload = "λ\r\nverbatim required suffix\r\n".encode()
    process, target, cleanup, options = _monitor(tmp_path, child_code=child_code,
                                               cleanup_code=cleanup_code, payload=payload)
    process.communicate(timeout=10)
    assert process.returncode == expected and target.read_bytes() == payload
    assert cleanup.is_file() and json.loads(options.read_text())["stdin"] == subprocess.PIPE


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
