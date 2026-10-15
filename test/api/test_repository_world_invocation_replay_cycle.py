"""Replay a real native-owner/MCP dispatch after its live context is unavailable.

Native Quack, signed command execution, source indexing, and the MCP child are
real. The provider response is authored. Historical replay grants no current
owner, source, signature, execution, or delivery authority.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_intent_world import (
    capture_repository_owner_intent_world,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import (
    freeze_repository_world_invocation,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import (
    freeze_repository_world_packet,
)
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _admit
from test.api.test_intent_requirement_observation_native import _owner, _claim, _run
from test.api.test_repository_context_compiler import compiler
from test.api.test_repository_intent_world_cycle import TASK, provider, prepared, dispatch
from test.api.test_repository_intent_world_native_cycle import native_tcp_task_count
from test.api.test_repository_source_mcp import real_client


_REPLAY_CHILD = r'''
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys

import duckdb
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import provider_subprocess_environment
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import replay_repository_world_invocation
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

artifact_path, expected_sha256, original_workspace, unavailable_root = sys.argv[1:]
assert not Path(original_workspace).exists()
assert provider_subprocess_environment(os.environ) == dict(os.environ)
raw = Path(artifact_path).read_bytes()

def denied(*args, **kwargs):
    raise AssertionError("historical replay attempted a live owner, index, provider, or process operation")

duckdb.connect = denied
sqlite3.connect = denied
socket.socket = denied
subprocess.Popen = denied
IntentRepository.__init__ = denied
RepositorySourceSession.__init__ = denied
llm_router.get_llm_provider = denied
llm_router.generate_text = denied

unavailable = Path(unavailable_root).absolute()
def no_fixture_reads(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        candidate = Path(os.fsdecode(args[0])).absolute()
        if candidate == unavailable or unavailable in candidate.parents:
            raise AssertionError("historical replay attempted to read the unavailable fixture")
sys.addaudithook(no_fixture_reads)

replayed = replay_repository_world_invocation(raw, expected_sha256=expected_sha256)
receipt = replayed.receipt()
for name in ("source_currentness_verified", "owner_signature_replayed", "execution_authority",
             "completion_authority", "provider_delivery_verified", "token_counts_replayed"):
    assert receipt[name] is False, name
assert replayed.artifact_sha256 == expected_sha256
print(json.dumps({"schema": "native-world-historical-replay-test@1",
    "artifact_sha256": replayed.artifact_sha256,
    "model_prompt_sha256": hashlib.sha256(replayed.model_prompt.encode()).hexdigest(),
    "model_prompt_bytes": len(replayed.model_prompt.encode()),
    "receipt": receipt,
    "live_apis_denied": True,
    "fixture_reads_denied": True,
    "original_workspace_unavailable": True,
    "native_credentials_absent": True}, sort_keys=True))
'''


def _assert_no_private(values, serialized):
    # Keep credential values out of pytest assertion introspection on failure.
    if any(value and value in serialized for value in values):
        raise AssertionError("native owner private binding leaked into replay input or output")


def test_native_owner_mcp_dispatch_replays_after_owner_and_workspace_are_unavailable(
        scenario, tmp_path, monkeypatch, record_property):
    admission, tasks = _admit(scenario)
    case = {**scenario, "admission": admission, "intent_manifest": admission["manifest"],
            "tasks": tasks}
    workspace, task_id = scenario["repository"], tasks["LOCAL-TASK"]
    original_check = (workspace / "test_answer.py").read_bytes()
    source_state = tmp_path / "replay-source-state"
    state = provider(monkeypatch, workspace, tmp_path)

    def worker_dispatch(packet, **kwargs):
        cleaned = provider_subprocess_environment(os.environ)
        with monkeypatch.context() as boundary:
            for key in set(os.environ) - set(cleaned):
                boundary.delenv(key)
            return dispatch(packet, **kwargs)

    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        assert attempt.task_cid == task_id
        assert native_tcp_task_count(owner) == 1
        failed = _run(owner, attempt, timeout=10)
        assert failed["passed"] is False
        command = shlex.join(owner.source.get_task(task_id).validations[0]["argv"])
        private = (owner.credentials.token, owner.identity.listen_uri,
                   str(owner.database), str(owner.server.typed_command_socket_path()))
        with RepositorySourceSession(workspace, source_state) as session:
            observation = capture_repository_owner_intent_world(session=session,
                server=owner.server, admission=admission, task_id=task_id, generation=1)
            packet = prepared(session.context_adapter(compiler()), observation,
                task_id=task_id, generation=1, command=command)
            # Freeze while the live observation is current. The actual worker
            # will edit the source, so a later live freeze must be refused.
            packet_artifact = freeze_repository_world_packet(packet)
            packet_sha256 = hashlib.sha256(packet_artifact).hexdigest()

            def edit(binding):
                with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
                    found, error = client.call("repository_context_query", generation=1,
                        query="answer", limit=8)
                    assert not error, found
                    candidate = next(row for row in found["candidates"]
                        if row["path"] == "answer.py" and row["name"] == "answer.answer")
                    read, error = client.call("repository_read", generation=1, path="answer.py",
                        symbol_id=candidate["symbol_id"])
                    assert not error, read
                    changed, error = client.call("repository_replace_symbol", generation=1,
                        path="answer.py", symbol=read["symbol"],
                        expected_sha256=read["source_sha256"], representation=read["representation"],
                        replacement=read["content"].replace("return 1", "return 2"))
                    assert not error and changed["canonical_changed"] is True

            state["callback"] = edit
            _, invocation = worker_dispatch(packet, repository_source_policy=policy.POLICIES[2])
            assert len(state["seen"]) == 1
            actual_prompt = state["seen"][0]
            assert failed["results"][0]["evidence_digest"] in actual_prompt
            assert (workspace / "answer.py").read_text() == "def answer():\n    return 2\n"
            assert (workspace / "test_answer.py").read_bytes() == original_check
            assert invocation["world_task_packet"]["provider_attempted"] is True
            assert invocation["world_task_packet"]["dispatch_rechecked"] is True
            assert invocation["completion_authority"] is False
            state["callback"] = None
            with pytest.raises(ValueError):
                worker_dispatch(packet)
            with pytest.raises(ValueError):
                freeze_repository_world_packet(packet)
            assert len(state["seen"]) == 1
            assert owner.source.get_task(task_id).status == "in_progress"
            artifact = freeze_repository_world_invocation(packet_artifact=packet_artifact,
                expected_packet_sha256=packet_sha256, invocation_receipt=invocation)
            _assert_no_private(private, artifact.decode())
            _assert_no_private(private, actual_prompt)
            child_environment = provider_subprocess_environment(os.environ)
            _assert_no_private(private, json.dumps(child_environment))

    # No open owner or source session survives the boundary. Remove every
    # original fixture location the native capture or source packet used.
    scenario["intent"].close()
    monkeypatch.chdir(tmp_path)
    archived = tmp_path / "unavailable-live-fixture"
    archived.mkdir()
    for original in (workspace, Path(scenario["intent"].database_path),
                     scenario["profile"], scenario["lifecycle"], source_state,
                     tmp_path / "requirement-native-owner"):
        if original.exists():
            original.rename(archived / original.name)
    assert not workspace.exists()
    artifact_path = tmp_path / "historical-invocation.json"
    artifact_path.write_bytes(artifact)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    result = subprocess.run([sys.executable, "-s", "-c", _REPLAY_CHILD,
        str(artifact_path), artifact_sha256, str(workspace), str(tmp_path)],
        cwd=tmp_path, env=child_environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith('{"')]
    replayed = next(row for row in rows
        if row.get("schema") == "native-world-historical-replay-test@1")
    assert replayed["artifact_sha256"] == artifact_sha256
    assert replayed["model_prompt_sha256"] == hashlib.sha256(actual_prompt.encode()).hexdigest()
    assert replayed["model_prompt_sha256"] == invocation["model_prompt_sha256"]
    assert replayed["model_prompt_bytes"] == invocation["model_prompt_bytes"]
    assert TASK not in json.dumps(replayed["receipt"])
    _assert_no_private(private, json.dumps(replayed))
    record_property("native_world_replay_cycle", json.dumps(dict(
        native_quack_listener=True, native_tcp_task_count=1, native_typed_attempt=True,
        native_signed_command_checks=1, native_source_generations=1,
        real_child_mcp=True, authored_provider_calls=1, live_provider_calls=0,
        stale_source_dispatches_refused=1, stale_live_freezes_refused=1,
        native_owner_and_source_sessions_closed=True, original_workspace_unavailable=True,
        fresh_python_replay_process=True, replay_native_credentials_absent=True,
        replay_live_apis_denied=True, replay_fixture_reads_denied=True,
        exact_final_model_prompt_reconstructed=True, replay_authority_granted=False,
        task_completed=False, full_session_token_savings=None), sort_keys=True))
