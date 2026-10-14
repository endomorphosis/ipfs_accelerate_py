"""A real Quack owner supplies signed evidence to source/MCP worker dispatch.

The listener, typed claim, validation subprocesses, source index and MCP child
are real. Provider responses are authored; no live model usage is measured.
"""
import json
import os
from pathlib import Path
import shlex

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime.repository_intent_world import (
    capture_repository_owner_intent_world,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import provider_subprocess_environment
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _admit
from test.api.test_intent_requirement_observation_native import _owner, _claim, _run
from test.api.test_repository_context_compiler import compiler
from test.api.test_repository_intent_world_cycle import provider, prepared, dispatch
from test.api.test_repository_source_mcp import real_client


def native_tcp_task_count(owner):
    """Prove the installed listener serves a fresh authenticated connection."""
    import duckdb
    connection = duckdb.connect()
    try:
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        connection.execute("LOAD httpfs")
        connection.execute("LOAD quack")
        row = connection.execute(
            "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
            [owner.identity.listen_uri, "SELECT COUNT(*) FROM tasks",
             owner.server._vault.resolve(owner.identity.secret_handle)],
        ).fetchone()
        return row[0]
    except Exception as exc:
        # Driver exceptions may include bound credentials; never print them.
        raise AssertionError("Native TCP task query failed: " + type(exc).__name__) from None
    finally:
        connection.close()


def test_native_quack_signed_checks_drive_mcp_edit_and_reject_newer_failure(
        scenario, tmp_path, monkeypatch, capsys, record_property):
    from ipfs_accelerate_py import llm_router

    admission, tasks = _admit(scenario)
    case = {**scenario, "admission": admission, "intent_manifest": admission["manifest"],
            "tasks": tasks}
    workspace, task_id = scenario["repository"], tasks["LOCAL-TASK"]
    original_check = (workspace / "test_answer.py").read_bytes()
    state = provider(monkeypatch, workspace, tmp_path)

    def worker_dispatch(packet, **kwargs):
        # Model the daemon's child boundary using its real environment scrub.
        # Keep owner credentials in the supervisor objects, outside worker env.
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
        native_revision = owner.source.get_task(task_id).revision
        with RepositorySourceSession(workspace, tmp_path / "quack-source-state") as session:
            adapter = session.context_adapter(compiler())

            def capture(generation):
                return capture_repository_owner_intent_world(session=session, server=owner.server,
                    admission=admission, task_id=task_id, generation=generation)

            first_observation = capture(1)
            assert first_observation.receipt()["intent_owner_id"] == scenario["intent"].owner_id
            assert first_observation.receipt()["native_owner_binding"]["server_id"] == owner.identity.server_id
            first = prepared(adapter, first_observation, task_id=task_id, generation=1, command=command)

            def edit(binding):
                with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
                    found, error = client.call("repository_context_query", generation=1, query="answer", limit=8)
                    assert not error, found
                    candidate = next(row for row in found["candidates"]
                        if row["path"] == "answer.py" and row["name"] == "answer.answer")
                    read, error = client.call("repository_read", generation=1, path="answer.py",
                        symbol_id=candidate["symbol_id"])
                    assert not error, read
                    changed, error = client.call("repository_replace_symbol", generation=1, path="answer.py",
                        symbol=read["symbol"], expected_sha256=read["source_sha256"],
                        representation=read["representation"],
                        replacement=read["content"].replace("return 1", "return 2"))
                    assert not error and changed["canonical_changed"] is True

            state["callback"] = edit
            _, first_receipt = worker_dispatch(first, repository_source_policy=policy.POLICIES[2])
            assert failed["results"][0]["evidence_digest"] in state["seen"][0]
            assert (workspace / "test_answer.py").read_bytes() == original_check
            state["callback"] = None
            with pytest.raises(ValueError):
                worker_dispatch(first)
            assert len(state["seen"]) == 1
            assert session.refresh(generation=1)["generation"] == 2
            passed = _run(owner, attempt, timeout=10)
            assert passed["passed"] is True
            successor = capture(2)
            value = json.loads(successor.reference.summary)
            assert value["observed_task"]["validations"][0]["status"] == "passed"
            assert passed["results"][0]["evidence_digest"] in successor.reference.summary
            assert failed["results"][0]["evidence_digest"] not in successor.reference.summary
            second = prepared(adapter, successor, task_id=task_id, generation=2, command=command)
            _, second_receipt = worker_dispatch(second)
            assert len(state["seen"]) == 2

            answer = workspace / "answer.py"
            original_answer = answer.read_bytes()
            def newer_failure(*args, **kwargs):
                answer.write_text("def answer():\n    return 3\n")
                try:
                    assert run_owner_local_task_validations(server=owner.server,
                        task_cid=task_id, attempt_id=attempt.attempt_id,
                        expected_revision=native_revision, timeout=10)["passed"] is False
                finally:
                    answer.write_bytes(original_answer)
                assert session.status_summary()["source_current"] is True
                return object()

            capsys.readouterr()
            monkeypatch.setattr(llm_router, "get_llm_provider", newer_failure)
            with pytest.raises(ValueError):
                worker_dispatch(second)
            rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
            refused = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
            assert refused["failure_phase"] == "world_task_packet_currentness"
            assert refused["world_task_packet"]["provider_attempted"] is False
            assert refused["world_task_packet"]["dispatch_rechecked"] is False
            assert len(state["seen"]) == 2
            for receipt, observation in ((first_receipt, first_observation), (second_receipt, successor)):
                assert receipt["world_task_packet"]["dispatch_rechecked"] is True
                assert receipt["world_task_packet"]["intent_world"] == observation.receipt()
                assert receipt["completion_authority"] is False
            private = (owner.credentials.token, owner.identity.listen_uri,
                       str(owner.database), str(owner.server.typed_command_socket_path()))
            for prompt in state["seen"]:
                assert all(value not in prompt for value in private)
            assert owner.source.get_task(task_id).status == "in_progress"
            record_property("native_quack_world_cycle", json.dumps(dict(
                native_quack_listener=True, native_tcp_task_count=1, native_typed_attempt=True,
                provider_environment_scrubbed=True,
                native_signed_command_checks=3, native_source_generations=2,
                real_child_mcp=True, authored_provider_calls=2, live_provider_calls=0,
                stale_source_dispatches_refused=1, newer_owner_observation_dispatches_refused=1,
                task_completed=False, full_session_token_savings=None), sort_keys=True))
