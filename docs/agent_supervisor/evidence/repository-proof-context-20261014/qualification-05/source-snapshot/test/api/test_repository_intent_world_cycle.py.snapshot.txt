"""Native signed checks feed real source/context/MCP/dispatch boundaries.

Provider responses are authored. Intent materialization, signature verification,
DuckDB reads, command execution, source indexing and MCP mutation are real.
"""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    local_planning_admission as local,
    repository_source_policy as policy,
    router_implementation_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_intent_world import capture_repository_intent_world
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _admit, _start
from test.api.test_repository_context_compiler import compile_context, compiler, obligation
from test.api.test_repository_source_mcp import real_client


TASK = "Change answer() to return 2; keep the public check unchanged."
EFFECTS = ("repository_write", "validation")
PATHS = ("answer.py",)


def provider(monkeypatch, workspace, tmp_path):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-native-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    state = {"seen": [], "callback": None}

    def generate(prompt, **kwargs):
        state["seen"].append(prompt)
        observed = {"exit_code": 0, "prompt_tokens": 15, "completion_tokens": 4}
        binding = kwargs.get("codex_repository_source_binding")
        if binding is not None:
            policy.validate_binding(binding, workspace=workspace, materialized=True)
            observed.update(codex_repository_source_configured=True,
                codex_repository_source_config_sha256=binding["config_sha256"])
        set_last_cli_observation("codex_cli", observed)
        if state["callback"] is not None:
            state["callback"](binding)
        return "Authored provider response; native owner validation remains required."

    monkeypatch.setattr(llm_router, "generate_text", generate)
    return state


def prepared(adapter, observed, *, task_id, generation, command):
    required = (obligation("preserve-check", summary="Do not change test_answer.py or the declared task scope."),
                observed.reference)
    compiled = compile_context(adapter, required, generation=generation, query="answer",
        goal={"intent": TASK}, scope={"paths": PATHS},
        authority={"mode": "proposal", "allowed_paths": PATHS,
            "world_task_packet": repository_world_task_binding(task_id, TASK, EFFECTS, PATHS)},
        acceptance={"criteria": ("The existing public answer check passes.",)})
    return prepare_repository_world_packet(adapter=adapter, compilation=compiled,
        generation=generation, task_id=task_id, task_text=TASK,
        unresolved_obligations=("Use current signed public-check observations to repair or review the task.",),
        validation_commands=(command,), allowed_effects=EFFECTS, write_paths=PATHS,
        intent_world_observation=observed)


def dispatch(packet, **changes):
    return runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, world_task_packet=packet, **changes)


def test_signed_native_failure_edit_pass_and_owner_drift_at_dispatch(
        scenario, tmp_path, monkeypatch, capsys, record_property):
    from ipfs_accelerate_py import llm_router
    admission, tasks = _admit(scenario)
    task_id = tasks["LOCAL-TASK"]
    _start(scenario, task_id)
    workspace, intent = scenario["repository"], scenario["intent"]
    unchanged_check = (workspace / "test_answer.py").read_bytes()
    command = shlex.join(intent.get_task(task_id)["validations"][0]["argv"])
    failed = local.run_local_task_validations(intent=intent, task_cid=task_id,
        attempt_id="world-cycle:failed", timeout=10)
    assert failed["passed"] is False
    state = provider(monkeypatch, workspace, tmp_path)

    with RepositorySourceSession(workspace, tmp_path / "native-source-owner") as session:
        adapter = session.context_adapter(compiler())
        first_observation = capture_repository_intent_world(session=session, intent=intent,
            admission=admission, task_id=task_id, generation=1)
        assert failed["results"][0]["evidence_digest"] in first_observation.reference.summary
        assert json.loads(first_observation.reference.summary)["observed_task"]["validations"][0]["status"] == "failed"
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
        _, first_receipt = dispatch(first, repository_source_policy=policy.POLICIES[2])
        assert failed["results"][0]["evidence_digest"] in state["seen"][0]
        assert (workspace / "test_answer.py").read_bytes() == unchanged_check
        state["callback"] = None
        with pytest.raises(ValueError):
            dispatch(first)
        assert len(state["seen"]) == 1
        assert session.refresh(generation=1)["generation"] == 2
        passed = local.run_local_task_validations(intent=intent, task_cid=task_id,
            attempt_id="world-cycle:passed", timeout=10)
        assert passed["passed"] is True
        successor = capture_repository_intent_world(session=session, intent=intent,
            admission=admission, task_id=task_id, generation=2)
        assert passed["results"][0]["evidence_digest"] in successor.reference.summary
        assert failed["results"][0]["evidence_digest"] not in successor.reference.summary
        assert json.loads(successor.reference.summary)["observed_task"]["validations"][0]["status"] == "passed"
        second = prepared(adapter, successor, task_id=task_id, generation=2, command=command)
        _, second_receipt = dispatch(second)
        assert len(state["seen"]) == 2 and second.prompt in state["seen"][1]

        # A later failed observation invalidates the old packet even after the
        # checked source is restored to the same bytes and source generation.
        answer = workspace / "answer.py"
        original = answer.read_bytes()
        def newer_failure(*args, **kwargs):
            answer.write_text("def answer():\n    return 3\n")
            try:
                result = local.run_local_task_validations(intent=intent, task_cid=task_id,
                    attempt_id="world-cycle:newer-failed", timeout=10)
                assert result["passed"] is False
            finally:
                answer.write_bytes(original)
            assert session.status_summary()["source_current"] is True
            return object()

        capsys.readouterr()
        monkeypatch.setattr(llm_router, "get_llm_provider", newer_failure)
        with pytest.raises(ValueError):
            dispatch(second)
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        refusal = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
        assert refusal["failure_phase"] == "world_task_packet_currentness"
        assert refusal["world_task_packet"]["dispatch_rechecked"] is False
        assert refusal["world_task_packet"]["provider_attempted"] is False
        assert len(state["seen"]) == 2
        for receipt in (first_receipt, second_receipt):
            assert receipt["world_task_packet"]["dispatch_rechecked"] is True
            assert receipt["world_task_packet"]["intent_world"] == (
                first_observation.receipt() if receipt is first_receipt else successor.receipt())
            assert receipt["completion_authority"] is False
        assert intent.get_task(task_id)["status"] == "in_progress"
        record_property("native_intent_world_cycle", json.dumps(dict(
            native_signed_command_checks=3, native_source_generations=2, authored_provider_calls=2,
            stale_source_dispatches_refused=1, newer_owner_observation_dispatches_refused=1,
            real_child_mcp=True, native_intent_database=True, live_provider_calls=0,
            full_session_token_savings=None, task_completed=False), sort_keys=True))
