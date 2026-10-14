"""Native admitted planning/index/router qualification with authored providers.

These fixtures measure displaced planning calls and complete input text only.
They neither solve a benchmark nor claim end-to-end provider token savings.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.indexed_doctor_lifecycle import _context
from benchmarks.agent_supervisor.container_coding.local_live_planner import preflight_proposal
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import (
    load_public_instruction, prepare_public_instruction_context,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
from test.api.test_semantic_router_integration import provider  # noqa: F401


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
                          check=True, capture_output=True).stdout


def _offline_counter(monkeypatch):
    """Optional named local estimate; no claim about the actual model tokenizer."""
    try:
        import tiktoken
        import tiktoken.load
        def no_network(*args, **kwargs):
            raise RuntimeError("offline tokenizer cache required")
        monkeypatch.setattr(tiktoken.load, "read_file", no_network)
        encoding = tiktoken.get_encoding("o200k_base")
    except (ImportError, RuntimeError, ValueError):
        return None
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def _prepare_case(directory, count, mode, monkeypatch):
    directory.mkdir()
    root, state, instruction = directory / "repository", directory / "state", directory / "instruction.md"
    root.mkdir()
    source = "".join(
        f"def normalize_label_{index}(value):\n"
        "    if not isinstance(value, str):\n"
        "        raise TypeError('text required')\n"
        "    return value.strip().casefold()\n\n"
        for index in range(count))
    (root / "labels.py").write_text(source)
    for args in (("init", "-q"), ("add", "."), ("-c", "user.name=Fixture",
                 "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Authored public input")):
        _git(root, *args)
    instruction.write_text("Improve labels.py while preserving its public signatures, string normalization, "
                           "and rejection of non-string values. Keep all declared source restrictions.")
    task_profile = dict(schema="terminal-public-task-profile@1",
        instruction_sha256=hashlib.sha256(instruction.read_bytes()).hexdigest(), input_paths=["labels.py"],
        outputs=[dict(path="labels.py", effect="modify", media_type="text/x-python")])
    prepared = prep.prepare(repository=root, instruction=instruction, state=state, task_profile=task_profile,
        disable_intent_autoencoder=True, provider_profile="codex-gpt-6.1-sol-0.160.0@1",
        **({"planning_mode": mode} if mode != "model" else {}))
    initial = prep.initial_context(state=state)
    calls = []
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    proposal = preflight_proposal({"spec": prepared["spec"],
        "evidence": _select_evidence(request, scan, prep._config(root))})
    proposal["root_goal_key"] = "TB-GOAL"
    proposal["goals"][0].update(goal_key="TB-GOAL", title="Authored label maintenance", objective="Fulfill the public instruction")
    proposal["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Authored label task", objective="Fulfill the public instruction")
    proposal["tasks"][0].update(task_key=prepared["spec"]["task_key"], goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction", predicted_files=["labels.py"])
    def authored_planner(prompt, **kwargs):
        if mode != "model":
            pytest.fail("declared contract must not invoke a planning provider")
        calls.append(prompt)
        return {"text": json.dumps(proposal), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    planned = prep.plan(state, provider_callable=authored_planner)
    assert planned["qualified"] is True
    assert len(calls) == planned["provider_calls"] == (1 if mode == "model" else 0)
    if mode != "model":
        assert planned["planning_strategy"] == "declared_contract"
        assert planned["execution_receipt"] is None and planned["model_request_sha256"] is None
        assert planned["declared_contract_planning"]
    indexed = prep.context(state=state)
    assert indexed["initial_indexes_reused"] and indexed["new_embedding_calls"] == 0
    assert indexed["index_id"] == initial["index_id"]
    admission = json.loads((state / "admission.json").read_text())
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    assert len(verified["graph"].tasks) == len(planned["task_cids"]) == 1
    # Match the owner bootstrap's public instruction storage, preserving the
    # production preparation/reader custody checks rather than mocking them.
    (root / ".runtime").chmod(0o755)
    instruction_context = prepare_public_instruction_context(repository=root, admission=admission,
        task_cid=planned["task_cids"][0], source_path=prep.INSTRUCTION,
        expected_source_sha256=verified["manifest"]["sources"][prep.INSTRUCTION]["sha256"])
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task = intent.get_task(planned["task_cids"][0])
        assert task["status"] == "ready"
        watermark = intent.event_watermark()
        spec = prepared["spec"]
        portal = PortalTask(task_id=task["task_alias"], title=task["body"]["title"], canonical_task_cid=task["task_cid"],
            status="ready", completion="manual", priority="P2", track="implementation",
            outputs=[row["path"] for row in spec["outputs"]],
            validation=[shlex.join(row["argv"]) for row in spec["validations"]],
            acceptance=" ; ".join(row["criterion"] for row in spec["acceptance"]),
            metadata={"database task cid": task["task_cid"],
                "local planning contract cid": local.content_identity(task["body"][local.CONTRACT_KEY])})
        daemon = TodoImplementationDaemon(todo_path=state / "pending.todo.md", state_path=state / "worker/tasks.json",
            strategy_path=state / "worker/strategy.json", events_path=state / "worker/events.jsonl",
            repo_root=root, task_header_prefix="## TB-")
        daemon._world_intent_repository = intent
        daemon._task_context_nomination_bundle = indexed["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
        finally:
            daemon.close_event_runtime()
        assert intent.get_task(task["task_cid"]) == task and intent.event_watermark() == watermark
    assert _context(prompt, "code-retrieval-context")["index_id"] == indexed["index_id"]
    assert _context(prompt, "intent-world-context")["world_snapshot_cid"] == indexed["world_snapshot_cid"]
    workspace = directory / "allocated"
    _git(root, "worktree", "add", "--detach", "-q", str(workspace))
    return dict(root=root, state=state, workspace=workspace, source=source, prompt=prompt,
        task=task, planned=planned, calls=calls, indexed=indexed, instruction_context=instruction_context)


@pytest.mark.parametrize("function_count", [4, 8])
def test_declared_planning_and_compact_transport_reach_native_indexed_router(
        tmp_path, monkeypatch, provider, record_property, function_count):
    counter = _offline_counter(monkeypatch)
    evidence = []
    for mode in ("model", "declared-contract@1"):
        case = _prepare_case(tmp_path / ("model" if mode == "model" else "declared"), function_count, mode, monkeypatch)
        root, workspace, prompt = case["root"], case["workspace"], case["prompt"]
        instruction = case["instruction_context"]
        request = dict(artifact=Path(instruction["artifact"]), expected_sha256=instruction["sha256"],
            task_cid=case["task"]["task_cid"], prompt=prompt, workspace=workspace, repository=root)
        block, instruction_receipt = load_public_instruction(**request)
        rows = []
        with monkeypatch.context() as worker:
            worker.chdir(workspace)
            worker.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
            for transport in (codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA):
                before = len(provider[1])
                encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=transport)
                output, receipt = runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
                    max_output_tokens=128, semantic_repository=root, semantic_transport_schema=transport,
                    public_instruction_artifact=request["artifact"], public_instruction_sha256=request["expected_sha256"],
                    public_instruction_task_cid=request["task_cid"])
                assert output == "literal summary" and len(provider[1]) == before + 1
                actual = provider[1][-1][0]
                expected, advisory = runner.render_model_prompt(prompt=encoded.provider_prompt + block,
                    purpose="coding", workspace=workspace, semantic_transport=True)
                assert actual == expected and actual.count(block) == 1
                assert receipt["public_instruction"] == instruction_receipt
                assert receipt["model_prompt_sha256"] == _sha(actual)
                assert receipt["native_prompt_sha256"] == _sha(prompt)
                assert codec.restore_semantic_router_prompt(provider_prompt=actual[len(advisory):-len(block)],
                    table=encoded.table, repository=root) == prompt
                assert (root / "labels.py").read_text() == case["source"] == (workspace / "labels.py").read_text()
                assert receipt["completion_authority"] is False
                rows.append(dict(transport=transport, complete_prompt_bytes=len(actual.encode()),
                    complete_prompt_sha256=_sha(actual), estimated_input_tokens=None if counter is None else counter(actual),
                    exact_native_restoration_verified=True, original_source_unchanged=True,
                    literal_instruction_inclusions=actual.count(block), authored_coding_provider_calls=1))
        assert rows[1]["complete_prompt_bytes"] < rows[0]["complete_prompt_bytes"]
        if counter is not None:
            assert rows[1]["estimated_input_tokens"] < rows[0]["estimated_input_tokens"]
        evidence.append(dict(planning_mode=mode, planning_strategy=case["planned"]["planning_strategy"],
            authored_planning_provider_calls=len(case["calls"]),
            planning_prompt_bytes=sum(len(text.encode()) for text in case["calls"]),
            estimated_planning_input_tokens=None if counter is None else sum(counter(text) for text in case["calls"]),
            persisted_ready_tasks=1, initial_indexes_reused=True, new_embedding_calls=0,
            indexed_symbols=case["indexed"]["indexed_symbols"], coding_inputs=rows))
        _git(root, "worktree", "remove", str(workspace))
    record_property("symbolic_efficiency_qualification", json.dumps(dict(
        schema="terminal-symbolic-efficiency-fixture@1", functions=function_count, observations=evidence,
        real_index_producer=True, real_signed_admission=True, real_router_dispatch=True,
        external_model_calls=0, provider_usage_measurements=False, benchmark_reward_measured=False,
        end_to_end_token_savings_claimed=False, proof_or_task_completion_claimed=False,
        tokenizer_estimate="o200k_base" if counter is not None else None,
        actual_provider_tokenizer_verified=False), sort_keys=True))
