"""Actual public-task parsing, signed admission and native prompt preflight.

All task instructions/sources below are authored fixtures. No provider, Docker
command, benchmark verifier or source mutation is used to claim task success.
"""
import hashlib
import json
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as preflight
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    PLANNER180_SOURCE384_PROFILE, execution_budget,
)
from benchmarks.agent_supervisor.container_coding.indexed_doctor_lifecycle import _context
from benchmarks.agent_supervisor.container_coding.test_terminal_multitask_preparation import multitask_case
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.entrypoints import local_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import instruction_sha256
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _git(repository, *args):
    return subprocess.check_output(["git", "-c", "gc.auto=0", "-C", str(repository), *args])


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def no_provider_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "lifecycle-registry")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    monkeypatch.setattr(llm_router, "generate_text", lambda *args, **kwargs: pytest.fail(
        "provider-free native preflight must not generate model text"))
    monkeypatch.setattr(prep, "generate_prompt_goal_graph", lambda *args, **kwargs: pytest.fail(
        "authored native preflight must not invoke the model planner"))


def _case(tmp_path, kind):
    repository = tmp_path / "app"
    repository.mkdir()
    if kind == "legacy":
        sources = {"bottle.py": b"def application():\n    return 'original authored application'\n"}
        text = "Inspect bottle.py, repair application and write report.jsonl with file_path and cwe_id fields.\n"
        profile = None
    elif kind == "data":
        sources = {"requests.json": b'{"requests":[{"item":"authored","count":2}]}\n'}
        text = "Read requests.json and create result.json describing the requested item counts.\n"
        profile = {
            "schema": "terminal-public-task-profile@2", "instruction_sha256": instruction_sha256(text),
            "input_paths": ["requests.json"],
            "data_inputs": [{"path": "requests.json", "media_type": "application/json"}],
            "outputs": [{"path": "result.json", "effect": "create", "media_type": "application/json"}],
        }
    else:
        sources = {
            "module.py": b"def compute_count(value):\n    return value + 1\n",
            "helper.py": b"def count_items(values):\n    return len(values)\n",
        }
        text = "Update module.py compute_count using helper.py count_items and create summary.json.\n"
        profile = {
            "schema": "terminal-public-task-profile@1", "instruction_sha256": instruction_sha256(text),
            "input_paths": sorted(sources),
            "outputs": [{"path": "module.py", "effect": "modify", "media_type": "text/x-python"},
                        {"path": "summary.json", "effect": "create", "media_type": "application/json"}],
        }
    for name, raw in sources.items():
        (repository / name).write_bytes(raw)
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=Authored native preflight", "-c",
         "user.email=preflight@example.invalid", "commit", "-qm", "Public authored input")
    instruction = tmp_path / "instruction.md"
    instruction.write_text(text)
    return dict(repository=repository, instruction=instruction, state=tmp_path / "state",
                task_profile=profile, sources=sources, text=text)


def _qualify(case, **options):
    return preflight.qualify(repository=case["repository"], instruction=case["instruction"],
        state=case["state"], task_profile=case["task_profile"], disable_intent_autoencoder=True,
        **options)


@pytest.mark.parametrize("kind", ["generic", "data", "legacy"])
def test_native_preflight_uses_exact_public_declaration_and_context(tmp_path, kind):
    case = _case(tmp_path, kind)
    result = _qualify(case, provider_profile=GROK_PROFILE)
    state, repository = case["state"], case["repository"]
    prepared = prep._load_prepared(state)
    admission = json.loads((state / "admission.json").read_text())
    verified = local.verify_local_benchmark_admission(admission)
    graph = verified["graph"]
    assert len(graph.goals) == 2 and len(graph.tasks) == 1
    graph_task = graph.to_dict()["tasks"][0]
    assert graph_task["task_key"] == prepared["spec"]["task_key"]
    assert [{key: row[key] for key in ("path", "effect", "media_type")}
            for row in graph_task["outputs"]] == prepared["spec"]["outputs"]
    assert graph_task["predicted_files"] == [item["path"] for item in prepared["spec"]["outputs"]]
    assert set(graph_task["scope_paths"]) == set(prepared["spec"]["scope_paths"])
    assert [{key: row[key] for key in ("criterion_key", "criterion", "evidence_cids", "validation_keys")}
            for row in graph_task["acceptance"]] == prepared["spec"]["acceptance"]
    validation_fields = ("validation_key", "argv", "cwd", "expected_exit_codes")
    assert [{key: row[key] for key in validation_fields} for row in graph_task["validations"]] == [
        {key: row[key] for key in validation_fields} for row in prepared["spec"]["validations"]]
    manifest_sources = prepared["manifest"]["payload"]["sources"]
    assert result["input_sha256"] == {name: row["sha256"] for name, row in manifest_sources.items()}
    assert result["source_count"] == len(manifest_sources)
    assert result["input_sha256"] == {name: _sha((repository / name).read_bytes()) for name in manifest_sources}
    assert all((repository / name).read_bytes() == raw for name, raw in case["sources"].items())
    assert prepared["provider"] == "grok_cli" and prepared["model"] == "grok-4.7"
    assert result["qualified"] is True and result["text_generation_calls"] == 0
    assert result["benchmark_success"] is None
    assert result["official_verifier_executed"] is False
    assert result["supervisor_execution_qualified"] is result["completion_authority"] is False
    assert result["query_sha256"] == _sha(case["text"].encode())
    assert result["query_is_exact_public_instruction"] is True
    prompt = (state / "worker-prompt.txt").read_text()
    retrieval, world = _context(prompt, "code-retrieval-context"), _context(prompt, "intent-world-context")
    assert retrieval["status"] == "current"
    assert retrieval["index_id"] == result["context"]["index_id"]
    assert world["world_snapshot_cid"] == result["context"]["world_snapshot_cid"]
    assert world["semantic_root_cid"] == result["context"]["semantic_root_cid"]
    assert result["native_prompt_sha256"] == _sha(prompt.encode())
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        tasks = intent.list_tasks()
        assert len(tasks) == 1 and tasks[0]["task_cid"] == result["context"]["task_cid"]
        assert tasks[0]["status"] == "ready"
        assert not intent.plan_projection(task_cids=[tasks[0]["task_cid"]])["tasks"][0]["dependencies"]
    for output in prepared["spec"]["outputs"]:
        if output["effect"] == "create":
            assert not (repository / output["path"]).exists()
    if kind == "legacy":
        assert result["bottle_sha256"] == _sha(case["sources"]["bottle.py"])
    else:
        assert "bottle_sha256" not in result
        assert "bottle.py" not in manifest_sources and "report.jsonl" not in graph_task["predicted_files"]
    if kind == "data":
        assert result["context"]["index_id"] is None
        assert result["context"]["indexed_symbols"] == 0
        assert result["context"]["embedding_calls"] == 0
        assert retrieval["hits"] == []


@pytest.mark.parametrize("changed", ["task_cid", "index_id", "semantic_root_cid", "world_snapshot_cid"])
def test_foreign_context_identity_cannot_qualify_native_prompt(tmp_path, monkeypatch, changed):
    case = _case(tmp_path, "data")
    context = prep.context

    def observed_context(**kwargs):
        value = context(**kwargs)
        value[changed] = "sha256:" + "f" * 64
        return value

    monkeypatch.setattr(prep, "context", observed_context)
    with pytest.raises((RuntimeError, ValueError), match="context|task|prompt"):
        _qualify(case)
    assert not (case["state"] / "native-preflight-result.json").exists()
    assert not (case["repository"] / "result.json").exists()


def test_multitask_preflight_refuses_before_repository_or_state_mutation(multitask_case):
    case = multitask_case
    before_head = _git(case["repository"], "rev-parse", "HEAD")
    before = {name: (case["repository"] / name).read_bytes() for name in case["source_bytes"]}
    with pytest.raises(ValueError, match="exactly one declared task"):
        preflight.qualify(repository=case["repository"], instruction=case["instruction"], state=case["state"],
                          task_profile=case["profile"], disable_intent_autoencoder=True)
    assert not case["state"].exists()
    assert not (case["repository"] / prep.INSTRUCTION).exists()
    assert not (case["repository"] / prep.SMOKE).exists()
    assert _git(case["repository"], "rev-parse", "HEAD") == before_head
    assert {name: (case["repository"] / name).read_bytes() for name in before} == before


def test_different_instruction_profile_is_refused_before_preparation(tmp_path):
    case = _case(tmp_path, "generic")
    case["task_profile"]["instruction_sha256"] = "0" * 64
    before = _git(case["repository"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="instruction"):
        _qualify(case)
    assert not case["state"].exists()
    assert _git(case["repository"], "rev-parse", "HEAD") == before


@pytest.mark.parametrize("timeout", [None, 12.5])
def test_source384_controls_precede_admission_and_never_enable_training(tmp_path, monkeypatch, timeout):
    """Check wiring only; the expensive Source384 stage is deliberately a double."""
    case = _case(tmp_path, "data")
    config = tmp_path / "source384-test-config.json"
    config.write_text('{}\n')
    stages, calls = [], []
    actual_graph, actual_context = preflight._authored_graph, prep.context

    def initial_context(**kwargs):
        # Real prepare has already written its signed preparation, but this
        # learned stage precedes graph admission and task materialization.
        prepared = prep._load_prepared(case["state"])
        assert prepared["provider"] == "grok_cli"
        assert prepared["planner_timeout_seconds"] == 180
        assert not (case["state"] / "admission.json").exists()
        assert not (case["state"] / "intent.duckdb").exists()
        calls.append(kwargs)
        stages.append("initial")
        return {"fixture_stage_only": True, "learned_inference_exercised": False}

    def authored_graph(prepared):
        stages.append("graph")
        return actual_graph(prepared)

    def context(**kwargs):
        stages.append("context")
        return actual_context(**kwargs)

    monkeypatch.setattr(prep, "initial_context", initial_context)
    monkeypatch.setattr(preflight, "_authored_graph", authored_graph)
    monkeypatch.setattr(prep, "context", context)
    result = _qualify(case, provider_profile=GROK_PROFILE,
        resource_profile=PLANNER180_SOURCE384_PROFILE, source384_config=config,
        source384_timeout_seconds=timeout)
    expected_timeout = timeout if timeout is not None else execution_budget(
        PLANNER180_SOURCE384_PROFILE)["source384_seconds"]
    assert stages == ["initial", "graph", "context"]
    assert calls == [{"state": case["state"], "model_snapshot": None, "model_revision": "",
        "source384_config": config, "train_autoencoder": False,
        "source384_timeout_seconds": expected_timeout}]
    assert result["source384_timeout_seconds"] == expected_timeout
    assert result["initial_context"] == {"fixture_stage_only": True, "learned_inference_exercised": False}
    assert result["provider_profile"] == GROK_PROFILE
    assert result["resource_profile"] == PLANNER180_SOURCE384_PROFILE
    assert result["source384_enabled"] is True


@pytest.mark.parametrize("timeout,has_config", [
    (0, True), (-1, True), (float("nan"), True), (float("inf"), True),
    (True, True), (181, True), (1, False),
])
def test_invalid_source384_timeout_refuses_before_preparation(tmp_path, monkeypatch, timeout, has_config):
    case = _case(tmp_path, "data")
    before = _git(case["repository"], "rev-parse", "HEAD")
    monkeypatch.setattr(prep, "prepare", lambda **kwargs: pytest.fail(
        "invalid Source384 bounds must be refused before signed preparation"))
    with pytest.raises(ValueError, match="Source384 timeout"):
        _qualify(case, resource_profile=PLANNER180_SOURCE384_PROFILE,
            source384_config=(tmp_path / "unused-config.json") if has_config else None,
            source384_timeout_seconds=timeout)
    assert not case["state"].exists()
    assert _git(case["repository"], "rev-parse", "HEAD") == before


def test_cli_parses_public_profile_and_forwards_optional_projection_controls(tmp_path, monkeypatch, capsys):
    """CLI parsing alone cannot qualify inference or native execution."""
    case = _case(tmp_path, "data")
    declaration = tmp_path / "public-task.json"
    declaration.write_text(json.dumps(case["task_profile"]))
    options = {"source384_config": tmp_path / "source384.json",
        "intent_checkpoint_descriptor": tmp_path / "intent.json",
        "intent_action_384_config": tmp_path / "action384.json",
        "intent_projection_request": tmp_path / "projection.json",
        "source_unit_security_decoder_descriptor": tmp_path / "security-decoder.json",
        "source_unit_intent_family_context": tmp_path / "families.json"}
    calls = []

    def qualify(**kwargs):
        calls.append(kwargs)
        return {"qualified": False, "cli_parsing_only": True}

    argv = ["terminal_native_preflight", "--repository", str(case["repository"]),
        "--instruction", str(case["instruction"]), "--state", str(case["state"]),
        "--task-profile", str(declaration), "--provider-profile", GROK_PROFILE,
        "--resource-profile", PLANNER180_SOURCE384_PROFILE,
        "--source384-timeout-seconds", "12.5", "--disable-intent-autoencoder",
        "--enable-source-unit-autoencoder", "--source-unit-project-logic-families",
        "--intent-projection-request-sha256", "a" * 64,
        "--source-unit-intent-logic-family", "fol",
        "--source-unit-intent-logic-family", "tla_plus"]
    for name, value in options.items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(preflight, "qualify", qualify)
    preflight.main()
    assert json.loads(capsys.readouterr().out) == {"qualified": False, "cli_parsing_only": True}
    assert len(calls) == 1
    observed = calls[0]
    assert observed["task_profile"] == case["task_profile"]
    assert all(observed[name] == value for name, value in options.items())
    assert observed["source_unit_intent_logic_families"] == ["fol", "tla_plus"]
    assert observed["intent_projection_request_sha256"] == "a" * 64
    assert observed["source384_timeout_seconds"] == 12.5
    assert observed["provider_profile"] == GROK_PROFILE
    assert observed["resource_profile"] == PLANNER180_SOURCE384_PROFILE
    assert observed["disable_intent_autoencoder"] is True
    assert observed["enable_source_unit_autoencoder"] is True
    assert observed["source_unit_project_logic_families"] is True
    assert not case["state"].exists()


def test_cli_duplicate_profile_keys_refuse_before_qualification(tmp_path, monkeypatch):
    declaration = tmp_path / "ambiguous-profile.json"
    declaration.write_text('{"schema":"terminal-public-task-profile@1","schema":"terminal-public-task-profile@2"}')
    monkeypatch.setattr(sys, "argv", ["terminal_native_preflight",
        "--repository", str(tmp_path / "repository"),
        "--instruction", str(tmp_path / "instruction.md"),
        "--state", str(tmp_path / "state"), "--task-profile", str(declaration)])
    monkeypatch.setattr(preflight, "qualify", lambda **kwargs: pytest.fail(
        "ambiguous public profile must not reach preparation or admission"))
    with pytest.raises(ValueError, match="duplicate"):
        preflight.main()
    assert not (tmp_path / "state").exists()
