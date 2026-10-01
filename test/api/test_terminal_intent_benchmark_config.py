"""Explicit IntentIR contract selection survives the high-level benchmark CLI."""

import asyncio
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from test.api.test_intent_plan_coverage import _contract


@pytest.fixture
def candidate(tmp_path):
    contract = _contract()
    contract["source_path"] = benchmark.INTENT_SOURCE_PATH
    text = "".join(unit["text"] for unit in contract["ledger"]["source_units"])
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(text)
    (task / "task.toml").write_text('version = "1.0"\n')
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(contract))
    return dataset, path, contract


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_old_config_calls_retain_default_arm_without_contract(tmp_path, arm):
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", arm)
    assert config["agents"][0]["kwargs"]["arm"] == arm
    assert "intent_requirement_contract" not in config["agents"][0]["kwargs"]
    assert config["job_name"] == "supervisor-" + arm + "-" + benchmark.TASK


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_config_forwards_an_independent_bounded_contract_copy(candidate, tmp_path, arm):
    dataset, _, contract = candidate
    config = benchmark.config_for(dataset, tmp_path / "output", tmp_path / "archive", arm,
                                  intent_requirement_contract=contract)
    selected = config["agents"][0]["kwargs"]["intent_requirement_contract"]
    assert selected == contract
    assert selected is not contract
    contract["requirements"].clear()
    assert selected["requirements"]
    assert config["verifier"]["disable"] is False
    assert config["n_attempts"] == 1


@pytest.mark.parametrize("value", [[{}], {"confidence": float("nan")}, {"value": "x" * benchmark.MAX_INTENT_CONTRACT_BYTES}])
def test_config_rejects_invalid_or_oversized_transport(value, tmp_path):
    with pytest.raises(ValueError):
        benchmark.config_for(tmp_path, tmp_path / "output", tmp_path / "archive", "full",
                             intent_requirement_contract=value)


def test_loader_checks_original_public_instruction_and_contract_identity(candidate):
    dataset, path, contract = candidate
    loaded = benchmark._load_intent_requirement_contract(path, dataset=dataset)
    assert loaded == contract
    selection = benchmark._intent_selection(loaded)
    assert selection["planning_strategy"] == "intent_coverage"
    assert selection["intent_requirement_contract_cid"] == cid_for_dag_json(loaded)
    assert len(selection["intent_requirement_contract_sha256"]) == 64
    assert benchmark._load_intent_requirement_contract(None, dataset=dataset) is None


def test_loader_rejects_contract_for_different_source(candidate):
    dataset, path, _ = candidate
    (dataset / benchmark.TASK / "instruction.md").write_text("agent may modify answer.")
    with pytest.raises(ValueError, match="ledger"):
        benchmark._load_intent_requirement_contract(path, dataset=dataset)


def test_loader_requires_container_public_instruction_path(candidate):
    dataset, path, contract = candidate
    contract["source_path"] = "other.md"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="original public instruction"):
        benchmark._load_intent_requirement_contract(path, dataset=dataset)


@pytest.mark.parametrize("raw", ['{"schema":1,"schema":2}', '{"value":NaN}'])
def test_loader_rejects_duplicate_keys_and_nonfinite_json(candidate, raw):
    dataset, path, _ = candidate
    path.write_text(raw)
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        benchmark._load_intent_requirement_contract(path, dataset=dataset)


def test_loader_rejects_symlinked_contract_and_public_instruction(candidate, tmp_path):
    dataset, path, _ = candidate
    linked = tmp_path / "linked-requirements.json"
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="canonical"):
        benchmark._load_intent_requirement_contract(linked, dataset=dataset)
    instruction = dataset / benchmark.TASK / "instruction.md"
    moved = tmp_path / "actual-instruction.md"
    instruction.rename(moved)
    instruction.symlink_to(moved)
    with pytest.raises(ValueError, match="canonical"):
        benchmark._load_intent_requirement_contract(path, dataset=dataset)


def test_collection_retains_selected_contract_metadata_and_detects_removed_config(candidate, tmp_path):
    dataset, _, contract = candidate
    output = tmp_path / "output"
    output.mkdir()
    selection = benchmark._intent_selection(contract)
    preparation = {"arm": "full", "dataset": str(dataset),
                   "task_input_sha256": benchmark._task_hashes(dataset / benchmark.TASK), **selection}
    (output / "preparation.json").write_text(json.dumps(preparation))
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
                                  intent_requirement_contract=contract)
    (output / "config.json").write_text(json.dumps(config))
    result = benchmark.collect(output)
    assert {key: result[key] for key in selection} == selection
    assert result["intent_selection_config_unchanged"] is True
    assert result["benchmark_advantage_claimed"] is False
    del config["agents"][0]["kwargs"]["intent_requirement_contract"]
    (output / "config.json").write_text(json.dumps(config))
    changed = benchmark.collect(output)
    assert changed["planning_strategy"] == "intent_coverage"
    assert changed["intent_selection_config_unchanged"] is False


def test_old_preparation_collects_as_direct_planning(candidate, tmp_path):
    dataset, _, _ = candidate
    output = tmp_path / "old-output"
    output.mkdir()
    (output / "preparation.json").write_text(json.dumps({"arm": "no-index", "dataset": str(dataset),
        "task_input_sha256": benchmark._task_hashes(dataset / benchmark.TASK)}))
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "no-index")
    (output / "config.json").write_text(json.dumps(config))
    receipt = benchmark.collect(output)
    assert receipt["planning_strategy"] == "direct"
    assert receipt["intent_requirement_contract_cid"] is None
    assert receipt["intent_selection_config_unchanged"] is True


def test_prepare_cli_passes_contract_path_without_running_harbor(candidate, tmp_path, monkeypatch, capsys):
    dataset, path, _ = candidate
    observed = {}
    def prepared(**kwargs):
        observed.update(kwargs)
        return {"prepared": True}
    monkeypatch.setattr(benchmark, "prepare", prepared)
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(dataset),
        "--output", str(tmp_path / "output"), "--archive", str(tmp_path / "archive"),
        "--intent-requirement-contract", str(path)])
    benchmark.main()
    assert observed["intent_requirement_contract"] == path
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_harbor_adapter_transports_contract_and_retains_coverage(candidate, tmp_path, monkeypatch, arm):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    from benchmarks.agent_supervisor.container_coding.terminal_deployment import ROOT

    dataset, path, _ = candidate
    contract = benchmark._load_intent_requirement_contract(path, dataset=dataset)
    coverage = {"schema": "intent-plan-coverage-receipt@1", "accepted": True,
                "contract_cid": cid_for_dag_json(contract), "execution_authority": False,
                "completion_authority": False, "semantic_alignment_verified": False}
    report = {"task_completed": False, "seconds": 1, "phases": {}, "provider_invocations": [],
              "planning": {"planning_strategy": "intent_coverage", "requirement_coverage": coverage}}
    uploads, calls = {}, []
    class Environment:
        async def upload_file(self, source, target):
            uploads[target] = Path(source).read_bytes()
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    before = AsyncMock(return_value={"status": "captured"})
    after = AsyncMock(return_value={"status": "captured"})
    monkeypatch.setattr(adapter, "capture_public_inputs", before)
    monkeypatch.setattr(adapter, "export_public_outputs", after)
    archive = tmp_path / ("archive-" + arm)
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "fixture", "learned_requirements": []}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / ("logs-" + arm), model_name=benchmark.MODEL,
        runtime_archive=str(archive), arm=arm, intent_requirement_contract=contract)
    context = AgentContext()
    asyncio.run(agent.run((dataset / benchmark.TASK / "instruction.md").read_text(), Environment(), context))
    target = ROOT + "/intent-requirements.json"
    assert json.loads(uploads[target]) == contract
    index = calls[0].index("--intent-requirement-contract")
    assert calls[0][index + 1] == target
    assert context.metadata["planning_strategy"] == "intent_coverage"
    assert context.metadata["requirement_coverage"] == coverage
    assert context.metadata["task_completed"] is False
    assert context.metadata["official_reward"] is None
    before.assert_awaited_once()
    after.assert_awaited_once()
