"""Registered comparisons preserve failed slots and never dispatch a provider."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_capsule_budget_matrix as matrix


def inputs(tmp_path):
    dataset, archive = tmp_path / "dataset", tmp_path / "archive"
    dataset.mkdir(); archive.mkdir()
    return dict(dataset=dataset, archive=archive, output=tmp_path / "matrix", task="authored-task")


def test_all_slots_registered_before_preflight_with_balanced_order(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    calls = []
    def prepare(**kwargs):
        registry = json.loads((args["output"] / "matrix.json").read_text())
        assert len(registry["slots"]) == 9
        assert len(calls) == len(list(args["output"].glob("preflight-*.json")))
        calls.append(kwargs)
        return {"prepared": True, "provider_calls": 0}
    monkeypatch.setattr(matrix.benchmark, "prepare", prepare)
    monkeypatch.setattr(matrix.benchmark, "execute", lambda *a, **k: pytest.fail("preparation dispatched"))
    result = matrix.prepare_matrix(**args)
    assert result["all_prepared"] and result["prepared_trials"] == result["registered_trials"] == 9
    assert result["provider_calls"] == result["containers_launched"] == result["trials_executed"] == 0
    registry = json.loads((args["output"] / "matrix.json").read_text())
    for position in range(3):
        assert Counter(row["capsule_budget"] for row in registry["slots"][position::3]) == {
            name: 1 for name in matrix.CAPSULE_BUDGETS}
    assert len({str(row["output"]) for row in calls}) == 9
    assert len({row["resource_profile"] for row in calls}) == 1
    assert {row["planning_mode"] for row in calls} == {"declared-contract@1"}
    assert all(row["arm"] == "full" for row in calls)
    for name in ("legacy", "one", "two"):
        assert registry["collect_argv"].count("--" + name) == 3
    assert all(slot["receipt"] in registry["collect_argv"] for slot in registry["slots"])


def test_preflight_failure_and_unprepared_result_are_retained(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    def prepare(**kwargs):
        if kwargs["capsule_budget"] == "one-capsule@1":
            raise ValueError("private diagnostics never copied into matrix")
        return {"prepared": kwargs["capsule_budget"] == "legacy"}
    monkeypatch.setattr(matrix.benchmark, "prepare", prepare)
    result = matrix.prepare_matrix(**args, repeats=1)
    assert result["prepared_trials"] == 1 and result["registered_trials"] == 3
    assert result["all_prepared"] is False
    assert result["preflights"][1]["error_type"] == "ValueError"
    assert len(list(args["output"].glob("preflight-*.json"))) == 3
    assert "private diagnostics" not in (args["output"] / "preflight.json").read_text()
    assert len(json.loads((args["output"] / "matrix.json").read_text())["slots"]) == 3


@pytest.mark.parametrize("repeats", [0, -1, 11, True, 1.0, "3"])
def test_invalid_repeat_bound_refuses_before_preflight(tmp_path, monkeypatch, repeats):
    args = inputs(tmp_path)
    monkeypatch.setattr(matrix.benchmark, "prepare", lambda **kwargs: pytest.fail("invalid matrix prepared"))
    with pytest.raises(ValueError, match="repeats"):
        matrix.prepare_matrix(**args, repeats=repeats)
    assert not args["output"].exists()


@pytest.mark.parametrize("options", [{"capsule_budget": "legacy"}, {"arm": "no-index"},
    {"output": "/tmp/foreign"}, {"resource_profile": None}, {"unknown": "value"}, []])
def test_matrix_owned_and_unbounded_options_refuse(tmp_path, monkeypatch, options):
    args = inputs(tmp_path)
    monkeypatch.setattr(matrix.benchmark, "prepare", lambda **kwargs: pytest.fail("invalid options prepared"))
    with pytest.raises(ValueError):
        matrix.prepare_matrix(**args, options=options)
    assert not args["output"].exists()


def test_common_path_options_resolved_and_caller_options_preserved(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    profile = tmp_path / "profile.json"; profile.write_text("{}")
    options = {"task_profile": str(profile), "planning_mode": "model"}
    original = deepcopy(options)
    calls = []
    monkeypatch.setattr(matrix.benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    matrix.prepare_matrix(**args, repeats=1, options=options)
    assert options == original
    assert all(row["task_profile"] == profile and isinstance(row["task_profile"], Path) for row in calls)
    assert all(row["planning_mode"] == "model" for row in calls)
    with pytest.raises(ValueError, match="fresh"):
        matrix.prepare_matrix(**args, repeats=1, options=options)
    assert len(calls) == 3


def test_matrix_real_preparation_seals_same_controls_for_every_budget(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
    from benchmarks.agent_supervisor.container_coding.test_terminal_capsule_budget_outer import inventory
    from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls
    from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import SOURCE_PATHS
    import hashlib

    dataset, task = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    files = {row["path"]: row for row in inventory()["files"]}
    checkout = Path(matrix.__file__).resolve().parents[3]
    for name in SOURCE_PATHS:
        raw = (checkout / name).read_bytes()
        files["source/" + name] = dict(path="source/" + name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest = _supervisor_manifest(dict(files=list(files.values()), codex_version=CLI_VERSION,
        archive_sha256=matrix.benchmark._hash(archive / "runtime.tar.gz")))
    (archive / "manifest.json").write_text(json.dumps(manifest))
    commands = []
    monkeypatch.setattr(matrix.benchmark.subprocess, "run", lambda argv, **kwargs:
        commands.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    output = tmp_path / "matrix"
    result = matrix.prepare_matrix(dataset=dataset, archive=archive, output=output, task=task.name, repeats=1)
    assert result["all_prepared"], result
    assert len(commands) == 3 and all("--dry-run" in row for row in commands)
    registry = json.loads((output / "matrix.json").read_text())
    preparations = [json.loads((Path(row["output"]) / "preparation.json").read_text()) for row in registry["slots"]]
    base = dict(status="observed", configuration_unchanged=True, declared=preparations[0]["comparison_controls"])
    for selected in preparations[1:]:
        compared = compare_controls(base, dict(status="observed", configuration_unchanged=True,
                                              declared=selected["comparison_controls"]))
        assert compared["differences"] == ["experimental_controls_sha256.capsule_budget"]
        assert selected["task_input_sha256"] == preparations[0]["task_input_sha256"]
        assert selected["archive_sha256"] == preparations[0]["archive_sha256"]
    assert all(row["provider_calls"] == 0 for row in preparations)
    from benchmarks.agent_supervisor.container_coding import terminal_capsule_budget_comparison as comparison
    grouped = {label: [Path(slot["receipt"]) for slot in registry["slots"]
        if matrix.LABELS[slot["capsule_budget"]] == label] for label in ("legacy", "one", "two")}
    collected = comparison.collect(**grouped)
    assert len(collected["attempts"]) == 3
    assert all(row["status"] == "missing_receipt" and row["frozen"]["verified"]
               for row in collected["attempts"])
    assert collected["matched_official_success"] is False
    for group in collected["budgets"].values():
        assert group["registered_attempts_without_trials"] == 1
        assert group["aggregate"]["tokens"]["total_tokens"] is None
    assert all(value["candidate_minus_legacy_tokens"] is None for value in collected["contrasts"].values())
