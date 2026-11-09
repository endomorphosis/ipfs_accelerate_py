"""Registered comparisons preserve failed slots and never dispatch a provider."""
from collections import Counter
from copy import deepcopy
import hashlib
import io
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
    assert result["prepared_trials"] == result["registered_trials"] == 9
    # A successful return alone cannot qualify missing frozen sidecars.
    assert result["all_prepared"] is False
    match = result["frozen_preparation_match"]
    assert match["matched"] is False and match["registry_unchanged"] is True
    assert match["verified_frozen_preparations"] == match["matching_frozen_preparations"] == 0
    assert [row["ordinal"] for row in match["slots"]] == list(range(1, 10))
    assert all("missing_frozen_preparation_or_configuration" in row["reasons"] for row in match["slots"])
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


def real_preparation_inputs(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
    from benchmarks.agent_supervisor.container_coding.test_terminal_capsule_budget_outer import inventory
    from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
    from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import SOURCE_PATHS

    dataset, task = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    files = {row["path"]: row for row in inventory()["files"]}
    checkout = Path(matrix.__file__).resolve().parents[3]
    for name in SOURCE_PATHS:
        raw = (checkout / name).read_bytes()
        files["source/" + name] = dict(path="source/" + name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    for name, path in matrix.benchmark._timeout_runtime_sources().items():
        raw = path.read_bytes()
        files[name] = dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest = _supervisor_manifest(dict(schema="terminal-supervisor-runtime-archive@1",
        files=list(files.values()), codex_version=CLI_VERSION))
    # Materialize the declared source bytes for the real archive admission
    # guard. Only Harbor's dry-run process is authored; source guards stay live.
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as held:
        for row in manifest["files"]:
            raw = (checkout / row["path"].removeprefix("source/")).read_bytes()
            member = tarfile.TarInfo(row["path"])
            member.size, member.mode = len(raw), 0o644
            held.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = matrix.benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(manifest))
    commands = []
    monkeypatch.setattr(matrix.benchmark.subprocess, "run", lambda argv, **kwargs:
        commands.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    return dict(dataset=dataset, archive=archive, output=tmp_path / "matrix", task=task.name,
                repeats=1, options={"planning_mode": "model"}), commands


def test_matrix_real_preparation_seals_same_controls_for_every_budget(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls

    args, commands = real_preparation_inputs(tmp_path, monkeypatch)
    output = args["output"]
    result = matrix.prepare_matrix(**args)
    assert result["all_prepared"], result
    assert len(commands) == 3 and all("--dry-run" in row for row in commands)
    registry = json.loads((output / "matrix.json").read_text())
    match = result["frozen_preparation_match"]
    assert match["schema"] == "terminal-capsule-budget-preflight-match@1"
    assert match["matched"] is True and match["registry_unchanged"] is True
    assert match["verified_frozen_preparations"] == match["matching_frozen_preparations"] == 3
    assert match["reasons"] == []
    assert match["registry_sha256"] == match["observed_registry_sha256"] == result["registry_sha256"]
    assert result["registry_sha256"] == hashlib.sha256((output / "matrix.json").read_bytes()).hexdigest()
    assert match["preregistration_verified"] is match["runtime_enforcement_verified"] is False
    assert match["benchmark_advantage_claimed"] is False
    assert registry["common_options"]["planning_mode"] == "model"
    for slot, observation in zip(registry["slots"], match["slots"]):
        assert observation["ordinal"] == slot["ordinal"]
        assert observation["capsule_budget"] == slot["capsule_budget"]
        assert observation["verified"] is observation["matches"] is True
        assert observation["reasons"] == []
        assert observation["preparation_sha256"] == hashlib.sha256(
            (Path(slot["output"]) / "preparation.json").read_bytes()).hexdigest()
        assert observation["configuration_sha256"] == hashlib.sha256(
            (Path(slot["output"]) / "config.json").read_bytes()).hexdigest()
    preparations = [json.loads((Path(row["output"]) / "preparation.json").read_text()) for row in registry["slots"]]
    base = dict(status="observed", configuration_unchanged=True, declared=preparations[0]["comparison_controls"])
    for selected in preparations[1:]:
        compared = compare_controls(base, dict(status="observed", configuration_unchanged=True,
                                              declared=selected["comparison_controls"]))
        assert compared["differences"] == ["experimental_controls_sha256.capsule_budget"]
        assert selected["task_input_sha256"] == preparations[0]["task_input_sha256"]
        assert selected["archive_sha256"] == preparations[0]["archive_sha256"]
    assert all(row["provider_calls"] == 0 for row in preparations)
    timeout_pins = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in matrix.benchmark._timeout_runtime_sources().items()}
    assert all(row["runtime_timeout_source_sha256"] == timeout_pins for row in preparations)
    from benchmarks.agent_supervisor.container_coding import terminal_capsule_budget_comparison as comparison
    grouped = {label: [Path(slot["receipt"]) for slot in registry["slots"]
        if matrix.LABELS[slot["capsule_budget"]] == label] for label in ("legacy", "one", "two")}
    collected = comparison.collect(**grouped)
    assert match["comparison_owner_sha256"] == collected["collector_sha256"]
    assert len(collected["attempts"]) == 3
    assert all(row["status"] == "missing_receipt" and row["frozen"]["verified"]
               for row in collected["attempts"])
    assert collected["matched_official_success"] is False
    for group in collected["budgets"].values():
        assert group["registered_attempts_without_trials"] == 1
        assert group["aggregate"]["tokens"]["total_tokens"] is None
    assert all(value["candidate_minus_legacy_tokens"] is None for value in collected["contrasts"].values())


@pytest.mark.parametrize("mutation,verified,reason", [
    ("configuration_tamper", False, "frozen_configuration_changed"),
    ("asset_difference", True, "frozen_runtime_or_assets_differ"),
    ("planning_difference", True, "unexpected_declared_control_difference"),
    ("missing_configuration", False, "missing_frozen_preparation_or_configuration"),
])
def test_successful_slots_require_matching_frozen_preparations(tmp_path, monkeypatch, mutation, verified, reason):
    args, commands = real_preparation_inputs(tmp_path, monkeypatch)
    original_prepare = matrix.benchmark.prepare

    def prepare(**kwargs):
        selected = kwargs["capsule_budget"] == "one-capsule@1"
        if selected and mutation == "planning_difference":
            # Both preparations are individually valid, but their common
            # planning policy differs beyond the registered capsule treatment.
            kwargs["planning_mode"] = "declared-contract@1"
        prepared = original_prepare(**kwargs)
        if selected:
            path = kwargs["output"]
            if mutation == "configuration_tamper":
                config = json.loads((path / "config.json").read_text())
                config["n_concurrent_trials"] = 2
                (path / "config.json").write_text(json.dumps(config))
            elif mutation == "asset_difference":
                sidecar = json.loads((path / "preparation.json").read_text())
                sidecar["archive_sha256"] = "f" * 64
                (path / "preparation.json").write_text(json.dumps(sidecar))
            elif mutation == "missing_configuration":
                (path / "config.json").unlink()
        return prepared

    monkeypatch.setattr(matrix.benchmark, "prepare", prepare)
    result = matrix.prepare_matrix(**args)
    assert len(commands) == result["prepared_trials"] == result["registered_trials"] == 3
    assert all(row["prepared"] for row in result["preflights"])
    assert result["all_prepared"] is False
    match = result["frozen_preparation_match"]
    assert match["matched"] is False and match["registry_unchanged"] is True
    failed = next(row for row in match["slots"] if row["capsule_budget"] == "one-capsule@1")
    assert failed["verified"] is verified and failed["matches"] is False
    assert reason in failed["reasons"]
    assert [row["ordinal"] for row in match["slots"]] == [1, 2, 3]
    assert len(list(args["output"].glob("preflight-*.json"))) == 3
    assert result["provider_calls"] == result["containers_launched"] == result["trials_executed"] == 0


def test_malformed_sidecar_retains_all_successful_preflights_without_copying_content(tmp_path, monkeypatch):
    args, commands = real_preparation_inputs(tmp_path, monkeypatch)
    original_prepare = matrix.benchmark.prepare

    def prepare(**kwargs):
        prepared = original_prepare(**kwargs)
        if kwargs["capsule_budget"] == "one-capsule@1":
            (kwargs["output"] / "config.json").write_text(
                '{"private diagnostic": "never export", "private diagnostic": 2}')
        return prepared

    monkeypatch.setattr(matrix.benchmark, "prepare", prepare)
    result = matrix.prepare_matrix(**args)
    assert len(commands) == result["prepared_trials"] == result["registered_trials"] == 3
    assert result["all_prepared"] is False
    match = result["frozen_preparation_match"]
    assert match["matched"] is False and match["error_type"] == "ValueError"
    assert match["reasons"] == ["frozen_preflight_comparison_unavailable"]
    assert [row["ordinal"] for row in match["slots"]] == [1, 2, 3]
    assert all(row["matches"] is False for row in match["slots"])
    assert len(list(args["output"].glob("preflight-*.json"))) == 3
    assert "private diagnostic" not in (args["output"] / "preflight.json").read_text()
    assert "never export" not in json.dumps(result)


@pytest.mark.parametrize("mutation", ["edit", "remove"])
def test_changed_registration_blocks_readiness_and_preserves_original_population(tmp_path, monkeypatch, mutation):
    args, commands = real_preparation_inputs(tmp_path, monkeypatch)
    original_prepare = matrix.benchmark.prepare
    initial = {}

    def prepare(**kwargs):
        registry_path = args["output"] / "matrix.json"
        if not initial:
            initial["sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        prepared = original_prepare(**kwargs)
        if kwargs["capsule_budget"] == "two-capsules@1":
            if mutation == "edit":
                registry = json.loads(registry_path.read_text())
                registry["slots"].pop()
                registry_path.write_text(json.dumps(registry))
            else:
                registry_path.unlink()
            initial["mutation_applied"] = True
        return prepared

    monkeypatch.setattr(matrix.benchmark, "prepare", prepare)
    result = matrix.prepare_matrix(**args)
    assert len(commands) == result["prepared_trials"] == result["registered_trials"] == 3
    assert initial.get("mutation_applied") is True
    assert result["all_prepared"] is False
    assert result["registry_sha256"] == initial["sha256"]
    match = result["frozen_preparation_match"]
    assert match["registry_unchanged"] is match["matched"] is False
    assert match["matching_frozen_preparations"] == 3
    assert match["reasons"] == ["registered_matrix_changed" if mutation == "edit"
                                else "registered_matrix_unavailable"]
    assert [row["ordinal"] for row in match["slots"]] == [row["ordinal"] for row in result["preflights"]] == [1, 2, 3]
    assert len(list(args["output"].glob("preflight-*.json"))) == 3
    assert (args["output"] / "preflight.json").is_file()
