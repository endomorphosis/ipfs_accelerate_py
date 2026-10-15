"""Authored staged tokenizer configuration; no Harbor trials or model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_metadata_tokenizer_bundle as bundle
from benchmarks.agent_supervisor.container_coding import terminal_semantic_metadata_policy as policy
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
from test.api.test_local_prompt_tokenizer import make_binding

MODE = "common-bindings@1"


def owner(tmp_path, *, special_tokens=False):
    tmp_path.mkdir()
    return make_binding(tmp_path, provider="codex_cli", model=benchmark.MODEL,
        add_special_tokens=special_tokens)


def staged(tmp_path):
    artifact, expected, _ = owner(tmp_path / "owner")
    selection = bundle.stage_metadata_tokenizer_bundle(artifact=artifact, expected_sha256=expected,
        output=tmp_path / "bundle", provider="codex_cli", model=benchmark.MODEL)
    return selection


def configured(tmp_path, selection=None, **options):
    return benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        semantic_metadata_view=MODE, semantic_metadata_tokenizer=selection, **options)


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort=benchmark.REASONING,
        cli_version=benchmark.CLI_VERSION)


def read(selection):
    return bundle.read_metadata_tokenizer_bundle(selection, arm="full", provider="codex_cli",
        model=benchmark.MODEL, semantic_metadata_view=MODE)


def prepared_state(selection, config):
    return dict(arm="full", semantic_metadata_view=MODE, semantic_metadata_tokenizer=selection,
        semantic_metadata_tokenizer_bundle=read(selection).observation(), comparison_controls=seal(config))


def inventory():
    root = Path(benchmark.__file__).resolve().parents[3]
    paths = sorted(set(policy.SOURCE_PATHS) | set(bundle.SOURCE_PATHS))
    return {"files": [{"path": "source/" + path,
        "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
        "bytes": (root / path).stat().st_size} for path in paths]}


def archive_fixture(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"), "codex_version": benchmark.CLI_VERSION})))
    return archive


def dry_run(monkeypatch):
    calls = []
    proxy = SimpleNamespace(**vars(subprocess))

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored Harbor dry run", stderr="")

    proxy.run = run
    monkeypatch.setattr(benchmark, "subprocess", proxy)
    return calls


def test_none_preserves_existing_configuration_and_controls(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    original = benchmark.config_for(*args)
    assert benchmark.config_for(*args, semantic_metadata_tokenizer=None) == original
    assert "semantic_metadata_tokenizer" not in original["agents"][0]["kwargs"]
    assert seal(original)["schema"] == controls.SCHEMA


def test_selected_configuration_binds_all_descriptor_asset_manifest_and_policy_pins(tmp_path):
    selection = staged(tmp_path)
    config = configured(tmp_path, selection)
    assert config["agents"][0]["kwargs"]["semantic_metadata_tokenizer"] == selection
    declared = seal(config)
    observed = read(selection).observation()
    assert declared["schema"] == controls.METADATA_TOKENIZER_SCHEMA
    assert controls.validate_controls(declared)
    assert declared["experimental_controls_sha256"]["semantic_metadata_tokenizer"] == controls._digest(observed)
    assert all(observed[key] for key in ("original_descriptor_sha256", "descriptor_sha256",
        "asset_sha256", "manifest_sha256", "provider", "model", "tokenizer_id", "special_token_policy"))
    config["agents"][0]["kwargs"]["semantic_metadata_tokenizer"]["manifest_sha256"] = "0" * 64
    assert selection["manifest_sha256"] != "0" * 64  # Detached control.


def test_combined_experiments_keep_every_control_under_new_schema(tmp_path):
    selected = configured(tmp_path, staged(tmp_path), capsule_budget="one-capsule@1",
        planning_mode="declared-contract@1", numerical_formalization_profile="python-numerical-control-structure@3",
        source_view_profile="identity@1", source_edit_mode="mapped-span@2")
    declared = seal(selected)
    assert controls.validate_controls(declared)
    assert set(declared["experimental_controls_sha256"]) == {"semantic_metadata_tokenizer", "capsule_budget",
        "planning_mode", "numerical_formalization_profile", "source_edit_mode"}


def test_same_content_in_another_staging_directory_preserves_experimental_identity(tmp_path):
    artifact, expected, _ = owner(tmp_path / "owner")
    first, second = [bundle.stage_metadata_tokenizer_bundle(artifact=artifact, expected_sha256=expected,
        output=tmp_path / name, provider="codex_cli", model=benchmark.MODEL) for name in ("first", "second")]
    left, right = [seal(configured(tmp_path, selection)) for selection in (first, second)]
    assert left["configuration_sha256"] != right["configuration_sha256"]
    assert left["experimental_controls_sha256"] == right["experimental_controls_sha256"]
    observed = lambda declared: dict(status="observed", configuration_unchanged=True, declared=declared)
    assert controls.compare_controls(observed(left), observed(right))["matches"] is True


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool", "schema", "digest"])
def test_controls_refuse_malformed_or_resealed_old_schema(tmp_path, mutation):
    record = seal(configured(tmp_path, staged(tmp_path)))
    if mutation == "missing":
        record["experimental_controls_sha256"].pop("semantic_metadata_tokenizer")
    elif mutation == "extra":
        record["experimental_controls_sha256"]["other"] = "a" * 64
    elif mutation == "bool":
        record["experimental_controls_sha256"]["semantic_metadata_tokenizer"] = True
    elif mutation == "schema":
        record["schema"] = controls.CAPSULE_BUDGET_SCHEMA
    else:
        record["experimental_controls_sha256"]["semantic_metadata_tokenizer"] = "f" * 64
    assert controls.validate_controls(record) is False


@pytest.mark.parametrize("mutation", ["manifest", "provider", "unknown", "null"])
def test_configuration_rejects_unbound_selection_before_task_config(tmp_path, monkeypatch, mutation):
    selection = staged(tmp_path)
    if mutation == "manifest":
        selection["manifest_sha256"] = "0" * 64
    elif mutation == "provider":
        selection["provider"] = "grok_cli"
    elif mutation == "unknown":
        selection["descriptor_path"] = "/foreign/descriptor.json"
    else:
        config = configured(tmp_path)
        config["agents"][0]["kwargs"]["semantic_metadata_tokenizer"] = None
        with pytest.raises(ValueError):
            seal(config)
        return
    monkeypatch.setattr(benchmark, "baseline_config", lambda *args, **kwargs: pytest.fail("task config accessed"))
    with pytest.raises(ValueError):
        configured(tmp_path, selection)


@pytest.mark.parametrize("artifact,expected,options", [
    (None, "a" * 64, {}), (Path("/missing/owner.json"), None, {}),
    (Path("/missing/owner.json"), "bad", {}),
    (Path("/missing/owner.json"), "a" * 64, {"semantic_metadata_view": "legacy"}),
    (Path("/missing/owner.json"), "a" * 64, {"arm": "no-index"}),
])
def test_invalid_prepare_pair_or_route_refuses_before_task_reads(tmp_path, monkeypatch, artifact, expected, options):
    monkeypatch.setattr(benchmark, "_task_path", lambda *args: pytest.fail("task read attempted"))
    selected = dict(arm="full", semantic_metadata_view=MODE, **{})
    selected.update(options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing-dataset", output=tmp_path / "out",
            archive=tmp_path / "missing-archive", semantic_metadata_tokenizer_artifact=artifact,
            semantic_metadata_tokenizer_sha256=expected, **selected)
    assert not (tmp_path / "out").exists()


def test_invalid_descriptor_pin_refuses_before_dataset_reads_and_preserves_fresh_output(tmp_path, monkeypatch):
    artifact, _, _ = owner(tmp_path / "owner")
    monkeypatch.setattr(benchmark, "_task_path", lambda *args: pytest.fail("task read attempted"))
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing-dataset", output=tmp_path / "out",
            archive=tmp_path / "missing-archive", arm="full", semantic_metadata_view=MODE,
            semantic_metadata_tokenizer_artifact=artifact, semantic_metadata_tokenizer_sha256="0" * 64)
    assert (tmp_path / "out").is_dir()
    assert not (tmp_path / "out/config.json").exists()


def test_prepare_stages_original_and_relocated_bytes_without_reading_model_or_provider(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = archive_fixture(tmp_path)
    artifact, expected, _ = owner(tmp_path / "owner")
    calls = dry_run(monkeypatch)
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", semantic_metadata_view=MODE, semantic_metadata_tokenizer_artifact=artifact,
        semantic_metadata_tokenizer_sha256=expected)
    selection = result["semantic_metadata_tokenizer"]
    assert Path(selection["bundle_path"]).parent == tmp_path / "out"
    assert result["semantic_metadata_tokenizer_bundle"]["original_descriptor_sha256"] == expected
    assert result["semantic_metadata_tokenizer_bundle"]["tokenizer_runtime_loaded"] is False
    assert result["comparison_controls"]["schema"] == controls.METADATA_TOKENIZER_SCHEMA
    assert result["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert config["agents"][0]["kwargs"]["semantic_metadata_tokenizer"] == selection
    collected = benchmark.collect(tmp_path / "out")
    assert collected["semantic_metadata_tokenizer_config_unchanged"] is True
    assert collected["complete_single_trial_receipt"] is False


@pytest.mark.parametrize("mutation", ["prepared_missing", "configured_missing", "both_missing", "observation",
                                      "boolean_observation", "asset"])
def test_changed_or_removed_selection_cannot_start_task_lookup(tmp_path, monkeypatch, mutation):
    selection = staged(tmp_path)
    config = configured(tmp_path, selection)
    prepared = prepared_state(selection, config)
    if mutation == "prepared_missing":
        prepared.pop("semantic_metadata_tokenizer")
    elif mutation == "configured_missing":
        config["agents"][0]["kwargs"].pop("semantic_metadata_tokenizer")
    elif mutation == "both_missing":
        prepared.pop("semantic_metadata_tokenizer")
        prepared.pop("semantic_metadata_tokenizer_bundle")
        config["agents"][0]["kwargs"].pop("semantic_metadata_tokenizer")
    elif mutation == "observation":
        prepared["semantic_metadata_tokenizer_bundle"]["asset_sha256"] = "0" * 64
    elif mutation == "boolean_observation":
        prepared["semantic_metadata_tokenizer_bundle"]["completion_authority"] = 0
    else:
        target = Path(selection["bundle_path"]) / "tokenizer.json"
        target.chmod(0o600)
        target.write_bytes(b"changed asset")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "preparation.json").write_text(json.dumps(prepared))
    (output / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(benchmark, "_prepared_task", lambda *args, **kwargs: pytest.fail("task lookup attempted"))
    with pytest.raises(ValueError, match="metadata tokenizer selection or bundle changed"):
        benchmark._execute_once(output)
    observed = benchmark._metadata_tokenizer_prepared_state(prepared, config)
    assert observed[2:] == (False, True)
    assert not (output / "invocation.json").exists()


def test_mutated_bundle_cannot_reseal_frozen_controls(tmp_path):
    selection = staged(tmp_path)
    config = configured(tmp_path, selection)
    declared = seal(config)
    target = Path(selection["bundle_path"]) / "descriptor.json"
    target.chmod(0o600)
    target.write_bytes(b"changed descriptor")
    observed = controls.observe_controls({"comparison_controls": declared}, config,
        current_task_hashes={"instruction.md": "a" * 64})
    assert observed["configuration_unchanged"] is False
    assert observed["declared"] == declared


def runtime_report(selection):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import load_semantic_metadata_tokenizer

    snapshot = read(selection)
    _, _, source = load_semantic_metadata_tokenizer(artifact=Path(selection["bundle_path"]) / "descriptor.json",
        expected_sha256=snapshot.manifest()["descriptor_sha256"], provider="codex_cli", model=benchmark.MODEL)
    row = dict(schema="router-implementation-invocation@1", invocation_id="authored-coding-call", phase="coding",
        provider="codex_cli", model=benchmark.MODEL, semantic_metadata_tokenizer_source=source,
        semantic_metadata_tokenizer_dispatch_rechecked=True)
    return dict(provider_invocations=[row])


@pytest.mark.parametrize("mutation", [None, "original_descriptor", "missing_source", "missing_recheck", "bool_recheck",
    "route", "policy", "authority_lookalike", "extra", "unknown_calls", "duplicate_invocation"])
def test_runtime_provenance_joins_relocated_descriptor_and_preserves_scope(tmp_path, mutation):
    selection = staged(tmp_path)
    snapshot = read(selection)
    report = runtime_report(selection)
    row = report["provider_invocations"][0]
    if mutation == "original_descriptor":
        row["semantic_metadata_tokenizer_source"]["descriptor_sha256"] = snapshot.manifest()["original_descriptor_sha256"]
    elif mutation == "missing_source":
        row.pop("semantic_metadata_tokenizer_source")
    elif mutation == "missing_recheck":
        row.pop("semantic_metadata_tokenizer_dispatch_rechecked")
    elif mutation == "bool_recheck":
        row["semantic_metadata_tokenizer_dispatch_rechecked"] = 1
    elif mutation == "route":
        row["model"] = "unselected-model"
    elif mutation == "policy":
        row["semantic_metadata_tokenizer_source"]["special_token_policy"] = "tokenizers.encode:add_special_tokens=true"
    elif mutation == "authority_lookalike":
        row["semantic_metadata_tokenizer_source"]["completion_authority"] = 0
    elif mutation == "extra":
        row["semantic_metadata_tokenizer_source"]["descriptor_path"] = "/foreign/descriptor.json"
    elif mutation == "unknown_calls":
        report["unreceipted_provider_attempt"] = None
    elif mutation == "duplicate_invocation":
        report["provider_invocations"].append(deepcopy(row))
    observed = benchmark._metadata_tokenizer_dispatch_observation(report, snapshot)
    assert observed["all_recorded_provenance_matches_retained_bundle"] is (mutation is None)
    assert observed["tokenizer_runtime_replayed"] is False
    assert observed["provider_execution_authenticated"] is observed["provider_ingestion_verified"] is False
    assert observed["billing_total_verified"] is observed["token_savings_established"] is False


@pytest.mark.parametrize("operation", ["execute", "collect"])
def test_cli_cannot_replace_prepared_tokenizer_selection(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
        "--semantic-metadata-tokenizer-artifact", "/owner/descriptor.json",
        "--semantic-metadata-tokenizer-sha256", "a" * 64])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
