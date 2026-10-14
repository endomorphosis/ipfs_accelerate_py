"""Authored public-profile fixtures exercise the complete offline experiment."""
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_view_experiment as experiment


@pytest.fixture
def public_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    task = dataset / "authored-public-task"
    (task / "environment" / "source").mkdir(parents=True)
    source = b"# Canonical retained comment.\ndef calculate(value):\n    # Readable context.\n    return value + 1\n\nUNCHANGED = 'tail'\n"
    metadata = {"instruction.md": b"Read the public inputs and write output.json.\n",
        "task.toml": b"[agent]\ntimeout_sec=60\n", "environment/Dockerfile": b"FROM python:3.12-slim\nCOPY source/ /app/\n"}
    for name, raw in metadata.items():
        (task / name).write_bytes(raw)
    inputs = {"source/example.py": source, "source/__init__.py": b"", "source/input.json": b'{"value":5}\n'}
    for name, raw in inputs.items():
        (task / "environment" / name).write_bytes(raw)
    # Sentinel paths must not be traversed or opened by the experiment.
    (task / "tests").symlink_to("/unavailable/private-verifier")
    (task / "solution").symlink_to("/unavailable/private-solution")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema": "terminal-reviewed-public-profile-catalog@1", "dataset_commit": "authored-fixture",
        "tasks": {"authored-public-task": {"public_metadata": {name: hashlib.sha256(raw).hexdigest() for name, raw in metadata.items()},
            "inputs": [{"environment_path": name, "path": name.removeprefix("source/"),
                "media_type": "application/json" if name.endswith(".json") else None, "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in inputs.items()],
            "outputs": [{"effect": "create", "path": "output.json", "media_type": "application/json"}]}}}))
    monkeypatch.setattr(experiment.catalog, "CATALOG", catalog_path)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (lambda text: len(text),
        {"available": True, "scope": "authored character-counter test double, not tokenizer/provider usage"}))
    return dataset, task, catalog_path


def test_complete_public_copy_read_edit_refresh_and_authored_behavior(public_dataset, tmp_path):
    dataset, task, _ = public_dataset
    original = (task / "environment/source/example.py").read_bytes()
    output = tmp_path / "qualification"
    result = experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert result["qualified"] is True
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    assert result["original_dataset_unchanged"] is True
    assert result["authored_behavior"]["behavior_change_verified"] is True
    row = result["tasks"][0]
    assert row["hidden_verifier_or_solution_read"] is False
    assert row["paired_final_canonical_bytes_identical"] is True
    assert row["unsupported_input_paths"] == ["input.json"]
    assert (task / "environment/source/example.py").read_bytes() == original
    for fmt in ("raw", "compact"):
        assert all(row[fmt]["checks"].values())
        assert row[fmt]["edit_cycles"] == 1
        assert row[fmt]["edit_cycle_protocol"]["protocol_messages"] == 8
        assert row[fmt]["edit_cycle_protocol"]["local_tokenizer_proxy_tokens"] > 0
        protocol = json.loads((output / "authored-public-task" / fmt / "protocol.json").read_text())
        assert protocol["cycles"][0]["stale_error"] == "stale_source"
        assert protocol["cycles"][0]["replace"]["refreshed_view"]["format"] == fmt
        assert row[fmt]["complete_qualification_protocol"] == experiment._metrics(protocol["complete_messages"], len)
        assert row[fmt]["complete_qualification_protocol"]["protocol_messages"] == 16
        assert b"pass" in (output / "authored-public-task" / fmt / "workspace/example.py").read_bytes()
        assert (output / "authored-public-task/canonical-before/example.py").read_bytes() == original
    assert json.loads((output / "qualification.json").read_text()) == result


def test_input_hash_drift_rejected_before_output(public_dataset, tmp_path):
    dataset, task, _ = public_dataset
    (task / "environment/source/example.py").write_bytes(b"changed = True\n")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="changed"):
        experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert not output.exists()


def test_new_output_cannot_be_under_dataset(public_dataset):
    dataset, _, _ = public_dataset
    output = dataset / "new-experiment"
    with pytest.raises(ValueError, match="outside"):
        experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert not output.exists()


def test_existing_output_is_never_overwritten(public_dataset, tmp_path):
    dataset, _, _ = public_dataset
    marker = tmp_path / "existing"
    marker.mkdir()
    (marker / "keep").write_text("keep")
    with pytest.raises(ValueError, match="new canonical"):
        experiment.run_experiment(dataset=dataset, output=marker, tasks=["authored-public-task"])
    assert (marker / "keep").read_text() == "keep"


@pytest.mark.parametrize("tasks", [[], ["authored-public-task"] * 2, [str(i) for i in range(9)]])
def test_task_selection_is_bounded_and_unique(public_dataset, tmp_path, tasks):
    with pytest.raises(ValueError, match="bounded"):
        experiment.run_experiment(dataset=public_dataset[0], output=tmp_path / "out", tasks=tasks)


def test_no_python_inputs_are_reported_without_claiming_qualification(public_dataset, tmp_path):
    dataset, _, catalog_path = public_dataset
    catalog = json.loads(catalog_path.read_text())
    row = catalog["tasks"]["authored-public-task"]
    row["inputs"] = [entry for entry in row["inputs"] if entry["path"].endswith(".json")]
    catalog_path.write_text(json.dumps(catalog))
    result = experiment.run_experiment(dataset=dataset, output=tmp_path / "out", tasks=["authored-public-task"])
    assert result["qualified"] is False
    assert result["tasks"][0]["disposition"] == "no_supported_python_input"


def test_absent_tokenizer_remains_null_instead_of_zero(public_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(dataset=public_dataset[0], output=tmp_path / "out", tasks=["authored-public-task"])
    assert result["qualified"] is True
    assert result["tasks"][0]["compact"]["edit_cycle_protocol"]["local_tokenizer_proxy_tokens"] is None


def test_invalid_tokenizer_cache_rejected_without_download(tmp_path):
    cache = tmp_path / "ranks"
    cache.write_text("invalid ranks")
    with pytest.raises(ValueError, match="pinned"):
        experiment.local_token_counter(cache)


def test_metric_counts_full_json_wrapper_and_newline():
    value = {"source": "é\n", "metadata": "retained"}
    encoded = experiment._render(value)
    result = experiment._metrics([value], lambda text: len(text))
    assert result["rendered_json_bytes"] == len(encoded.encode("utf-8"))
    assert result["local_tokenizer_proxy_tokens"] == len(encoded)
    assert result["rendered_json_bytes"] > len(value["source"].encode("utf-8"))


def test_cli_writes_bounded_summary(public_dataset, tmp_path, capsys):
    dataset, _, _ = public_dataset
    assert experiment.main(["--dataset", str(dataset), "--output", str(tmp_path / "cli"), "--task", "authored-public-task"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["qualified"] is True
    assert result["provider_calls"] == 0
    assert result["tasks"] == ["authored-public-task"]
