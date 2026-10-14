"""Public-only fixtures exercise the self-contained repository experiment."""
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_view_experiment as experiment


@pytest.fixture
def public_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    task = dataset / "authored-public-task"
    (task / "environment" / "source" / "nested").mkdir(parents=True)
    metadata = {"instruction.md": b"Read the public input and write output.json.\n",
        "task.toml": b"[agent]\ntimeout_sec=60\n",
        "environment/Dockerfile": b"FROM python:3.12-slim\nCOPY source/ /app/\n"}
    for path, raw in metadata.items():
        (task / path).write_bytes(raw)
    inputs = {
        "source/example.py": b"# Retain original canonical formatting and comments.\ndef calculate(value):\n    # This fixture never solves a benchmark.\n    return value + 1\n\nUNCHANGED = 'tail'\n",
        "source/nested/second.py": b"# A second independently edited public module.\ndef second(value):\n    # A comment.\n    return value * 2\n",
        "source/__init__.py": b"",
        "source/input.json": b'{"value":5}\n',
        "source/data.bin": b"\x00\xff\x10\x80",
    }
    for path, raw in inputs.items():
        (task / "environment" / path).write_bytes(raw)
    # These unreviewed trees cannot be resolved and must never be opened.
    (task / "tests").symlink_to("/unavailable/private-verifier")
    (task / "solution").symlink_to("/unavailable/private-solution")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema": "terminal-reviewed-public-profile-catalog@1",
        "dataset_commit": "authored-fixture", "tasks": {"authored-public-task": {
            "public_metadata": {path: hashlib.sha256(raw).hexdigest() for path, raw in metadata.items()},
            "inputs": [{"environment_path": path, "path": path.removeprefix("source/"),
                "media_type": "application/json" if path.endswith(".json") else None,
                "sha256": hashlib.sha256(raw).hexdigest()} for path, raw in inputs.items()],
            "outputs": [{"effect": "create", "path": "output.json", "media_type": "application/json"}],
        }}}))
    monkeypatch.setattr(experiment.catalog, "CATALOG", catalog)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (len,
        {"available": True, "scope": "authored character counter; not tokenizer or provider usage"}))
    return dataset, task, catalog, inputs


def test_complete_public_tree_restore_compact_reads_edits_and_refresh(public_dataset, tmp_path):
    dataset, task, _, inputs = public_dataset
    output = tmp_path / "qualification"
    result = experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert result["qualified"] is True
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    assert result["original_dataset_unchanged"] is True
    assert result["authored_behavior"]["qualified"] is True
    assert all(result["authored_behavior"]["checks"].values())
    row = result["tasks"][0]
    assert row["hidden_verifier_or_solution_read"] is row["benchmark_source_executed"] is False
    assert row["file_count"] == len(inputs)
    assert row["transformed_paths"] == ["example.py", "nested/second.py"]
    assert row["passthrough_paths"] == ["__init__.py", "data.bin", "input.json"]
    assert row["compact_reads"] == {"requested_files": 5, "compacted_python_content_delivered": 2,
        "passthrough_content_delivered": 3}
    assert all(row["checks"].values())
    assert not (output / "authored-public-task/disposable-source").exists()
    assert row["original_file_bytes"] == sum(len(raw) for raw in inputs.values())
    assert row["compact_file_bytes"] < row["original_file_bytes"]
    assert row["host_storage"]["sent_in_read_protocol"] is False
    assert row["host_storage"]["package_bytes"] > row["original_file_bytes"]
    assert row["host_storage"]["package_bytes"] == row["host_storage"]["manifest_mapping_bytes"] + row["host_storage"]["unique_blob_bytes"]
    for path, raw in inputs.items():
        assert (task / "environment" / path).read_bytes() == raw
        assert (output / "authored-public-task/restored-original" / path.removeprefix("source/")).read_bytes() == raw
    for representation in ("original", "compact"):
        arm = row["protocol"][representation]
        assert arm["read_protocol"]["protocol_messages"] == len(inputs) * 2
        assert arm["edits"]["edit_count"] == 2
        assert arm["edits"]["stale_error"] == "stale_source"
        assert all(arm["edits"]["checks"].values())
        transcript = json.loads((output / "authored-public-task" / representation / "protocol.json").read_text())
        assert arm["read_protocol"] == experiment._metrics(transcript["read_messages"], len)
        assert arm["read_edit_refresh_protocol"] == experiment._metrics(transcript["read_messages"] + transcript["edit_messages"], len)
        assert arm["read_edit_refresh_protocol"]["protocol_messages"] == len(inputs) * 2 + 6
        responses = transcript["read_messages"][1::2]
        assert any(response["encoding"] == "base64" for response in responses)
    assert json.loads((output / "qualification.json").read_text()) == result


def test_changed_public_input_refused_before_output(public_dataset, tmp_path):
    dataset, task, _, _ = public_dataset
    (task / "environment/source/example.py").write_bytes(b"changed = True\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="changed"):
        experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert not output.exists()


def test_output_cannot_be_inside_dataset(public_dataset):
    dataset = public_dataset[0]
    output = dataset / "out"
    with pytest.raises(ValueError, match="outside"):
        experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert not output.exists()


def test_existing_output_is_never_overwritten(public_dataset, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(ValueError, match="new canonical"):
        experiment.run_experiment(dataset=public_dataset[0], output=output, tasks=["authored-public-task"])
    assert (output / "keep").read_text() == "keep"


@pytest.mark.parametrize("tasks", [[], ["authored-public-task"] * 2, [str(n) for n in range(9)]])
def test_task_selection_bounded_and_unique(public_dataset, tmp_path, tasks):
    with pytest.raises(ValueError, match="bounded"):
        experiment.run_experiment(dataset=public_dataset[0], output=tmp_path / "out", tasks=tasks)


def test_only_passthrough_inputs_still_roundtrip_without_minification_claim(public_dataset, tmp_path):
    dataset, _, catalog, _ = public_dataset
    value = json.loads(catalog.read_text())
    row = value["tasks"]["authored-public-task"]
    row["inputs"] = [item for item in row["inputs"] if item["path"] in {"data.bin", "input.json"}]
    catalog.write_text(json.dumps(value))
    result = experiment.run_experiment(dataset=dataset, output=tmp_path / "out", tasks=["authored-public-task"])
    assert result["qualified"] is True
    row = result["tasks"][0]
    assert row["disposition"] == "roundtrip_passthrough_only"
    assert row["transformed_paths"] == []
    assert row["original_file_bytes"] == row["compact_file_bytes"]
    for arm in row["protocol"].values():
        assert arm["edits"]["edit_count"] == 0
        assert arm["read_protocol"] == arm["read_edit_refresh_protocol"]


def test_absent_tokenizer_metrics_are_null(public_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(dataset=public_dataset[0], output=tmp_path / "out", tasks=["authored-public-task"])
    assert result["qualified"] is True
    for arm in result["tasks"][0]["protocol"].values():
        assert arm["read_protocol"]["local_tokenizer_proxy_tokens"] is None
        assert arm["read_edit_refresh_protocol"]["local_tokenizer_proxy_tokens"] is None


def test_corruption_failures_are_required_for_qualification(public_dataset, tmp_path, monkeypatch):
    real = experiment._refusals

    def failed_qualification(**kwargs):
        result = real(**kwargs)
        result["damaged_blob_rejected"] = False
        return result

    monkeypatch.setattr(experiment, "_refusals", failed_qualification)
    result = experiment.run_experiment(dataset=public_dataset[0], output=tmp_path / "out", tasks=["authored-public-task"])
    assert result["qualified"] is False
    assert result["tasks"][0]["checks"]["damaged_blob_rejected"] is False


def test_protocol_metric_includes_json_wrapper_not_only_source():
    value = {"content": "é\n", "hash": "0" * 64, "encoding": "utf-8"}
    rendered = experiment._render(value)
    metrics = experiment._metrics([value], len)
    assert metrics["rendered_json_bytes"] == len(rendered.encode())
    assert metrics["local_tokenizer_proxy_tokens"] == len(rendered)
    assert metrics["rendered_json_bytes"] > len(value["content"].encode())


def test_cli_bounded_summary(public_dataset, tmp_path, capsys):
    result = experiment.main(["--dataset", str(public_dataset[0]), "--output", str(tmp_path / "out"),
        "--task", "authored-public-task"])
    assert result == 0
    row = json.loads(capsys.readouterr().out)
    assert row["qualified"] is True
    assert row["provider_calls"] == 0
    assert row["tasks"] == ["authored-public-task"]
