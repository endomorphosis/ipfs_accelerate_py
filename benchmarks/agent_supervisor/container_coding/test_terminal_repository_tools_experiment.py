"""Actual offline MCP subprocess cycles; no provider, container or target execution."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_tools_experiment as experiment
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_view_experiment import public_dataset


def test_real_stdio_public_source_cycle_and_content_free_artifact(public_dataset, tmp_path, monkeypatch):
    dataset, task, _catalog, inputs = public_dataset
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (len,
        {"available": True, "scope": "authored character counter; not provider usage"}))
    output = tmp_path / "qualification"
    result = experiment.run_experiment(dataset=dataset, output=output, tasks=["authored-public-task"])
    assert result["qualified"], result
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    assert result["benchmark_source_executed"] is False
    assert result["content_bearing_messages_retained"] is False
    assert result["original_dataset_unchanged"] is True
    row = result["tasks"][0]
    assert row["paired_final_canonical_sources_identical"]
    for representation, arm in row["arms"].items():
        assert arm["qualified"] and all(arm["checks"].values())
        assert arm["content_bearing_messages_retained"] is False
        assert arm["public_source_edits"] == 1
        assert arm["audit"]["canonical_edits"] == 1
        assert arm["audit"]["refreshes"] == 2
        assert arm["audit"]["errors"] == 2
        assert arm["subprocess_exit_code"] == 0
        assert arm["refusal_codes"] == {"old_generation": "stale_generation", "external_source_change": "stale_source"}
        assert arm["final_ast_stats"]["reuse_hits"] >= 1
        assert arm["protocol"]["setup"]["messages"] == 7
        for metric in ("messages", "request_bytes", "response_bytes", "rendered_json_bytes", "local_tokenizer_proxy_tokens"):
            assert arm["protocol"]["total"][metric] == sum(arm["protocol"][phase][metric] for phase in ("setup", "workflow"))
        if representation == "compact":
            assert arm["compacted_file_reads"] == 2
            assert arm["audit"]["compact_reads"] >= len(inputs)
        else:
            assert arm["compacted_file_reads"] == arm["audit"]["compact_reads"] == 0
    files = list(output.rglob("*"))
    assert files == [output / "qualification.json"]
    raw = files[0].read_text()
    assert "Retain original canonical formatting" not in raw
    assert "def calculate" not in raw
    assert json.loads(raw) == result
    for path, raw in inputs.items():
        assert (task / "environment" / path).read_bytes() == raw


def test_changed_public_file_refused_before_process_or_output(public_dataset, tmp_path, monkeypatch):
    dataset, task, _catalog, _inputs = public_dataset
    (task / "environment/source/example.py").write_text("tampered\n")
    monkeypatch.setattr(experiment, "_arm", lambda **kwargs: pytest.fail("unreviewed source dispatched"))
    with pytest.raises(ValueError, match="changed"):
        experiment.run_experiment(dataset=dataset, output=tmp_path / "out", tasks=["authored-public-task"])
    assert not (tmp_path / "out").exists()


def test_failed_arm_is_retained_without_silent_success(public_dataset, tmp_path, monkeypatch):
    def failed(**kwargs):
        raise ValueError("source-bearing text must not be retained")
    monkeypatch.setattr(experiment, "_arm", failed)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    output = tmp_path / "out"
    result = experiment.run_experiment(dataset=public_dataset[0], output=output, tasks=["authored-public-task"])
    assert result["qualified"] is False
    assert set(result["tasks"][0]["arms"]) == {"original", "compact"}
    assert all(arm["error_type"] == "ValueError" for arm in result["tasks"][0]["arms"].values())
    assert "source-bearing text" not in (output / "qualification.json").read_text()
