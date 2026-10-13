"""Actual offline MCP subprocess cycles; no provider, container or target execution."""
import io
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
        protocol = arm["protocol"]
        assert protocol["component_totals_match"] is True
        assert protocol["wire_accounting_complete"] is True
        assert protocol["token_accounting_complete"] is True
        assert protocol["accounting_errors"] == []
        assert set(protocol["components"]) == set(experiment.WIRE_COMPONENTS)
        for metric, value in protocol["total"].items():
            assert value == sum(component[metric] for component in protocol["components"].values())
        assert protocol["components"]["setup"]["requests"] == 3
        assert protocol["components"]["setup"]["responses"] == 2
        assert protocol["components"]["status"]["requests"] == 2
        assert protocol["components"]["list"]["requests"] == 1
        assert protocol["components"]["symbols"]["requests"] == 1
        assert protocol["components"]["read"]["requests"] == len(inputs) + 4
        assert protocol["components"]["edit"]["requests"] == 1
        assert protocol["components"]["refresh"]["requests"] == 1
        assert protocol["components"]["cache_inventory"]["messages"] == 0
        for component in protocol["components"].values():
            assert component["messages"] == component["requests"] + component["responses"]
            assert component["local_tokenizer_proxy_tokens"] == (
                component["request_local_tokenizer_proxy_tokens"] + component["response_local_tokenizer_proxy_tokens"])
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


def accounting_client(tmp_path, monkeypatch, *, response, counter=len, write_limit=None, flush_error=False):
    """Controlled byte streams test client failures, not actual server behavior."""
    class Input(io.BytesIO):
        def write(self, value):
            return super().write(value if write_limit is None else value[:write_limit])

        def flush(self):
            if flush_error:
                raise BrokenPipeError("authored failed flush")
            return super().flush()

    class Process:
        stdin = Input()
        stdout = io.BytesIO(response)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(experiment.select, "select", lambda reads, writes, errors, timeout: (reads, [], []))
    config = tmp_path / "config.json"
    config.write_text("{}")
    return experiment._Client(config, counter)


@pytest.mark.parametrize("response", [b'{"jsonrpc":', b"x" * 40])
def test_failed_frame_counts_received_prefix_without_claiming_complete_traffic(tmp_path, monkeypatch, response):
    monkeypatch.setattr(experiment, "MAX_RESPONSE_BYTES", 24)
    client = accounting_client(tmp_path, monkeypatch, response=response)
    with pytest.raises(ValueError, match="response framing"):
        client.request("ping", phase="setup")
    client.close()
    report = client.metrics()
    assert report["total"]["response_bytes"] == min(len(response), 25)
    assert report["total"]["request_bytes"] == len(experiment._wire(dict(jsonrpc="2.0", method="ping", id=1)))
    assert report["total"]["messages"] == 2
    assert report["components"]["setup"]["response_bytes"] == report["total"]["response_bytes"]
    assert report["wire_accounting_complete"] is report["token_accounting_complete"] is False
    assert "response_framing" in report["accounting_errors"]
    assert "unobserved subprocess traffic" in report["traffic_scope"]


@pytest.mark.parametrize("bad_count", [None, False, 0, -1, 1.5, "7", ValueError("private tokenizer error")])
@pytest.mark.parametrize("fail_at", [1, 2])
def test_tokenizer_failure_retains_bytes_and_invalidates_all_token_totals(tmp_path, monkeypatch, bad_count, fail_at):
    calls = 0
    def counter(text):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            if isinstance(bad_count, Exception):
                raise bad_count
            return bad_count
        return len(text)
    response = experiment._wire(dict(jsonrpc="2.0", id=1, result={}))
    client = accounting_client(tmp_path, monkeypatch, response=response, counter=counter)
    with pytest.raises(ValueError, match="repository wire tokenizer failed"):
        client.request("ping", phase="setup")
    client.close()
    report = client.metrics()
    assert report["total"]["request_bytes"] == len(experiment._wire(dict(jsonrpc="2.0", method="ping", id=1)))
    assert report["total"]["response_bytes"] == (len(response) if fail_at == 2 else 0)
    assert report["total"]["messages"] == fail_at
    for row in (report["total"], report["setup"], report["workflow"], *report["components"].values()):
        assert all(value is None for key, value in row.items() if key.endswith("tokens"))
    assert report["component_totals_match"] is True
    assert report["wire_accounting_complete"] is report["token_accounting_complete"] is False
    assert "local_tokenizer_failed" in report["accounting_errors"]
    assert "private tokenizer error" not in json.dumps(report)


def test_partial_confirmed_write_counts_prefix_and_marks_exchange_incomplete(tmp_path, monkeypatch):
    client = accounting_client(tmp_path, monkeypatch, response=b"", write_limit=9)
    with pytest.raises(ValueError, match="partial request write"):
        client.request("ping", phase="setup")
    client.close()
    report = client.metrics()
    assert report["total"]["request_bytes"] == 9
    assert report["total"]["response_bytes"] == 0
    assert report["total"]["requests"] == 1
    assert report["wire_accounting_complete"] is False
    assert "request_write_partial" in report["accounting_errors"]


def test_failed_flush_does_not_invent_delivered_byte_count(tmp_path, monkeypatch):
    client = accounting_client(tmp_path, monkeypatch, response=b"", flush_error=True)
    with pytest.raises(BrokenPipeError):
        client.request("ping", phase="setup")
    client.close()
    report = client.metrics()
    assert report["total"]["request_bytes"] == 0
    assert report["wire_accounting_complete"] is report["token_accounting_complete"] is False
    assert "request_write_unconfirmed" in report["accounting_errors"]
