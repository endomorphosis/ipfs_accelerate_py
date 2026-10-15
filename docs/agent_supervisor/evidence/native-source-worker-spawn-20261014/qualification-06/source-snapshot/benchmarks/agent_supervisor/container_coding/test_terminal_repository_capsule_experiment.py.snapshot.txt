"""Real native DuckDB and stdio qualification over authored source only."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_capsule_experiment as experiment


def test_real_native_capsules_and_paired_source_lookup_keep_every_transmission(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None:
        (len, {"available": True, "scope": "authored character proxy, not provider usage"}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"], result
    assert result["paired_final_sources_identical"] is True
    assert result["representation"] == "original"
    assert set(result["arms"]) == {"full_source", "native_symbol_targeted", "context_targeted", "capsule_targeted"}
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    for key in ("target_code_executed", "training_executed", "proof_search_executed", "complete_multi_ir_qualified"):
        assert result[key] is False
    original_hashes = {name: experiment._sha(raw) for name, raw in experiment.authored_sources().items()}
    for mode, arm in result["arms"].items():
        assert arm["qualified"] and all(arm["checks"].values())
        assert arm["failure"] is None
        assert arm["checks"]["native_duckdb_head_persisted"] is True
        assert arm["checks"]["native_capsule_identity"] is True
        assert arm["checks"]["old_root_refused"] is True
        assert arm["checks"]["old_generation_refused"] is True
        assert arm["checks"]["refreshed_query_generation_refused"] is True
        assert arm["final_source_hashes"] == original_hashes
        metrics = arm["protocol"]
        assert metrics["wire_accounting_complete"] is metrics["token_accounting_complete"] is True
        assert metrics["accounting_errors"] == []
        for key, value in metrics["total"].items():
            assert value == sum(row[key] for row in metrics["components"].values())
            assert value == metrics["setup"][key] + metrics["workflow"][key]
        components = metrics["components"]
        assert components["context_query"]["requests"] == (5 if mode in {"context_targeted", "capsule_targeted"} else 4)
        assert components["capsule_expand"]["requests"] == (6 if mode == "capsule_targeted" else 5)
        assert components["symbols"]["requests"] == (1 if mode == "native_symbol_targeted" else 0)
        assert components["read"]["requests"] == (4 if mode == "full_source" else 2)
        assert components["edit"]["requests"] == 2
        assert components["refresh"]["requests"] == 1
        assert arm["audit"]["errors"] == 3
        assert arm["audit"]["canonical_edits"] == 2
        assert arm["audit"]["refreshes"] == 3
        lookup = arm["lookup"]
        assert lookup["total"]["messages"] == 2 * {
            "full_source": 4, "native_symbol_targeted": 2, "context_targeted": 2, "capsule_targeted": 3}[mode]
        assert lookup["total"]["rendered_json_bytes"] < metrics["total"]["rendered_json_bytes"]
        assert lookup["components"]["read"]["requests"] == (3 if mode == "full_source" else 1)
        assert lookup["components"]["symbols"]["requests"] == (1 if mode == "native_symbol_targeted" else 0)
        assert lookup["components"]["context_query"]["requests"] == (1 if mode in {"context_targeted", "capsule_targeted"} else 0)
        assert lookup["components"]["capsule_expand"]["requests"] == (1 if mode == "capsule_targeted" else 0)
        for key, value in lookup["total"].items():
            assert value == sum(row[key] for row in lookup["components"].values())
    outputs = list((tmp_path / "results").iterdir())
    assert outputs == [tmp_path / "results/qualification.json"]
    text = outputs[0].read_text()
    assert json.loads(text) == result
    assert "def compute_invoice_total" not in text
    assert "Authored display examples" not in text
    assert '"capsules":' not in text and '"admissions":' not in text


def test_unknown_tokenizer_preserves_real_bytes_and_unknown_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"], result
    for arm in result["arms"].values():
        assert arm["protocol"]["wire_accounting_complete"] is True
        assert arm["protocol"]["token_accounting_complete"] is False
        for measurements in (arm["lookup"], arm["protocol"]):
            for row in (measurements["total"], *measurements["components"].values()):
                assert row["local_tokenizer_proxy_tokens"] is None
                assert row["request_local_tokenizer_proxy_tokens"] is None
                assert row["response_local_tokenizer_proxy_tokens"] is None
        assert arm["lookup"]["total"]["rendered_json_bytes"] > 0


def test_existing_output_refuses_before_any_native_process(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(experiment, "_arm", lambda **kwargs: pytest.fail("existing output launched a process"))
    with pytest.raises(ValueError, match="new canonical directory"):
        experiment.run_experiment(output=output)


def test_failed_arm_is_retained_and_cannot_qualify_pair(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ValueError("private capsule body must not be retained")
    monkeypatch.setattr(experiment, "_arm", fail)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"] is result["paired_final_sources_identical"] is False
    assert set(result["arms"]) == set(experiment.ARMS)
    assert all(row["error_type"] == "ValueError" for row in result["arms"].values())
    assert "private capsule body" not in (tmp_path / "results/qualification.json").read_text()
