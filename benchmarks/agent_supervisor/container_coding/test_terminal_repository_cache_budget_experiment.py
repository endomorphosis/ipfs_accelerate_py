"""Cache-heavy wire measurements exercise the real server, without provider calls."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_cache_budget_experiment as experiment


def test_cache_summary_stays_bounded_and_pages_are_exact_and_charged(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None:
        (len, {"available": True, "scope": "authored character proxy; not provider usage"}))
    result = experiment.run_experiment(output=tmp_path / "results", cache_counts=(0, 128), status_repeats=4)
    assert result["qualified"], result
    assert result["paired_canonical_source_identical"] is True
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    assert result["target_code_executed"] is result["content_bearing_messages_retained"] is False
    empty, populated = result["cases"]
    assert empty["successful_inventory_pages"] == 1
    assert populated["successful_inventory_pages"] == 5
    assert populated["cache_summary_json_bytes"] < 1024
    assert populated["cache_summary_json_bytes"] < populated["reconstructed_inventory_json_bytes"] / 10
    assert populated["cache_summary_json_bytes"] - empty["cache_summary_json_bytes"] < 32
    assert populated["checks"]["refresh_skips_unchanged_source_parse"] is True
    assert populated["checks"]["refreshed_native_scope_current"] is True
    assert populated["checks"]["refreshed_canonical_read_hash_bound"] is True
    for row in result["cases"]:
        assert all(row["checks"].values())
        assert row["failure"] is None
        measured = row["protocol"]
        assert measured["component_totals_match"] is True
        for metric, value in measured["total"].items():
            assert value == sum(component[metric] for component in measured["components"].values())
        assert measured["components"]["read"]["requests"] == (2 if row["cache_file_count"] else 1)
        assert measured["components"]["status"]["requests"] == (5 if row["cache_file_count"] else 4)
        assert measured["components"]["cache_inventory"]["requests"] == row["successful_inventory_pages"] + (
            3 if row["cache_file_count"] else 0)
        assert row["audit"]["errors"] == (3 if row["cache_file_count"] else 0)
        assert row["audit"]["refreshes"] == (1 if row["cache_file_count"] else 0)
    files = list((tmp_path / "results").iterdir())
    assert files == [tmp_path / "results/qualification.json"]
    raw = files[0].read_text()
    assert "def authored_value" not in raw and "authored opaque cache" not in raw
    assert "entry_0000" not in raw
    assert json.loads(raw) == result


def test_missing_tokenizer_keeps_all_component_token_counts_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results", cache_counts=(0,), status_repeats=1)
    assert result["qualified"], result
    protocol = result["cases"][0]["protocol"]
    for row in [protocol["total"], protocol["setup"], protocol["workflow"], *protocol["components"].values()]:
        assert row["local_tokenizer_proxy_tokens"] is None
        assert row["request_local_tokenizer_proxy_tokens"] is None
        assert row["response_local_tokenizer_proxy_tokens"] is None
    assert protocol["total"]["rendered_json_bytes"] > 0


@pytest.mark.parametrize("options", [
    {"cache_counts": ()}, {"cache_counts": (1, 1)}, {"cache_counts": (False,)},
    {"cache_counts": (385,)}, {"status_repeats": 0}, {"status_repeats": True},
    {"page_limit": 0}, {"page_limit": 101},
])
def test_unsafe_population_refuses_before_output_or_server(tmp_path, monkeypatch, options):
    monkeypatch.setattr(experiment, "_cache_case", lambda **kwargs: pytest.fail("invalid fixture launched"))
    with pytest.raises(ValueError, match="bounded cache experiment"):
        experiment.run_experiment(output=tmp_path / "results", **options)
    assert not (tmp_path / "results").exists()


def test_failed_cases_remain_in_content_free_report(tmp_path, monkeypatch):
    def failure(**kwargs):
        raise ValueError("authored private source must not be exported")
    monkeypatch.setattr(experiment, "_cache_case", failure)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results", cache_counts=(0, 64))
    assert result["qualified"] is result["paired_canonical_source_identical"] is False
    assert len(result["cases"]) == 2
    assert all(row["error_type"] == "ValueError" for row in result["cases"])
    assert "authored private source" not in (tmp_path / "results/qualification.json").read_text()
