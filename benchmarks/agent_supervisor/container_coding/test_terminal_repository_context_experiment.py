"""Actual authored repository context workflows and honest traffic accounting."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_context_experiment as experiment


def test_real_native_context_expansion_and_retry_preserve_complete_domain_population(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None:
        (len, {"available": True, "scope": "authored character proxy, not provider usage"}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"], result
    assert result["paired_scripted_decisions_identical"]
    assert result["original_sources_unchanged"] and result["private_fixture_removed"]
    assert result["required_domain_population"] == 3
    assert set(result["arms"]) == set(experiment.ARMS)
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    for field in ("target_code_executed", "training_executed", "proof_search_executed", "complete_multi_ir_qualified"):
        assert result[field] is False
    full, compact = (result["arms"][name] for name in experiment.ARMS)
    assert full["scripted_decision_sha256"] == compact["scripted_decision_sha256"]
    assert full["initial_native_body_count"] == 3 and full["question_expanded_body_count"] == 0
    assert compact["initial_native_body_count"] == 0 and compact["question_expanded_body_count"] == 1
    for arm in (full, compact):
        assert arm["qualified"] and all(arm["checks"].values())
        assert arm["required_reference_count"] == 3
        assert arm["content_bearing_messages_retained"] is False
        traffic = arm["traffic"]
        assert traffic["wire_accounting_complete"] and traffic["component_totals_match"]
        assert traffic["total"]["token_accounting_complete"]
        assert set(traffic["components"]) == {"initial_context", "question", "question_expansion", "diagnostic_retry"}
        for field in ("messages", "rendered_json_bytes", "local_tokenizer_proxy_tokens"):
            assert traffic["total"][field] == sum(row[field] for row in traffic["components"].values())
        # ASCII fixture and this deliberately exact character proxy have the
        # same measured bytes/tokens; no fabricated provider tokenizer count.
        assert traffic["total"]["rendered_json_bytes"] == traffic["total"]["local_tokenizer_proxy_tokens"]
    assert full["traffic"]["components"]["question"] == compact["traffic"]["components"]["question"]
    assert full["traffic"]["components"]["question_expansion"]["messages"] == 0
    assert compact["traffic"]["components"]["question_expansion"]["messages"] == 1
    assert full["traffic"]["total"]["messages"] == 3
    assert compact["traffic"]["total"]["messages"] == 4
    files = list((tmp_path / "results").iterdir())
    assert files == [tmp_path / "results" / "qualification.json"]
    saved = files[0].read_text()
    assert json.loads(saved) == result
    for source in experiment.authored_sources().values():
        assert source.decode() not in saved
    for body in experiment.REQUIREMENTS.values():
        assert body["required_text"] not in saved
    assert experiment.DIAGNOSTIC["message"] not in saved
    assert '"capsules":' not in saved and '"admissions":' not in saved


def test_unavailable_tokenizer_preserves_bytes_and_unknown_token_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"], result
    for arm in result["arms"].values():
        traffic = arm["traffic"]
        assert traffic["wire_accounting_complete"]
        assert traffic["total"]["rendered_json_bytes"] > 0
        for row in (traffic["total"], *traffic["components"].values()):
            assert row["local_tokenizer_proxy_tokens"] is None
            assert row["token_accounting_complete"] is False


def test_existing_output_is_refused_before_native_session(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(experiment, "_exercise", lambda counter: pytest.fail("existing output launched native work"))
    with pytest.raises(ValueError, match="new canonical directory"):
        experiment.run_experiment(output=output)


def test_failed_native_work_remains_unqualified_and_content_free(tmp_path, monkeypatch):
    def fail(counter):
        raise ValueError("private native capsule and required source body")
    monkeypatch.setattr(experiment, "_exercise", fail)
    monkeypatch.setattr(experiment, "local_token_counter", lambda cache=None: (None, {"available": False}))
    result = experiment.run_experiment(output=tmp_path / "results")
    assert result["qualified"] is result["paired_scripted_decisions_identical"] is False
    assert result["error_type"] == "ValueError" and result["arms"] == {}
    assert "private native capsule" not in (tmp_path / "results/qualification.json").read_text()


@pytest.mark.parametrize("bad", [True, -1, "seven"])
def test_invalid_local_tokenizer_counts_never_become_reported_savings(bad):
    metrics = experiment._traffic({"initial_context": "authored frame"}, lambda text: bad)
    assert metrics["wire_accounting_complete"] is True
    assert metrics["total"]["rendered_json_bytes"] == len("authored frame")
    assert metrics["total"]["local_tokenizer_proxy_tokens"] is None
    assert metrics["total"]["token_accounting_complete"] is False
