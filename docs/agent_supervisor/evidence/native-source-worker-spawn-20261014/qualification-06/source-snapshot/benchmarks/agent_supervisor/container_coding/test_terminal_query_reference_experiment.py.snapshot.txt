"""Real indexed producer and captured router, without a model or benchmark run."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_query_reference_experiment as experiment


SOURCES = {"calculation.py": b"def increment(value):\n    return value + 1\n"}
INSTRUCTION = ("Inspect increment in calculation.py and preserve its arithmetic behavior. "
               "Use the declared output and validate the existing interface. ") * 12


def test_full_inputs_restore_and_replay_after_workspace_removal(tmp_path):
    result = experiment.qualify_task(name="authored", sources=SOURCES, instruction=INSTRUCTION,
        output=tmp_path / "qualified", counter=lambda text: (len(text.encode()) + 3) // 4)
    assert result["qualified"] and result["observations_valid"]
    assert result["provider_calls"] == result["containers_started"] == 0
    assert result["local_capture_calls"] == 2
    assert result["provider_usage_tokens"] is result["official_reward"] is result["task_solved"] is None
    assert result["benchmark_source_executed"] is False
    assert result["coding_reply_mode"] == "ordinary-completion@1"
    assert [row["policy"] for row in result["arms"]] == list(experiment.POLICIES)
    before, after = result["arms"]
    assert before["positive_reference_observed"] is False and before["reference_count"] == 0
    assert before["native_output_schema_sha256"] == after["native_output_schema_sha256"]
    assert before["native_output_schema_bytes"] == after["native_output_schema_bytes"] > 0
    assert after["model_prompt_and_native_output_schema_bytes"] == after["full_input_bytes"] + after["native_output_schema_bytes"]
    assert after["positive_reference_observed"] is True and after["reference_count"] == 1
    assert result["comparison"]["input_bytes_change"] < 0
    assert result["comparison"]["input_proxy_tokens_change"] < 0
    assert result["complete_input_bytes_reduced"]
    assert after["query_reference_receipt"]["byte_savings"] == -result["comparison"]["input_bytes_change"]
    assert all(row["historical_replay_after_workspace_removed"] and row["exact_legacy_transport_restored"]
               and row["required_instruction_inline_exact"] for row in result["arms"])
    assert result["stale_required_instruction_refused"] and result["allocated_worktree_removed"]
    assert result["canonical_support_inputs_unchanged"] and result["canonical_public_inputs_unchanged"]
    assert result["canonical_payload_unchanged"] and result["matched_source_selection"]
    assert result["required_instruction_exact"] and result["optional_program_raw_omitted"]
    assert (tmp_path / "qualified/repository/calculation.py").read_bytes() == SOURCES["calculation.py"]
    retained = json.loads((tmp_path / "qualified/reference-receipt.json").read_text())
    assert retained["query_reference_selected"] and retained["query_reference_dispatched"]
    assert retained["model_prompt_sha256"] == after["full_input_sha256"]
    assert "model_prompt" not in retained and "provider_observation" not in retained
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    native = experiment._json(json.loads((tmp_path / "qualified/native-context-input.json").read_text()))
    parsed = audit._receipt(retained, tmp_path / "qualified")
    reconstructed, _, checks = audit._model_projection(rendered=native, receipt=parsed,
        repository=tmp_path / "qualified/repository")
    assert experiment._sha(reconstructed) == after["full_input_sha256"] and all(checks.values())


@pytest.mark.parametrize("instruction,query,reason", [
    ("increment", None, "unprofitable_query_reference"),
    (INSTRUCTION, "A different authored ranking query", "no_exact_required_source"),
])
def test_ineligible_or_unprofitable_treatment_is_retained(tmp_path, instruction, query, reason):
    result = experiment.qualify_task(name="ineligible", sources=SOURCES, instruction=instruction,
        worker_query=query, output=tmp_path / "ineligible")
    assert result["observations_valid"] and result["qualified"] is False
    assert result["local_capture_calls"] == 1 and result["provider_calls"] == 0
    assert result["arms"][0]["disposition"] == "captured"
    refused = result["arms"][1]
    assert refused["disposition"] == "ineligible_before_provider_boundary"
    assert refused["reason_code"] == reason
    assert refused["positive_reference_observed"] is False
    assert refused["full_input_bytes"] is refused["full_input_proxy_tokens"] is None
    assert result["comparison"]["input_bytes_change"] is None
    assert result["allocated_worktree_removed"]


def test_public_profile_uses_canonical_three_support_inputs(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import (
        INSTRUCTION as PATH, PROFILE, SMOKE, instruction_sha256,
    )
    profile = {"schema": "terminal-public-task-profile@1", "instruction_sha256": instruction_sha256(INSTRUCTION),
        "input_paths": ["calculation.py"], "outputs": [
            {"path": "calculation.py", "effect": "modify", "media_type": "text/x-python"}]}
    result = experiment.qualify_task(name="public-authored", sources=SOURCES, instruction=INSTRUCTION,
        public_profile=profile, output=tmp_path / "profile")
    assert set(result["support_input_sha256"]) == {PATH, PROFILE, SMOKE}
    assert result["required_raw_paths"] == [PATH, SMOKE]
    assert result["program_paths"] == ["calculation.py"]
    assert result["worker_query_sha256"] == result["public_instruction_sha256"]
    assert result["qualified"] and result["provider_usage_tokens"] is None


@pytest.mark.parametrize("path", ["../private.py", "/absolute.py", ".hidden.py", "dir/../file.py"])
def test_nonpublic_scope_refused_before_outputs(tmp_path, path):
    with pytest.raises(ValueError, match="visible canonical"):
        experiment.qualify_task(name="bad", sources={path: b"x=1\n"}, instruction=INSTRUCTION,
                                output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path / "keep"
    marker.write_text("existing evidence")
    with pytest.raises(ValueError, match="new canonical"):
        experiment.qualify_task(name="bad", sources=SOURCES, instruction=INSTRUCTION, output=tmp_path)
    assert marker.read_text() == "existing evidence"


def test_reviewed_input_change_refused_before_outputs(tmp_path, monkeypatch):
    def refuse(**kwargs):
        raise ValueError("reviewed public input changed")
    monkeypatch.setattr(experiment.catalog, "reviewed_profile", refuse)
    with pytest.raises(ValueError, match="changed"):
        experiment.run_experiment(dataset=tmp_path, output=tmp_path / "out", tasks=["authored"])
    assert not (tmp_path / "out").exists()


def test_non_python_scope_is_retained_without_invented_usage(tmp_path):
    result = experiment.qualify_task(name="unsupported", sources={"events.ics": b"BEGIN:VCALENDAR\n"},
        instruction=INSTRUCTION, output=tmp_path / "unsupported")
    assert result["disposition"] == "no_supported_python_input"
    assert result["qualified"] is False and result["observations_valid"] and result["arms"] == []
    assert result["provider_usage_tokens"] is None


@pytest.mark.parametrize("count", [True, -1, 3.5])
def test_proxy_counters_cannot_impersonate_usage(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        experiment._measure("public authored input", lambda _: count)
