"""Real indexed-source qualification through a captured router boundary."""
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_input_experiment as experiment


def test_full_captured_inputs_replay_after_workspace_removal(tmp_path):
    sources = {"calculation.py": (
        b"# Public authored fixture, never executed by the experiment.\n"
        b"def increment(value):\n"
        b"    # This comment is intentionally retained only in canonical bytes.\n"
        b"    # The source input representation is independently measured.\n"
        b"    return value + 1\n")}
    result = experiment.qualify_task(name="authored", sources=sources,
        instruction="Inspect increment in calculation.py and preserve its result.",
        output=tmp_path / "qualification", counter=lambda text: (len(text.encode()) + 3) // 4)
    assert result["qualified"] is True
    assert result["provider_calls"] == result["containers_started"] == result["helper_operations"] == 0
    assert result["local_capture_calls"] == len(experiment.POLICIES)
    assert result["provider_usage_tokens"] is result["official_reward"] is None
    assert result["benchmark_source_executed"] is False
    assert [row["policy"] for row in result["arms"]] == list(experiment.POLICIES)
    assert all(row["historical_replay_after_workspace_removed"] for row in result["arms"])
    assert result["stale_allocated_source_refused"]
    assert result["canonical_payload_unchanged"] and result["canonical_public_inputs_unchanged"]
    assert result["required_instruction_exact"] and result["optional_program_raw_omitted"]
    assert len(result["comparisons"]) == 5
    assert not (tmp_path / "qualification/allocated-worktree").exists()
    assert (tmp_path / "qualification/repository/calculation.py").read_bytes() == sources["calculation.py"]


def test_non_python_case_is_ineligible_without_fake_zero_usage(tmp_path):
    result = experiment.qualify_task(name="unsupported", sources={"events.ics": b"BEGIN:VCALENDAR\n"},
        instruction="Inspect the public calendar.", output=tmp_path / "unsupported")
    assert result["disposition"] == "no_supported_python_input"
    assert result["qualified"] is False and result["arms"] == []
    assert result["provider_usage_tokens"] is None
    assert not (tmp_path / "unsupported/repository").exists()


@pytest.mark.parametrize("name", ["../private.py", "/absolute.py", ".hidden.py", "dir/../file.py"])
def test_visible_scope_rejected_before_outputs(tmp_path, name):
    with pytest.raises(ValueError, match="visible canonical"):
        experiment.qualify_task(name="bad", sources={name: b"x = 1\n"}, instruction="Read x.",
            output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path / "keep"
    marker.write_text("existing artifact")
    with pytest.raises(ValueError, match="new canonical"):
        experiment.qualify_task(name="bad", sources={"file.py": b"x = 1\n"}, instruction="Read x.", output=tmp_path)
    assert marker.read_text() == "existing artifact"


def test_changed_public_input_fails_before_output(tmp_path, monkeypatch):
    def refuse(**kwargs):
        raise ValueError("reviewed public input changed")
    monkeypatch.setattr(experiment.catalog, "reviewed_profile", refuse)
    with pytest.raises(ValueError, match="changed"):
        experiment.run_experiment(dataset=tmp_path, output=tmp_path / "out", tasks=["authored"])
    assert not (tmp_path / "out").exists()
