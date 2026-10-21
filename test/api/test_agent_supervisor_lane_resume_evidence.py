"""Completion must use evidence produced by the latest declared task attempt."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume


pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise real producer and validator invocations")


@pytest.fixture(autouse=True)
def _one_failure_before_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 1)


def _task(
    root: Path, *, producer: str, validation: str = "python3 validate.py",
    declared: str = "- Outputs: report.json\n", command: str = "python3 produce.py",
) -> None:
    (root / "produce.py").write_text("from pathlib import Path\n" + producer, encoding="utf-8")
    (root / "validate.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "Path('validator-ran').touch()\n"
        "assert json.loads(Path('report.json').read_text())['decision'] == 'GO'\n",
        encoding="utf-8",
    )
    (root / "board.md").write_text(
        "## TEST-001 Produce evidence\n- Status: pending\n- Depends on:\n"
        f"- Deterministic command: {command}\n- Validation: {validation}\n"
        + declared + "- Acceptance: Fresh validated evidence\n",
        encoding="utf-8",
    )


def _cycle(root: Path, **options: Any) -> dict[str, Any]:
    return lane_resume.run_resume_cycle(
        repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
        attempt_path=root / "attempts.json", log_path=root / "lane.log",
        heartbeat=lambda **_kwargs: None, **options,
    )


def _assert_waiting(root: Path, result: dict[str, Any]) -> None:
    assert result["phase"] == "attempt_retry_wait"
    assert result["work_complete"] is False
    assert result["completion_authority"] is False
    assert "- Status: pending" in (root / "board.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["python3 produce.py", ""])
@pytest.mark.parametrize("validator", [
    "grok --prompt validate", "python3 validate.py && test -f report.json",
    "python3 'unterminated", "missing-validator.py",
])
def test_rejected_validator_prevents_production_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, validator: str,
) -> None:
    _task(tmp_path, producer="Path('producer-ran').touch()\n", validation=validator, command=command)

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid validation must be refused before invoking a producer")

    monkeypatch.setattr(lane_resume, "_run_argv", unexpected)
    monkeypatch.setattr(lane_resume, "prepare_bounded_implement", unexpected)
    result = _cycle(tmp_path, allow_bounded_implement=True)

    assert result["state"] == "blocked"
    assert "TEST-001:validation_provider_or_shell_refused" in result["backpressure_reasons"]
    assert "- Status: pending" in (tmp_path / "board.md").read_text(encoding="utf-8")
    assert not (tmp_path / "producer-ran").exists()


def test_satisfies_cannot_complete_a_source_with_a_rejected_validator(tmp_path: Path) -> None:
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\n")
    board = tmp_path / "board.md"
    repair = board.read_text(encoding="utf-8") + "- Satisfies: TEST-000\n"
    board.write_text(
        "## TEST-000 Original task\n- Status: pending\n- Depends on: TEST-001\n"
        "- Validation: grok --prompt validate\n\n" + repair,
        encoding="utf-8",
    )

    result = _cycle(tmp_path)

    assert result["phase"] == "task_command_passed"
    assert result["advanced_task_id"] == "TEST-001"
    source, repair = board.read_text(encoding="utf-8").split("## TEST-001", 1)
    assert "- Status: pending" in source
    assert "- Status: completed" in repair


def test_unchanged_old_go_does_not_complete_after_a_failed_attempt(tmp_path: Path) -> None:
    _task(tmp_path, producer="import os\nos.utime('report.json', (1, 1))\nraise SystemExit(1)\n")
    (tmp_path / "report.json").write_text('{"decision":"GO"}\n', encoding="utf-8")

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path, _cycle(tmp_path))

    assert not (tmp_path / "validator-ran").exists()
    receipt = json.loads((tmp_path / "attempts.json").read_text())["_command_evidence"]["TEST-001"]
    assert receipt["before"] == receipt["after"]


def test_failed_attempt_without_declared_evidence_cannot_complete(tmp_path: Path) -> None:
    _task(tmp_path, producer="raise SystemExit(1)\n", declared="")
    (tmp_path / "report.json").write_text('{"decision":"GO"}\n', encoding="utf-8")

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert not (tmp_path / "validator-ran").exists()


@pytest.mark.parametrize("declared", ["- Outputs: report.json\n", "- Predicted files: report.json\n"])
def test_fresh_valid_output_can_complete_during_backoff_after_reload(
    tmp_path: Path, declared: str,
) -> None:
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\nraise SystemExit(1)\n",
          declared=declared)

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    # Each cycle reloads the persisted attempt state, just as a replacement lane does.
    result = _cycle(tmp_path)

    assert result["phase"] == "task_command_passed"
    assert result["advanced_task_id"] == "TEST-001"
    assert result["completion_authority"] is False
    assert (tmp_path / "validator-ran").exists()
    assert "- Status: completed" in (tmp_path / "board.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("changed", ["contract", "producer", "validator", "artifact"])
def test_backoff_rejects_changes_after_evidence_was_produced(tmp_path: Path, changed: str) -> None:
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\nraise SystemExit(1)\n")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"

    paths = {"contract": "board.md", "producer": "produce.py", "validator": "validate.py"}
    if changed == "artifact":
        (tmp_path / "report.json").write_text('{"decision":"GO","revision":2}', encoding="utf-8")
    else:
        path = tmp_path / paths[changed]
        body = path.read_text(encoding="utf-8")
        path.write_text(body.replace("Fresh validated evidence", "Different acceptance")
                        if changed == "contract" else body + "# Changed after the attempt\n",
                        encoding="utf-8")

    result = _cycle(tmp_path)
    if changed == "artifact":
        _assert_waiting(tmp_path, result)
    else:
        assert result["phase"] == "deterministic_command_failed"
        _assert_waiting(tmp_path, _cycle(tmp_path))
    assert not (tmp_path / "validator-ran").exists()


def _counting_failure(root: Path) -> None:
    _task(root, producer=
          "count = Path('producer-count')\n"
          "count.write_text(str(int(count.read_text()) + 1) if count.exists() else '1')\n"
          "raise SystemExit(1)\n")


def test_changed_producer_rearms_once_and_preserves_other_task_cooldowns(tmp_path: Path) -> None:
    _counting_failure(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert (tmp_path / "producer-count").read_text() == "1"
    attempt_path = tmp_path / "attempts.json"
    saved = json.loads(attempt_path.read_text())
    other_due = saved["_retry_after_ms"]["TEST-001"]
    saved["OTHER-001"] = 7
    saved["_retry_after_ms"]["OTHER-001"] = other_due
    attempt_path.write_text(json.dumps(saved), encoding="utf-8")

    producer = tmp_path / "produce.py"
    old_stat = producer.stat()
    producer.write_text(producer.read_text() + "# Repaired producer content\n", encoding="utf-8")
    os.utime(producer, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert (tmp_path / "producer-count").read_text() == "2"
    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert (tmp_path / "producer-count").read_text() == "2"
    saved = json.loads(attempt_path.read_text())
    assert saved["TEST-001"] == 1
    assert saved["OTHER-001"] == 7
    assert saved["_retry_after_ms"]["OTHER-001"] == other_due
    assert (tmp_path / "lane.log").read_text().count("retry rearmed TEST-001:") == 1


def test_producer_mtime_change_does_not_rearm_retry(tmp_path: Path) -> None:
    _counting_failure(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    producer = tmp_path / "produce.py"
    old_stat = producer.stat()
    os.utime(producer, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))

    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert (tmp_path / "producer-count").read_text() == "1"


def test_repaired_producer_can_complete_without_waiting_for_old_cooldown(tmp_path: Path) -> None:
    _counting_failure(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    producer = tmp_path / "produce.py"
    producer.write_text(producer.read_text().replace(
        "raise SystemExit(1)", "Path('report.json').write_text('{\"decision\":\"GO\"}')",
    ), encoding="utf-8")

    result = _cycle(tmp_path)

    assert result["phase"] == "task_command_passed"
    assert (tmp_path / "producer-count").read_text() == "2"
    assert (tmp_path / "validator-ran").exists()
    assert "- Status: completed" in (tmp_path / "board.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("record", [None, {}, {"contract": "0" * 64}])
def test_changed_script_without_attempt_provenance_keeps_cooldown(
    tmp_path: Path, record: dict[str, Any] | None,
) -> None:
    _counting_failure(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    attempt_path = tmp_path / "attempts.json"
    saved = json.loads(attempt_path.read_text())
    if record is None:
        saved.pop("_command_evidence")
    else:
        saved["_command_evidence"]["TEST-001"] = record
    attempt_path.write_text(json.dumps(saved), encoding="utf-8")
    producer = tmp_path / "produce.py"
    producer.write_text(producer.read_text() + "# Changed without retained provenance\n", encoding="utf-8")

    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert (tmp_path / "producer-count").read_text() == "1"


def test_latest_unchanged_failure_invalidates_an_earlier_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 2)
    _task(tmp_path, producer=
          "if not Path('report.json').exists():\n"
          "    Path('report.json').write_text('{\"decision\":\"GO\"}')\n"
          "raise SystemExit(1)\n")

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert not (tmp_path / "validator-ran").exists()


def test_successful_validator_cannot_change_the_evidence_it_validates(tmp_path: Path) -> None:
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\nraise SystemExit(1)\n")
    with (tmp_path / "validate.py").open("a", encoding="utf-8") as handle:
        handle.write("Path('report.json').write_text('{\"decision\":\"GO\",\"revision\":2}')\n")

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path, _cycle(tmp_path))
    assert (tmp_path / "validator-ran").exists()


def test_interrupted_latest_attempt_cannot_keep_previous_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 2)
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\nraise SystemExit(1)\n")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"

    def interrupted(*_args: Any, **_kwargs: Any) -> int:
        raise lane_resume._LaneResumeInterrupted

    monkeypatch.setattr(lane_resume, "_run_argv", interrupted)
    with pytest.raises(lane_resume._LaneResumeInterrupted):
        _cycle(tmp_path)

    saved = json.loads((tmp_path / "attempts.json").read_text(encoding="utf-8"))
    assert "TEST-001" not in saved.get("_command_evidence", {})
