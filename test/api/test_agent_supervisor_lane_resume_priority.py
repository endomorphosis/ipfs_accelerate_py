"""Ready foundational work must not wait behind earlier lower-priority retries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields


pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise real priority-selected producer and validator commands")


def _add_task(
    root: Path, task_id: str, *, priority: str | None = "P1", depends: str = "",
    code: int = 0, validation: str | None = None, extra_fields: str = "",
) -> None:
    producer = root / f"produce_{task_id}.py"
    producer.write_text(
        "from pathlib import Path\n"
        f"with Path('producer-order').open('a') as output: output.write({task_id!r} + '\\n')\n"
        + (f"Path({task_id + '.result'!r}).write_text({task_id!r})\n" if code == 0 else "")
        + f"raise SystemExit({code})\n",
        encoding="utf-8",
    )
    validator = root / f"validate_{task_id}.py"
    validator.write_text(
        "from pathlib import Path\n"
        f"with Path('validator-order').open('a') as output: output.write({task_id!r} + '\\n')\n"
        f"assert Path({task_id + '.result'!r}).read_text() == {task_id!r}\n",
        encoding="utf-8",
    )
    with (root / "board.md").open("a", encoding="utf-8") as board:
        board.write(
            f"## TEST-{task_id} Work\n- Status: ready\n- Depends on: {depends}\n"
            + (f"- Priority: {priority}\n" if priority is not None else "")
            + f"- Deterministic command: python3 {producer.name}\n"
            + f"- Validation: {validation if validation is not None else 'python3 ' + validator.name}\n"
            + f"- Outputs: {task_id}.result\n- Acceptance: Actual validated output for {task_id}\n"
            + extra_fields + "\n"
        )


def _cycle(root: Path) -> dict[str, Any]:
    return lane_resume.run_resume_cycle(
        repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
        attempt_path=root / "attempts.json", log_path=root / "lane.log",
        heartbeat=lambda **_kwargs: None,
    )


def _statuses(root: Path) -> dict[str, str]:
    return {task_id: fields["status"] for task_id, fields in parse_todo_task_fields(
        (root / "board.md").read_text(), task_header_prefix="## TEST-",
    )}


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_ready_foundational_p0_runs_before_earlier_failing_p1(tmp_path: Path) -> None:
    _add_task(tmp_path, "001", priority="P1", code=1)
    _add_task(tmp_path, "900", priority="P0")

    result = _cycle(tmp_path)

    assert result["phase"] == "task_command_passed"
    assert result["advanced_task_id"] == "TEST-900"
    assert _lines(tmp_path / "producer-order") == ["900"]
    assert _lines(tmp_path / "validator-order") == ["900"]
    assert _statuses(tmp_path) == {"TEST-001": "ready", "TEST-900": "completed"}
    assert result["work_complete"] is False
    assert result["completion_authority"] is False


def test_priority_is_numeric_stable_and_cannot_override_dependencies(tmp_path: Path) -> None:
    _add_task(tmp_path, "001", priority="P10")
    _add_task(tmp_path, "002", priority="P2")
    _add_task(tmp_path, "003", priority="p2")
    _add_task(tmp_path, "004", priority="P0", depends="TEST-001")

    advanced = [_cycle(tmp_path)["advanced_task_id"] for _ in range(4)]

    assert advanced == ["TEST-002", "TEST-003", "TEST-001", "TEST-004"]
    assert _lines(tmp_path / "producer-order") == ["002", "003", "001", "004"]
    assert set(_statuses(tmp_path).values()) == {"completed"}


def test_unknown_missing_numeric_and_fib_priorities_keep_document_positions(tmp_path: Path) -> None:
    _add_task(tmp_path, "001", priority=None)
    _add_task(tmp_path, "002", priority="P2")
    _add_task(tmp_path, "003", priority="1")
    _add_task(tmp_path, "004", priority="P0")
    _add_task(tmp_path, "005", priority="urgent", extra_fields="- Fib priority: 1\n")

    advanced = [_cycle(tmp_path)["advanced_task_id"] for _ in range(5)]

    assert advanced == ["TEST-001", "TEST-004", "TEST-002", "TEST-003", "TEST-005"]


def test_cooling_p0_yields_to_independent_p1_without_reusing_stale_green_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 1)
    _add_task(tmp_path, "001", priority="P0", code=7)
    _add_task(tmp_path, "002", priority="P1")
    # This validator would accept an old green artifact. A failing producer
    # did not change it, so priority selection must not complete the task.
    (tmp_path / "001.result").write_text("001")

    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    original = json.loads((tmp_path / "attempts.json").read_text())
    assert _cycle(tmp_path)["advanced_task_id"] == "TEST-002"
    waiting = _cycle(tmp_path)

    assert waiting["phase"] == "attempt_retry_wait"
    assert _lines(tmp_path / "producer-order") == ["001", "002"]
    assert _lines(tmp_path / "validator-order") == ["002"]
    assert _statuses(tmp_path) == {"TEST-001": "ready", "TEST-002": "completed"}
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved["_retry_after_ms"]["TEST-001"] == original["_retry_after_ms"]["TEST-001"]
    assert waiting["work_complete"] is False


def test_repaired_p0_contract_rearms_before_unattempted_lower_priority_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 1)
    _add_task(tmp_path, "001", priority="P0", code=1)
    _add_task(tmp_path, "002", priority="P1")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    producer = tmp_path / "produce_001.py"
    producer.write_text("from pathlib import Path\nPath('001.result').write_text('001')\n")

    result = _cycle(tmp_path)

    assert result["advanced_task_id"] == "TEST-001"
    assert _statuses(tmp_path) == {"TEST-001": "completed", "TEST-002": "ready"}
    assert not (tmp_path / "002.result").exists()


def test_invalid_p0_validator_cannot_launch_producer_or_block_valid_independent_work(tmp_path: Path) -> None:
    _add_task(tmp_path, "001", priority="P0", validation="grok --prompt validate")
    _add_task(tmp_path, "002", priority="P1")

    result = _cycle(tmp_path)

    assert result["advanced_task_id"] == "TEST-002"
    assert _lines(tmp_path / "producer-order") == ["002"]
    assert _statuses(tmp_path)["TEST-001"] == "ready"


def test_withheld_p0_gate_yields_without_marking_it_complete(tmp_path: Path) -> None:
    _add_task(tmp_path, "001", priority="P0", code=75)
    _add_task(tmp_path, "002", priority="P1")

    result = _cycle(tmp_path)

    assert result["advanced_task_id"] == "TEST-002"
    assert _lines(tmp_path / "producer-order") == ["001", "002"]
    assert _statuses(tmp_path) == {"TEST-001": "ready", "TEST-002": "completed"}
    assert result["completion_authority"] is False
