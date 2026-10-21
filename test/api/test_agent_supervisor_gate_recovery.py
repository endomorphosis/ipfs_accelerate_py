"""A withheld gate must expose missing work without becoming its own progress."""

from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import gate_recovery, lane_resume
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    board = tmp_path / "board.md"
    board.write_text(
        "## TEST-010 Source\n- Status: completed\n- Depends on:\n"
        "- Predicted files: source.py\n- Validation: python3 source.py\n\n"
        "## TEST-040 Aggregate\n- Status: completed\n- Depends on: TEST-010\n"
        "- Deterministic command: python3 aggregate.py\n"
        "- Predicted files: coverage.json\n- Validation: test -f coverage.json\n\n"
        "## TEST-060 Closeout\n- Status: pending\n- Depends on: TEST-040\n"
        "- Deterministic command: python3 gate.py\n"
        "- Validation: python3 validate.py\n- Predicted files: board.md\n",
        encoding="utf-8",
    )
    for name in ("source.py", "aggregate.py", "gate.py", "validate.py", "repair.py"):
        (tmp_path / name).write_text("# original source\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"decision": "NO_GO", "passed": False, "generated_at": "old",
                                  "reasons": ["missing live invocation"]}), encoding="utf-8")
    now = [1000.0]
    monkeypatch.setattr(lane_resume.time, "time", lambda: now[0])
    monkeypatch.setattr(lane_resume, "_refill_module", lambda _root: None)
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> int:
        calls.append(list(argv))
        with kwargs["log_path"].open("a", encoding="utf-8") as stream:
            stream.write(f"fresh log {len(calls)}\n")
        payload = json.loads(report.read_text())
        payload["generated_at"] = f"probe-{len(calls)}"
        report.write_text(json.dumps(payload))
        return 75 if argv[-1] == "gate.py" else 0

    monkeypatch.setattr(lane_resume, "_run_argv", run)
    kwargs = dict(repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-",
                  attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "lane.log",
                  heartbeat=lambda **_kwargs: None)
    return SimpleNamespace(root=tmp_path, board=board, report=report, now=now, calls=calls,
                           kwargs=kwargs, run=run, cycle=lambda: lane_resume.run_resume_cycle(**kwargs))


def test_withheld_fingerprint_survives_restarts_and_ignores_clock_and_log_churn(lane: SimpleNamespace) -> None:
    original = lane.board.read_text()
    first = lane.cycle()
    second = lane.cycle()
    assert first["phase"] == second["phase"] == "gate_withheld"
    assert first["evidence_changed"] is True and second["evidence_changed"] is False
    third = lane.cycle()
    assert third["phase"] == "gate_recovery_needed"
    assert third["gate_recovery_needed"] is True
    assert third["gate_evidence_fingerprint"] == first["gate_evidence_fingerprint"]
    packet = json.loads(Path(third["repair_packet"]).read_text())
    assert packet["dependencies"] == ["TEST-040", "TEST-010"]
    assert packet["artifacts"]["coverage.json"]["decision"] == "NO_GO"
    assert "generated_at" not in packet["artifacts"]["coverage.json"]
    assert packet["dependency_validations"][0]["validation_strength"] == "existence_only"
    assert all(result["resolved_gate"] is False for result in packet["dependency_validations"])
    assert "board.md" not in packet["allowed_source_paths"]
    assert "gate.py" not in packet["allowed_source_paths"]
    assert "source.py" in packet["allowed_source_paths"]
    assert packet["work_complete"] is packet["completion_authority"] is False
    assert packet["body_free"] is True
    calls = len(lane.calls)
    waiting = lane.cycle()
    assert waiting["phase"] == "gate_retry_wait"
    assert waiting["gate_recovery_needed"] is True
    assert waiting["gate_evidence_fingerprint"] == third["gate_evidence_fingerprint"]
    assert len(lane.calls) == calls
    assert lane.board.read_text() == original
    assert "TEST-060" not in json.loads((lane.root / "attempts.json").read_text())
    persisted = gate_recovery.GateRecovery(lane.root / "attempts.json")
    assert persisted.current("TEST-060", first["gate_evidence_fingerprint"])["unchanged_probes"] == 3


@pytest.mark.parametrize("change", ["source", "decision", "reason"])
def test_semantic_changes_bypass_gate_cooldown(lane: SimpleNamespace, change: str) -> None:
    for _ in range(3):
        previous = lane.cycle()
    count = len(lane.calls)
    if change == "source":
        (lane.root / "source.py").write_text("# real implementation change\n")
    else:
        report = json.loads(lane.report.read_text())
        report["decision" if change == "decision" else "reasons"] = "GO" if change == "decision" else ["new gap"]
        lane.report.write_text(json.dumps(report))
    changed = lane.cycle()
    assert len(lane.calls) == count + 1
    assert changed["phase"] == "gate_withheld"
    assert changed["gate_recovery_needed"] is False
    assert changed["gate_evidence_fingerprint"] != previous["gate_evidence_fingerprint"]


def test_inline_validator_evidence_paths_are_part_of_semantic_fingerprint(lane: SimpleNamespace) -> None:
    inline = "python3 -c 'import json; assert json.load(open(\"coverage.json\"))[\"decision\"] == \"GO\"'"
    fields = {"TEST-060": {"status": "pending", "validation": inline}}
    kwargs = dict(repo_root=lane.root, todo_path=lane.board, log_path=lane.root / "lane.log")
    before, evidence = gate_recovery.gate_evidence("TEST-060", fields, **kwargs)
    assert evidence["artifacts"]["coverage.json"]["decision"] == "NO_GO"
    report = json.loads(lane.report.read_text())
    report.update(generatedAt="new", logs=["a fresh log"], updated_at_ms=999)
    lane.report.write_text(json.dumps(report))
    assert gate_recovery.gate_evidence("TEST-060", fields, **kwargs)[0] == before
    report["decision"] = "GO"
    lane.report.write_text(json.dumps(report))
    assert gate_recovery.gate_evidence("TEST-060", fields, **kwargs)[0] != before


def test_unchanged_retries_back_off_and_revalidate_dependencies_only_once(lane: SimpleNamespace) -> None:
    for _ in range(3):
        previous = lane.cycle()
    for expected_delay in (120_000, 240_000, 480_000, 900_000, 900_000):
        lane.now[0] = previous["next_retry_at_ms"] / 1000 + 1
        calls = len(lane.calls)
        previous = lane.cycle()
        assert previous["phase"] == "gate_recovery_needed"
        assert previous["next_retry_at_ms"] - int(lane.now[0] * 1000) == expected_delay
        assert len(lane.calls) == calls + 1, "upstream validators ran again without changed evidence"


@pytest.mark.parametrize("during_cooldown", [False, True])
def test_withheld_gate_does_not_starve_later_ready_task(lane: SimpleNamespace, during_cooldown: bool) -> None:
    if during_cooldown:
        for _ in range(3):
            lane.cycle()
    with lane.board.open("a") as stream:
        stream.write("\n## TEST-070 Useful repair\n- Status: pending\n- Depends on:\n"
                     "- Deterministic command: python3 repair.py\n- Validation: python3 validate.py\n")
    result = lane.cycle()
    assert result["advanced_task_id"] == "TEST-070"
    tasks = dict(parse_todo_task_fields(lane.board.read_text(), task_header_prefix="## TEST-"))
    assert tasks["TEST-060"]["status"] == "pending"
    assert tasks["TEST-070"]["status"] == "completed"


def test_explicit_deterministic_repair_runs_once_and_cannot_force_closeout(lane: SimpleNamespace) -> None:
    lane.board.write_text(lane.board.read_text() + "- Deterministic repair: python3 repair.py\n")
    for _ in range(3):
        previous = lane.cycle()
    assert previous["phase"] == "gate_recovery_needed"
    assert sum(argv[-1] == "repair.py" for argv in lane.calls) == 1
    lane.now[0] = previous["next_retry_at_ms"] / 1000 + 1
    assert lane.cycle()["phase"] == "gate_recovery_needed"
    assert sum(argv[-1] == "repair.py" for argv in lane.calls) == 1
    assert "- Status: pending" in lane.board.read_text()


def test_secondary_validator_withheld_uses_same_semantic_recovery(
    lane: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def command(argv: list[str], **_kwargs: Any) -> int:
        lane.calls.append(list(argv))
        return 75 if argv[-1] == "validate.py" else 0

    monkeypatch.setattr(lane_resume, "_run_argv", command)
    lane.board.write_text(lane.board.read_text() + "- Deterministic repair: python3 repair.py\n")
    for _ in range(3):
        result = lane.cycle()
    assert result["phase"] == "gate_recovery_needed"
    packet = json.loads(Path(result["repair_packet"]).read_text())
    assert packet["withheld_stage"] == "validation"
    assert packet["deterministic_repair"]["gate_exit_code"] == 75
    assert "- Status: pending" in lane.board.read_text()


def test_packet_excludes_artifact_rows_and_bodies(lane: SimpleNamespace) -> None:
    report = json.loads(lane.report.read_text())
    report.update(rows=[{"request_body": "private input"}], response_body="private response", inputs=["raw"])
    lane.report.write_text(json.dumps(report))
    for _ in range(3):
        result = lane.cycle()
    packet = Path(result["repair_packet"]).read_text()
    assert "private input" not in packet and "private response" not in packet
    assert "request_body" not in packet and "response_body" not in packet
    assert "semantic_sha256" in packet


def test_manifest_level_operator_hold_blocks_nested_lane(lane: SimpleNamespace) -> None:
    lane.kwargs["attempt_path"] = lane.root / "watchdog/lanes/planning/attempts.json"
    (lane.root / "watchdog").mkdir()
    (lane.root / "watchdog/operator.hold").touch()
    assert lane.cycle()["phase"] == "operator_hold"
    assert not lane.calls


def test_new_hold_between_gate_probe_and_repair_stops_followup_actions(
    lane: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for _ in range(2):
        lane.cycle()
    def command(argv: list[str], **kwargs: Any) -> int:
        result = lane.run(argv, **kwargs)
        (lane.root / "operator.hold").touch()
        return result
    monkeypatch.setattr(lane_resume, "_run_argv", command)
    before = len(lane.calls)
    assert lane.cycle()["phase"] == "operator_hold"
    assert len(lane.calls) == before + 1


@pytest.mark.parametrize("hold_after", ["gate.py", "validate.py"])
def test_new_hold_during_success_stops_validation_or_completion(
    lane: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, hold_after: str,
) -> None:
    original = lane.board.read_text()
    def command(argv: list[str], **_kwargs: Any) -> int:
        lane.calls.append(list(argv))
        if argv[-1] == hold_after:
            (lane.root / "operator.hold").touch()
        return 0
    monkeypatch.setattr(lane_resume, "_run_argv", command)
    assert lane.cycle()["phase"] == "operator_hold"
    assert lane.board.read_text() == original
    assert [argv[-1] for argv in lane.calls] == (
        ["gate.py"] if hold_after == "gate.py" else ["gate.py", "validate.py"]
    )


@pytest.mark.parametrize("hold", ["hold", "operator.hold", "HOLD", "OPERATOR_STOP"])
def test_operator_hold_blocks_all_gate_repair_actions(lane: SimpleNamespace, hold: str) -> None:
    (lane.root / hold).touch()
    assert lane.cycle()["phase"] == "operator_hold"
    assert not lane.calls


def _admit(lane: SimpleNamespace, **overrides: Any) -> dict[str, Any]:
    fields = dict(repo_root=lane.root, task_id="TEST-060", repair_task_id="TEST-R060",
                  command="python3 repair.py", validation="python3 validate.py",
                  predicted_files=["source.py"], acceptance="Real invocation receipts satisfy the coverage gate.",
                  depends_on=["TEST-040"])
    fields.update(overrides)
    return gate_recovery.admit_gate_repair_task(lane.board.read_text(), **fields)


def test_repair_admission_is_idempotent_and_preserves_original_task_statuses(lane: SimpleNamespace) -> None:
    first = _admit(lane)
    assert first["admitted"] is True
    tasks = dict(parse_todo_task_fields(first["todo_text"], task_header_prefix="## TEST-"))
    assert tasks["TEST-060"]["status"] == "pending"
    assert tasks["TEST-040"]["status"] == "completed"
    assert tasks["TEST-R060"]["status"] == "pending"
    assert tasks["TEST-060"]["depends_on"] == "TEST-040, TEST-R060"
    assert "satisfies" not in tasks["TEST-R060"]
    lane.board.write_text(first["todo_text"])
    again = _admit(lane)
    assert again["admitted"] is False
    assert again["todo_text"] == first["todo_text"]
    completed = lane.cycle()
    assert completed["advanced_task_id"] == "TEST-R060"
    assert all(argv[-1] != "gate.py" for argv in lane.calls)


@pytest.mark.parametrize("invalid", [
    {"depends_on": ["TEST-060"]},
    {"command": "grok --prompt repair"},
    {"command": "python3 repair.py; echo GO"},
    {"validation": "test -f coverage.json"},
    {"validation": "/usr/bin/test -f coverage.json"},
    {"predicted_files": ["../outside.py"]},
    {"predicted_files": ["implementation_plan/gate.todo.md"]},
])
def test_unsafe_or_false_evidence_repair_admission_is_refused(lane: SimpleNamespace, invalid: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _admit(lane, **invalid)


@pytest.mark.parametrize("drift", [
    {"predicted_files": ["aggregate.py"]}, {"acceptance": "weaken acceptance"}, {"depends_on": []},
])
def test_repair_admission_rejects_contract_drift(lane: SimpleNamespace, drift: dict[str, Any]) -> None:
    lane.board.write_text(_admit(lane)["todo_text"])
    with pytest.raises(ValueError, match="different contract"):
        _admit(lane, **drift)


def test_gate_status_preserves_semantic_repair_evidence(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    lane_resume.write_lane_status(path, target_id="lane", state="degraded", phase="gate_retry_wait",
                                  transition_at_ms=1000, gate_evidence_fingerprint="fixed",
                                  repair_packet="packet.json", next_retry_at_ms=9000,
                                  gate_recovery_needed=True)
    status = json.loads(path.read_text())
    assert status["last_task_transition_at_ms"] == 1000
    assert status["gate_evidence_fingerprint"] == "fixed"
    assert status["gate_recovery_needed"] is True
    assert status["next_retry_at_ms"] == 9000
    assert status["repair_packet"] == "packet.json"


def test_serve_does_not_renew_progress_timestamp_for_repeated_gate_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}
    monkeypatch.setattr(lane_resume.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    elapsed = [0.0]
    monkeypatch.setattr(lane_resume.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(lane_resume.time, "time", lambda: 1000.0 + elapsed[0])
    monkeypatch.setattr(lane_resume.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    seen: list[int] = []
    cycles = [0]
    original_status = lane_resume.write_lane_status

    def write(path: Path, **kwargs: Any) -> None:
        if kwargs["phase"] == "gate_withheld":
            seen.append(kwargs["transition_at_ms"])
        original_status(path, **kwargs)

    def cycle(**_kwargs: Any) -> dict[str, Any]:
        cycles[0] += 1
        if cycles[0] >= 3:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {"phase": "gate_withheld", "state": "healthy", "active_task_id": "TEST-060",
                "evidence_changed": cycles[0] == 1}

    monkeypatch.setattr(lane_resume, "run_resume_cycle", cycle)
    monkeypatch.setattr(lane_resume, "write_lane_status", write)
    assert lane_resume.serve([
        "--repo-root", str(tmp_path), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
        "--status-path", str(tmp_path / "status.json"), "--pid-path", str(tmp_path / "lane.pid"),
        "--log-path", str(tmp_path / "lane.log"), "--bundle-key", "test", "--max-seconds", "90",
    ]) == 0
    assert cycles[0] == 3
    assert len(seen) >= 2 and set(seen) == {1_000_000}


def test_serve_preserves_prior_semantic_transition_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"target_id": "test", "last_task_transition_at_ms": 1234}))
    observed: list[int] = []
    original = lane_resume.write_lane_status
    def write(path: Path, **kwargs: Any) -> None:
        observed.append(kwargs["transition_at_ms"])
        original(path, **kwargs)
    monkeypatch.setattr(lane_resume, "write_lane_status", write)
    monkeypatch.setattr(lane_resume, "run_resume_cycle", lambda **_kwargs: {
        "phase": "deterministic_resume_unavailable", "state": "blocked",
    })
    assert lane_resume.serve([
        "--repo-root", str(tmp_path), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
        "--status-path", str(status_path), "--pid-path", str(tmp_path / "lane.pid"),
        "--log-path", str(tmp_path / "lane.log"), "--bundle-key", "test", "--max-seconds", "1",
    ]) == 0
    assert observed == [1234]


def test_admitted_browser_repair_preserves_its_two_hour_bound(lane: SimpleNamespace) -> None:
    result = _admit(lane, resource_class="cpu-browser", implementation_timeout_seconds=7200)
    rows = dict(parse_todo_task_fields(result["todo_text"], task_header_prefix="## TEST-"))
    repair = rows[result["derived_task_id"]]
    assert repair["resource_class"] == "cpu-browser"
    assert repair["implementation_timeout_seconds"] == "7200"
    lane.board.write_text(result["todo_text"])
    assert _admit(lane, resource_class="cpu-browser", implementation_timeout_seconds=7200)["reason"] == "duplicate_refill"
    with pytest.raises(ValueError, match="different contract"):
        _admit(lane)


def test_admission_cannot_assign_a_browser_bound_to_cpu_small(lane: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="resource bound"):
        _admit(lane, implementation_timeout_seconds=7200)


def test_an_admitted_repair_can_receive_a_sibling_repair_dependency(lane: SimpleNamespace) -> None:
    first = _admit(lane)
    lane.board.write_text(first["todo_text"])
    result = _admit(lane, task_id=first["derived_task_id"], repair_task_id="TEST-041-R01")
    rows = dict(parse_todo_task_fields(result["todo_text"], task_header_prefix="## TEST-"))
    assert "TEST-041-R01" in rows[first["derived_task_id"]]["depends_on"]
    assert rows["TEST-041-R01"]["depends_on"] == "TEST-040"
    assert rows[first["derived_task_id"]]["status"] == "pending"
    assert rows["TEST-060"]["status"] == "pending"


def test_fixed_point_selects_new_source_repair_without_completing_gate(lane: SimpleNamespace) -> None:
    for _ in range(3):
        stalled = lane.cycle()
    assert stalled["phase"] == "gate_recovery_needed"
    packet_path = Path(stalled["repair_packet"])
    packet = packet_path.read_bytes()
    original_report = json.loads(lane.report.read_text())
    admitted = _admit(lane)
    lane.board.write_text(admitted["todo_text"])
    lane.calls.clear()
    resumed = lane.cycle()
    assert resumed["advanced_task_id"] == "TEST-R060"
    assert all(argv[-1] != "gate.py" for argv in lane.calls)
    tasks = dict(parse_todo_task_fields(lane.board.read_text(), task_header_prefix="## TEST-"))
    assert tasks["TEST-040"]["status"] == "completed"
    assert tasks["TEST-060"]["status"] == "pending"
    assert json.loads(lane.report.read_text())["decision"] == original_report["decision"] == "NO_GO"
    assert packet_path.read_bytes() == packet
