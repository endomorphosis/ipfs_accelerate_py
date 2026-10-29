"""Real producer boundaries distinguish resource refusal from application failure."""
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise durable contention cooldown with real producer and validator processes")


def setup(root, *, exit_code=75, task_id="TEST-001", typed=True, packet=None):
    diagnostic = {"task_id": task_id, "error": {"code": "browser_contention"}} if typed else {"error": "browser_contention"}
    if packet is not None:
        diagnostic = packet
    (root / "busy").touch()
    (root / "producer.py").write_text("from pathlib import Path\nimport json\n"
        "p=Path('runs');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        f"if Path('busy').exists():\n print({json.dumps(diagnostic)!r})\n raise SystemExit({exit_code})\n"
        "Path('report.json').write_text('{\"actual\":42}')\n")
    (root / "validator.py").write_text("from pathlib import Path\nimport json\n"
        "assert json.loads(Path('report.json').read_text())['actual']==42\nPath('validated').touch()\n")
    (root / "board.md").write_text("## TEST-001 Real browser workflow\n- Status: ready\n"
        "- Deterministic command: python3 producer.py\n- Validation: python3 validator.py\n- Outputs: report.json\n")


def cycle(root):
    return lane_resume.run_resume_cycle(repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
        attempt_path=root / "attempts.json", log_path=root / "lane.log", heartbeat=lambda **_: None)


def progress(root):
    return lane_resume.execution_progress_snapshot(root / "attempts.json", "TEST-001")


def test_contention_waits_without_application_failure_validation_or_paid_repair(tmp_path):
    setup(tmp_path)
    first = cycle(tmp_path)
    assert first['phase'] == 'resource_retry_wait' and first['active_task_id'] == ''
    for _ in range(3):
        assert cycle(tmp_path)['phase'] == 'resource_retry_wait'
    assert (tmp_path / 'runs').read_text() == '1'
    state = progress(tmp_path)
    assert state['semantic_failures'] == state['gate_refusals'] == 0
    assert state['resource_refusals'] == 1
    assert not (tmp_path / 'validated').exists()
    assert not (tmp_path / 'attempts.semantic-recovery.json').exists()
    assert not (tmp_path / 'attempts.gates.json').exists()
    assert '- Status: ready' in (tmp_path / 'board.md').read_text()


def test_resource_available_retries_original_producer_and_validator_after_cooldown(tmp_path, monkeypatch):
    setup(tmp_path)
    first = cycle(tmp_path)
    (tmp_path / 'busy').unlink()
    monkeypatch.setattr(lane_resume.time, 'time', lambda: first['next_retry_at_ms'] / 1000 + 1)
    assert cycle(tmp_path)['advanced_task_id'] == 'TEST-001'
    assert (tmp_path / 'runs').read_text() == '2' and (tmp_path / 'validated').exists()
    assert progress(tmp_path)['state'] == 'validated'


def test_contention_keeps_prior_semantic_failures_and_spent_recovery_history(tmp_path):
    setup(tmp_path)
    fields = {'deterministic_command': 'python3 producer.py', 'validation': 'python3 validator.py', 'outputs': 'report.json'}
    contract = lane_resume._command_contract(fields, ['python3','producer.py'], ['python3','validator.py'], repo_root=tmp_path)
    prior = {'execution_progress': {'contract':contract,'state':'failed','semantic_failures':2,'attempts_started':2,
        'unresolved_prior_failure':True}}
    lane_resume._store_attempt_state(tmp_path/'attempts.json', {'TEST-001':2}, {}, {'TEST-001':prior}, {})
    journal = tmp_path / 'attempts.semantic-recovery.json'
    journal.write_text(json.dumps({'schema':'agent-supervisor/semantic-recovery-journal@1','tasks':{'TEST-001':{
        'spent':1,'history':[{'qualification_fingerprint':'a'*64,'state':'failed','reason':'old','recovery_limit':2}]}}}))
    before = journal.read_bytes()
    assert cycle(tmp_path)['phase'] == 'resource_retry_wait'
    assert progress(tmp_path)['semantic_failures'] == 2
    assert progress(tmp_path)['unresolved_prior_failure'] is True
    assert json.loads((tmp_path/'attempts.json').read_text())['TEST-001'] == 2
    assert journal.read_bytes() == before


def test_bounded_resource_window_retries_after_cooldown_without_source_or_budget_reset(tmp_path, monkeypatch):
    setup(tmp_path)
    board = tmp_path/'board.md'
    board.write_text(board.read_text()+'- Implementation timeout seconds: 1200\n')
    for index in range(lane_resume._RESOURCE_RETRY_LIMIT):
        result = cycle(tmp_path)
        assert result['phase'] == 'resource_retry_wait'
        if index < lane_resume._RESOURCE_RETRY_LIMIT - 1:
            instant = result['next_retry_at_ms'] / 1000 + 1
            monkeypatch.setattr(lane_resume.time, 'time', lambda instant=instant: instant)
    assert result['resource_window_exhausted'] is True
    contract = progress(tmp_path)['contract']
    for _ in range(2):
        assert cycle(tmp_path)['phase'] == 'resource_retry_wait'
    assert int((tmp_path/'runs').read_text()) == lane_resume._RESOURCE_RETRY_LIMIT
    assert progress(tmp_path)['semantic_failures'] == progress(tmp_path)['gate_refusals'] == 0
    (tmp_path/'busy').unlink()
    instant = result['next_retry_at_ms'] / 1000 + 1
    monkeypatch.setattr(lane_resume.time, 'time', lambda: instant)
    assert cycle(tmp_path)['advanced_task_id'] == 'TEST-001'
    assert (tmp_path/'validated').exists()
    state = json.loads((tmp_path/'attempts.json').read_text())['_validation_fixed_points']['TEST-001']
    assert state['execution_progress']['contract'] == contract
    assert state['resource_refusals_total'] == lane_resume._RESOURCE_RETRY_LIMIT
    assert state['resource_wait_history'][0]['refusals'] == lane_resume._RESOURCE_RETRY_LIMIT
    assert state['resource_wait_history'][0]['end_reason'] == 'bounded_window_cooldown_finished'
    assert not (tmp_path/'attempts.semantic-recovery.json').exists()


@pytest.mark.parametrize('task_id,typed', [('OTHER-001',True),('TEST-001',False)])
def test_unbound_or_untyped_exit75_stays_ordinary_gate(tmp_path, task_id, typed):
    setup(tmp_path, task_id=task_id, typed=typed)
    assert cycle(tmp_path)['phase'] != 'resource_retry_wait'
    assert progress(tmp_path)['state'] == 'gate_withheld'


def test_real_application_exit_is_not_reclassified_by_contention_text(tmp_path):
    setup(tmp_path, exit_code=1)
    assert cycle(tmp_path)['phase'] == 'deterministic_command_failed'
    assert progress(tmp_path)['semantic_failures'] == 1


def test_old_task_diagnostic_does_not_classify_current_untyped_gate(tmp_path):
    setup(tmp_path, typed=False)
    (tmp_path/'lane.log').write_text(json.dumps({'task_id':'TEST-001','error':{'code':'browser_contention'}})+'\n')
    assert cycle(tmp_path)['phase'] != 'resource_retry_wait'
    assert progress(tmp_path)['state'] == 'gate_withheld'


def storage_packet(reason="ENOSPC", number=28):
    return {"task_id": "TEST-001", "error": {"code": "storage_pressure", "reason": reason, "errno": number},
            "retryable": True, "work_complete": False, "completion_authority": False}


@pytest.mark.parametrize("reason,number", [("ENOSPC", 28), ("EDQUOT", 122), ("insufficient_free_bytes", None)])
def test_storage_pressure_preserves_budgets_then_retries_original_validator(tmp_path, monkeypatch, reason, number):
    setup(tmp_path, packet=storage_packet(reason, number))
    fields = {"deterministic_command": "python3 producer.py", "validation": "python3 validator.py", "outputs": "report.json"}
    contract = lane_resume._command_contract(fields, ["python3", "producer.py"], ["python3", "validator.py"], repo_root=tmp_path)
    prior = {"execution_progress": {"contract": contract, "state": "failed", "semantic_failures": 2,
             "attempts_started": 2, "gate_refusals": 1, "unresolved_prior_failure": True}}
    lane_resume._store_attempt_state(tmp_path / "attempts.json", {"TEST-001": 2}, {}, {"TEST-001": prior}, {})
    # Refusal is before any model/phase reservation. Historical recovery evidence must survive byte-for-byte.
    phase = tmp_path / "phase-budget.json"
    phase.write_text('{"spent":4,"limit":6}')
    journal = tmp_path / "attempts.semantic-recovery.json"
    journal.write_text(json.dumps({"schema": "agent-supervisor/semantic-recovery-journal@1", "tasks": {"TEST-001": {
        "spent": 1, "history": [{"qualification_fingerprint": "a" * 64, "state": "failed", "reason": "old", "recovery_limit": 2}]}}}))
    preserved = {path: path.read_bytes() for path in (phase, journal)}
    first = cycle(tmp_path)
    assert first["phase"] == "resource_retry_wait"
    assert first["backpressure_reasons"] == ["TEST-001:storage_pressure"]
    assert first["work_complete"] is first["completion_authority"] is False
    for _ in range(2):
        assert cycle(tmp_path)["phase"] == "resource_retry_wait"
    state = progress(tmp_path)
    assert (state["semantic_failures"], state["gate_refusals"], state["resource_refusals"]) == (2, 1, 1)
    assert state["unresolved_prior_failure"] is True
    assert json.loads((tmp_path / "attempts.json").read_text())["TEST-001"] == 2
    assert (tmp_path / "runs").read_text() == "1"
    assert all(path.read_bytes() == before for path, before in preserved.items())
    assert not (tmp_path / "validated").exists()
    assert not (tmp_path / "attempts.gates.json").exists()
    assert "- Status: ready" in (tmp_path / "board.md").read_text()
    (tmp_path / "busy").unlink()
    monkeypatch.setattr(lane_resume.time, "time", lambda: first["next_retry_at_ms"] / 1000 + 1)
    assert cycle(tmp_path)["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "runs").read_text() == "2"
    assert (tmp_path / "validated").exists()
    assert progress(tmp_path)["state"] == "validated"
    assert all(path.read_bytes() == before for path, before in preserved.items())


@pytest.mark.parametrize("change", [
    {"task_id": "OTHER-001"}, {"retryable": False}, {"retryable": 1},
    {"work_complete": True}, {"work_complete": 0}, {"completion_authority": True},
    {"completion_authority": 0}, {"error": {"code": "storage_pressure"}},
    {"error": {"code": "resource_unavailable", "reason": "ENOSPC", "errno": 28}},
    {"error": {"code": "storage_pressure", "reason": "EACCES", "errno": 13}},
    {"error": {"code": "storage_pressure", "reason": "ENOSPC", "errno": 122}},
    {"error": {"code": "storage_pressure", "reason": "EDQUOT", "errno": 28}},
    {"error": {"code": "storage_pressure", "reason": "ENOSPC", "errno": "28"}},
    {"error": {"code": "storage_pressure", "reason": "ENOSPC", "errno": True}},
    {"error": {"code": "storage_pressure", "reason": "insufficient_free_bytes", "errno": 28}},
    {"error": {"code": "storage_pressure", "reason": "insufficient_free_bytes"}},
])
def test_malformed_or_unrelated_storage_packet_remains_gate(tmp_path, change):
    packet = storage_packet()
    packet.update(change)
    setup(tmp_path, packet=packet)
    assert cycle(tmp_path)["phase"] != "resource_retry_wait"
    assert progress(tmp_path)["state"] == "gate_withheld"
    assert progress(tmp_path)["gate_refusals"] == 1


def test_storage_packet_requires_exit75_and_current_invocation(tmp_path):
    setup(tmp_path, packet=storage_packet(), exit_code=1)
    assert cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert progress(tmp_path)["semantic_failures"] == 1


def test_storage_packet_before_current_log_offset_is_not_reused(tmp_path):
    log = tmp_path / "lane.log"
    old = json.dumps(storage_packet()) + "\n"
    log.write_text(old + '{"task_id":"TEST-001","error":{"code":"validation_failed"}}\n')
    assert lane_resume._transient_resource_refusal(log, len(old.encode()), "TEST-001") == ""
    log.write_text(old + 'not JSON\n')
    assert lane_resume._transient_resource_refusal(log, len(old.encode()), "TEST-001") == ""


def test_storage_packet_requires_explicit_authority_and_retry_flags(tmp_path):
    log = tmp_path / "lane.log"
    for key in ("retryable", "work_complete", "completion_authority"):
        packet = storage_packet()
        packet.pop(key)
        log.write_text(json.dumps(packet) + "\n")
        assert lane_resume._transient_resource_refusal(log, 0, "TEST-001") == ""
    log.write_text(json.dumps(storage_packet()) + "\n" + "x" * 131072)
    assert lane_resume._transient_resource_refusal(log, 0, "TEST-001") == ""


def test_storage_pressure_before_semantic_reservation_keeps_all_history(tmp_path, monkeypatch):
    import test_agent_supervisor_semantic_recovery as semantic_fixture
    semantic_fixture.task(tmp_path)
    before = (tmp_path / "attempts.json").read_bytes()
    original_check = lane_resume.check_storage_capacity
    calls = []
    def pressure(paths):
        calls.append(tuple(paths))
        return {"ok": False, "storage_pressure": True, "reason": "ENOSPC", "errno": 28}
    monkeypatch.setattr(lane_resume, "check_storage_capacity", pressure)
    for _ in range(3):
        result = semantic_fixture.cycle(tmp_path)
        assert result["phase"] == "resource_retry_wait"
        assert result["resource_code"] == "storage_pressure" and result["next_retry_at_ms"] > 0
        assert result["work_complete"] is result["completion_authority"] is False
    assert len(calls) == 3
    assert all(paths == (tmp_path, tmp_path, tmp_path) for paths in calls)
    assert (tmp_path / "attempts.json").read_bytes() == before
    assert not (tmp_path / "attempts.semantic-recovery.json").exists()
    assert not list(tmp_path.glob("semantic-recovery-*.context.json"))
    assert not (tmp_path / "recovery-count").exists() and not (tmp_path / "producer-count").exists()
    monkeypatch.setattr(lane_resume, "check_storage_capacity", original_check)
    assert semantic_fixture.cycle(tmp_path)["phase"] == "semantic_recovery_applied"
    assert semantic_fixture.summary(tmp_path)["spent"] == 1
    assert json.loads((tmp_path / "attempts.json").read_text())["TEST-001"] == 6
    assert semantic_fixture.cycle(tmp_path)["advanced_task_id"] == "TEST-001"
    assert semantic_fixture.summary(tmp_path)["spent"] == 1


def test_recovery_pressure_never_rewrites_existing_spent_journal(tmp_path, monkeypatch):
    import test_agent_supervisor_semantic_recovery as semantic_fixture
    semantic_fixture.task(tmp_path, body="pass\n", limit=2)
    assert semantic_fixture.cycle(tmp_path)["phase"] == "semantic_recovery_noop"
    journal = tmp_path / "attempts.semantic-recovery.json"
    prior = journal.read_bytes()
    monkeypatch.setattr(lane_resume, "check_storage_capacity", lambda _: {
        "ok": False, "storage_pressure": True, "reason": "EDQUOT", "errno": 122})
    fields = semantic_fixture._fields(tmp_path)
    result = lane_resume._run_semantic_recovery("TEST-001", fields, ["python3", "produce.py"],
        ["python3", "validate.py"], repo_root=tmp_path, todo_path=tmp_path / "board.md",
        attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "lane.log", progress={},
        heartbeat=lambda **_: None, hold_result=lambda: None)
    assert result["phase"] == "resource_retry_wait"
    assert journal.read_bytes() == prior
    assert (tmp_path / "recovery-count").read_text() == "1"


def test_gate_recovery_capacity_precedes_journal_and_repair_side_effects(tmp_path, monkeypatch):
    setup(tmp_path, typed=False)
    (tmp_path / "repair.py").write_text("from pathlib import Path\nPath('repair-count').write_text('1')\n")
    board = tmp_path / "board.md"
    board.write_text(board.read_text() + "- Deterministic repair: python3 repair.py\n")
    cycle(tmp_path)
    cycle(tmp_path)
    journal = tmp_path / "attempts.gates.json"
    prior = journal.read_bytes()
    original_check = lane_resume.check_storage_capacity
    monkeypatch.setattr(lane_resume, "check_storage_capacity", lambda _: {
        "ok": False, "storage_pressure": True, "reason": "insufficient_free_bytes"})
    for _ in range(3):
        result = cycle(tmp_path)
        assert result["phase"] == "resource_retry_wait" and result["storage_pressure"] is True
        assert journal.read_bytes() == prior
    assert not (tmp_path / "repair-count").exists()
    monkeypatch.setattr(lane_resume, "check_storage_capacity", original_check)
    assert cycle(tmp_path)["phase"] == "gate_recovery_needed"
    assert (tmp_path / "repair-count").read_text() == "1"
    saved = json.loads(journal.read_text())
    assert saved["TEST-001"]["repair_attempted"] is True


def test_recovery_capacity_does_not_disguise_permission_failure(tmp_path, monkeypatch):
    def permission_denied(_):
        raise PermissionError(13, "permission denied")
    monkeypatch.setattr(lane_resume, "check_storage_capacity", permission_denied)
    with pytest.raises(PermissionError):
        lane_resume._recovery_storage_wait("TEST-001", repo_root=tmp_path,
            attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "lane.log")


def test_capacity_lost_during_gate_dependency_probe_does_not_reserve_repair(tmp_path, monkeypatch):
    setup(tmp_path, typed=False)
    (tmp_path / "dependency.py").write_text("from pathlib import Path\nPath('dependency-probed').touch()\n")
    (tmp_path / "repair.py").write_text("from pathlib import Path\nPath('repair-count').write_text('1')\n")
    board = tmp_path / "board.md"
    text = board.read_text().replace("- Status: ready\n", "- Status: ready\n- Depends on: TEST-000\n")
    board.write_text("## TEST-000 Existing native prerequisite\n- Status: completed\n"
        "- Validation: python3 dependency.py\n- Predicted files: dependency.py\n\n" + text
        + "- Deterministic repair: python3 repair.py\n")
    cycle(tmp_path)
    cycle(tmp_path)
    original_check = lane_resume.check_storage_capacity
    def capacity(paths):
        if (tmp_path / "dependency-probed").exists():
            return {"ok": False, "storage_pressure": True, "reason": "ENOSPC", "errno": 28}
        return original_check(paths)
    monkeypatch.setattr(lane_resume, "check_storage_capacity", capacity)
    result = cycle(tmp_path)
    assert (tmp_path / "dependency-probed").exists()
    assert result["phase"] == "resource_retry_wait" and result["storage_reason"] == "ENOSPC"
    assert not (tmp_path / "repair-count").exists()
    saved = json.loads((tmp_path / "attempts.gates.json").read_text())["TEST-001"]
    assert saved["dependencies_revalidated"] is True
    assert not saved.get("repair_attempted")
    assert saved["unchanged_probes"] == 3  # Admitted pre-pressure observations stay monotonic.
    monkeypatch.setattr(lane_resume, "check_storage_capacity", original_check)
    instant = result["next_retry_at_ms"] / 1000 + 1000
    monkeypatch.setattr(lane_resume.time, "time", lambda: instant)
    assert cycle(tmp_path)["phase"] == "gate_recovery_needed"
    assert (tmp_path / "repair-count").read_text() == "1"



@pytest.mark.parametrize("suffix", ["", '{"task_id":"TEST-001","error":', "x" * 131073], ids=["empty", "truncated", "oversized"])
def test_lost_refusal_log_uses_only_fresh_measured_storage_capacity(tmp_path, monkeypatch, suffix):
    setup(tmp_path, typed=False)
    # A real producer exits75, but a full output filesystem lost its refusal.
    (tmp_path / "producer.py").write_text("raise SystemExit(75)\n")
    log = tmp_path / "lane.log"
    log.write_text(suffix)
    measurements = []
    def pressure(paths):
        measurements.append(tuple(paths))
        return {"ok": False, "storage_pressure": True, "reason": "insufficient_free_bytes"}
    monkeypatch.setattr(lane_resume, "check_storage_capacity", pressure)
    assert lane_resume._transient_resource_refusal(log, 0, "TEST-001") == "storage_pressure"
    assert measurements == [(log,)]
    result = cycle(tmp_path)
    assert result["phase"] == "resource_retry_wait"
    state = progress(tmp_path)
    assert state["semantic_failures"] == state["gate_refusals"] == 0
    assert state["resource_refusals"] == 1
    assert not (tmp_path / "attempts.gates.json").exists()
    assert not (tmp_path / "attempts.semantic-recovery.json").exists()
    assert not (tmp_path / "validated").exists()


@pytest.mark.parametrize("suffix", ["", '{"task_id":"TEST-001","error":',
    '{"task_id":"OTHER-001","error":{"code":"storage_pressure","reason":"ENOSPC","errno":28}}',
    "x" * 131073], ids=["empty", "truncated", "wrong-task", "oversized"])
def test_missing_or_unbound_packet_is_not_storage_when_actual_capacity_healthy(tmp_path, monkeypatch, suffix):
    log = tmp_path / "lane.log"
    log.write_text(suffix)
    observed = []
    def healthy(paths):
        observed.append(tuple(paths))
        return {"ok": True, "storage_pressure": False, "reason": ""}
    monkeypatch.setattr(lane_resume, "check_storage_capacity", healthy)
    assert lane_resume._transient_resource_refusal(log, 0, "TEST-001") == ""
    assert observed == [(log,)]


@pytest.mark.parametrize("packet", [
    {"task_id": "TEST-001", "error": {"code": "permission_denied"}},
    {"task_id": "TEST-001", "error": {"code": "storage_pressure", "reason": "ENOSPC", "errno": 13}},
    {"task_id": "TEST-001", "error": "storage_pressure"},
])
def test_current_malformed_or_unrelated_diagnostic_never_uses_capacity_fallback(tmp_path, monkeypatch, packet):
    log = tmp_path / "lane.log"
    log.write_text(json.dumps(packet) + "\n")
    def unexpected(_):
        pytest.fail("Current task failure must not be replaced by a capacity measurement")
    monkeypatch.setattr(lane_resume, "check_storage_capacity", unexpected)
    assert lane_resume._transient_resource_refusal(log, 0, "TEST-001") == ""



@pytest.mark.parametrize("failure", ["insufficient_free_bytes", "ENOSPC", "EDQUOT"])
def test_lost_packet_real_capacity_api_admits_no_repair_and_healthy_identical_loss_rejects(tmp_path, monkeypatch, failure):
    import errno
    import os
    from ipfs_accelerate_py.agent_supervisor.rescue import storage_pressure
    full, healthy = tmp_path / "full", tmp_path / "healthy"
    full.mkdir()
    healthy.mkdir()
    for root in (full, healthy):
        setup(root, typed=False)
        (root / "producer.py").write_text("raise SystemExit(75)\n")
    with monkeypatch.context() as patch:
        if failure == "insufficient_free_bytes":
            real_statvfs = storage_pressure.os.statvfs
            def no_free_blocks(path):
                values = list(real_statvfs(path))
                values[4] = 0  # f_bavail; exercise the real clean API's exact low-space branch.
                return os.statvfs_result(values)
            patch.setattr(storage_pressure.os, "statvfs", no_free_blocks)
        else:
            def failed_durable_probe(_):
                raise OSError(getattr(errno, failure), "injected space failure")
            patch.setattr(storage_pressure, "_durable_probe", failed_durable_probe)
        result = cycle(full)
        assert result["phase"] == "resource_retry_wait"
        assert result["backpressure_reasons"] == ["TEST-001:storage_pressure"]
        assert progress(full)["semantic_failures"] == progress(full)["gate_refusals"] == 0
        assert not (full / "attempts.semantic-recovery.json").exists()
        assert not (full / "attempts.gates.json").exists()
    assert cycle(healthy)["phase"] != "resource_retry_wait"
    assert progress(healthy)["state"] == "gate_withheld"
    assert progress(healthy)["gate_refusals"] == 1
