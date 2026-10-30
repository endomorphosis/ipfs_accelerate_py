"""Private source-pinned preflight tests; all artifacts live in tmp_path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume as lane
from ipfs_accelerate_py.agent_supervisor.rescue import vdg_pre_command_recovery as recovery

pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise actual attempt and command-contract boundaries")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy(root):
    script = root / recovery.SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("// actual private recovery implementation\n")
    value = {"schema": recovery.SCHEMA, "task_id": recovery.TASK_ID,
             "producer_argv": list(recovery.PRODUCER), "script": recovery.SCRIPT,
             "script_sha256": sha(script), "module_sha256": sha(Path(recovery.__file__)),
             "transports": ["http", "libp2p"], "timeout_seconds": 40,
             "require_uninterrupted_after_producer": True}
    store_policy(root, value)
    return value


def store_policy(root, value, **extra):
    path = root / recovery.MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lanes": [{"bundle_key": "vdg-planning", "pre_command_recovery": value}], **extra}))


def task(root):
    command = " ".join(recovery.PRODUCER)
    (root / "validate.py").write_text("# private exact validator\n")
    for relative in (recovery.PRODUCER[1], recovery.PRODUCER[4]):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// private producer\n")
    (root / "board.md").write_text(
        f"## {recovery.TASK_ID} Actual acquisition task\n- Status: pending\n- Depends on:\n"
        f"- Deterministic command: {command}\n- Validation: python3 validate.py\n"
        "- Resource class: cpu-browser\n- Implementation timeout seconds: 100\n"
        "- Outputs: report.json\n- Acceptance: actual source-bound acquisition\n")


def cycle(root, heartbeat=None):
    return lane.run_resume_cycle(repo_root=root, todo_path=root / "board.md", task_header_prefix="## VDG-",
                                 attempt_path=root / "attempts.json", log_path=root / "lane.log", heartbeat=heartbeat or (lambda **_: None))


def fields(root):
    return {"deterministic_command": " ".join(recovery.PRODUCER), "validation": "python3 validate.py", "outputs": "report.json"}


def test_missing_configuration_leaves_ordinary_producer_inert(tmp_path):
    assert recovery.recovery_contract(recovery.PRODUCER, repo_root=tmp_path) is None
    assert recovery.recovery_commands(recovery.TASK_ID, recovery.PRODUCER, repo_root=tmp_path) == []


def test_exact_owner_task_and_producer_only(tmp_path):
    policy(tmp_path)
    hooks = recovery.recovery_commands(recovery.TASK_ID, recovery.PRODUCER, repo_root=tmp_path)
    assert len(hooks) == 2
    assert [argv[argv.index("--transport") + 1] for argv, _ in hooks] == ["http", "libp2p"]
    assert all(argv[argv.index("--project-root") + 1] == str(tmp_path / "swissknife") for argv, _ in hooks)
    assert recovery.recovery_commands("VDG-060", recovery.PRODUCER, repo_root=tmp_path) == []
    assert recovery.recovery_commands(recovery.TASK_ID, [*recovery.PRODUCER[:-1], "1"], repo_root=tmp_path) == []


@pytest.mark.parametrize("change", [
    {"task_id": "VDG-060"}, {"script": "outside.py"}, {"transports": ["http"]},
    {"timeout_seconds": True}, {"timeout_seconds": 301}, {"extra": "undeclared"},
    {"script_sha256": "0" * 64}, {"module_sha256": "0" * 64},
    {"require_uninterrupted_after_producer": False},
])
def test_malformed_or_unpinned_policy_refuses_before_any_command(tmp_path, change):
    value = policy(tmp_path)
    value.update(change)
    store_policy(tmp_path, value)
    with pytest.raises(recovery.RecoveryPolicyError):
        recovery.recovery_commands(recovery.TASK_ID, recovery.PRODUCER, repo_root=tmp_path)


def test_config_and_actual_implementation_bind_contract_without_timestamp_churn(tmp_path):
    value = policy(tmp_path)
    original = lane._command_contract(fields(tmp_path), recovery.PRODUCER, ["python3", "validate.py"], repo_root=tmp_path)
    store_policy(tmp_path, value, updated_at="a later unrelated daemon timestamp")
    assert lane._command_contract(fields(tmp_path), recovery.PRODUCER, ["python3", "validate.py"], repo_root=tmp_path) == original
    (tmp_path / recovery.SCRIPT).write_text("// genuinely changed implementation\n")
    assert lane._command_contract(fields(tmp_path), recovery.PRODUCER, ["python3", "validate.py"], repo_root=tmp_path) != original
    with pytest.raises(recovery.RecoveryPolicyError, match="source pin"):
        recovery.recovery_commands(recovery.TASK_ID, recovery.PRODUCER, repo_root=tmp_path)


def test_closeout_contract_unchanged_by_hook_policy_or_implementation(tmp_path):
    argv = ["node", "swissknife/scripts/run-with-vdg-openrouter-catalog-input.mjs", "--", "python3", "swissknife/scripts/vdg_closeout_requires_go.py"]
    original = lane._command_contract({}, argv, argv, repo_root=tmp_path)
    policy(tmp_path)
    (tmp_path / recovery.SCRIPT).write_text("// later recovery implementation\n")
    assert lane._command_contract({}, argv, argv, repo_root=tmp_path) == original


def test_loaded_old_hook_source_refuses_even_when_disk_pin_is_current(tmp_path, monkeypatch):
    policy(tmp_path)
    monkeypatch.setattr(recovery, "_LOADED_MODULE_SHA256", "0" * 64)
    with pytest.raises(recovery.RecoveryPolicyError, match="loaded source"):
        recovery.recovery_commands(recovery.TASK_ID, recovery.PRODUCER, repo_root=tmp_path)


def test_hooks_run_before_original_and_time_is_deducted_from_existing_budget(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(lane.time, "monotonic", lambda: clock[0])
    calls = []
    def execute(argv, **options):
        calls.append((list(argv), options["timeout"]))
        if "--transport" in argv:
            clock[0] += 7 if argv[argv.index("--transport") + 1] == "http" else 11
        elif list(argv) == list(recovery.PRODUCER):
            (tmp_path / "report.json").write_text('{"actual":"fresh"}')
        return 0
    monkeypatch.setattr(lane, "_run_argv", execute)
    result = cycle(tmp_path)
    assert result["phase"] == "task_command_passed"
    assert [limit for _, limit in calls[:3]] == [40.0, 40.0, 82.0]
    assert calls[2][0] == list(recovery.PRODUCER)
    assert all("--require-clean" in argv for argv, _ in calls[3:5])
    assert calls[5][0] == ["python3", "validate.py"]
    assert "- Status: completed" in (tmp_path / "board.md").read_text()


@pytest.mark.parametrize("exit_code,recorded", [(1, 1), (75, 1), (124, 124)])
def test_hook_failure_is_genuine_ordinary_attempt_never_gate_or_false_completion(tmp_path, monkeypatch, exit_code, recorded):
    policy(tmp_path); task(tmp_path); calls = []
    monkeypatch.setattr(lane, "_run_argv", lambda argv, **_: calls.append(list(argv)) or exit_code)
    result = cycle(tmp_path)
    assert result["phase"] == "deterministic_command_failed"
    assert len(calls) == 1 and "--transport" in calls[0]
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved[recovery.TASK_ID] == 1
    assert saved["_command_evidence"][recovery.TASK_ID]["exit_code"] == recorded
    assert saved["_command_evidence"][recovery.TASK_ID]["failed_stage"] == "pre_command_recovery"
    assert saved["_command_evidence"][recovery.TASK_ID]["producer_invoked"] is False
    assert "- Status: pending" in (tmp_path / "board.md").read_text()
    assert not (tmp_path / "attempts.gates.json").exists()


def test_invalid_configuration_fails_durably_without_crashing_or_running_producer(tmp_path, monkeypatch):
    value = policy(tmp_path); task(tmp_path); value["script_sha256"] = "0" * 64; store_policy(tmp_path, value)
    monkeypatch.setattr(lane, "_run_argv", lambda *_a, **_k: pytest.fail("unpinned hook reached a subprocess"))
    result = cycle(tmp_path)
    assert result["phase"] == "deterministic_command_failed"
    assert json.loads((tmp_path / "attempts.json").read_text())[recovery.TASK_ID] == 1


def test_unchanged_prelaunch_refusal_reaches_bounded_backoff_without_fake_popen(tmp_path, monkeypatch):
    value = policy(tmp_path); task(tmp_path); value["script_sha256"] = "0" * 64; store_policy(tmp_path, value)
    monkeypatch.setattr(lane, "_run_argv", lambda *_a, **_k: pytest.fail("no actual child should be pretended"))
    for _ in range(3):
        assert cycle(tmp_path)["phase"] == "deterministic_command_failed"
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved[recovery.TASK_ID] == 3
    assert saved["_retry_after_ms"][recovery.TASK_ID] > time.time() * 1000
    assert saved["_command_evidence"][recovery.TASK_ID]["producer_invoked"] is False
    assert cycle(tmp_path)["phase"] == "attempt_retry_wait"
    assert json.loads((tmp_path / "attempts.json").read_text())[recovery.TASK_ID] == 3


def test_policy_or_source_replaced_after_first_hook_cannot_reach_second_or_producer(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path); calls = []
    def execute(argv, **_):
        calls.append(argv)
        (tmp_path / recovery.SCRIPT).write_text("// replaced during real invocation\n")
        return 0
    monkeypatch.setattr(lane, "_run_argv", execute)
    assert cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert len(calls) == 1


def test_existing_operator_hold_prevents_hook_and_preserves_ledger(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path)
    saved = b'{"VDG-039-R01":2,"_retry_after_ms":{}}\n'
    (tmp_path / "attempts.json").write_bytes(saved)
    (tmp_path / "HOLD").write_text("owned intentional hold\n")
    monkeypatch.setattr(lane, "_run_argv", lambda *_a, **_k: pytest.fail("hold reached a subprocess"))
    assert cycle(tmp_path)["phase"] == "operator_hold"
    assert (tmp_path / "attempts.json").read_bytes() == saved


def test_old_watchdog_missing_mode_uses_actual_wrapped_capture_environment_without_owner_mutation(monkeypatch):
    monkeypatch.delenv("VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT", raising=False)
    monkeypatch.setenv("SWISSKNIFE_IPFS_KIT_MCP_URL", "http://owned-private-fixture:18014")
    env = recovery.recovery_environment()
    assert env["VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT"] == "1"
    assert env["SWISSKNIFE_IPFS_KIT_MCP_URL"] == "http://owned-private-fixture:18014"
    assert "VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT" not in os.environ


def test_conflicting_explicit_snapshot_environment_cannot_run_hook_or_producer(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path)
    monkeypatch.setenv("VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT", "0")
    monkeypatch.setattr(lane, "_run_argv", lambda *_a, **_k: pytest.fail("conflicting mode reached a subprocess"))
    assert cycle(tmp_path)["phase"] == "deterministic_command_failed"


def test_cancellation_unwinds_hook_without_attempt_or_producer_receipt(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path)
    def cancel(*_, **__):
        raise lane._LaneResumeInterrupted
    monkeypatch.setattr(lane, "_run_argv", cancel)
    with pytest.raises(lane._LaneResumeInterrupted):
        cycle(tmp_path)
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert recovery.TASK_ID not in saved
    assert recovery.TASK_ID not in saved.get("_command_evidence", {})


def install_actual_hook(root, body, *, timeout=40):
    value = policy(root); task(root)
    (root / recovery.SCRIPT).write_text(body)
    value.update(script_sha256=sha(root / recovery.SCRIPT), timeout_seconds=timeout)
    store_policy(root, value)


def test_actual_owned_hook_timeout_never_launches_original_producer(tmp_path, monkeypatch):
    install_actual_hook(tmp_path, "require('fs').writeFileSync('hook-started','actual');setInterval(()=>{},1000);", timeout=1)
    monkeypatch.setattr(lane, "_COMMAND_POLL_SECONDS", 0.05)
    monkeypatch.setattr(lane, "_COMMAND_STOP_GRACE_SECONDS", 0.1)
    result = cycle(tmp_path)
    assert result["phase"] == "deterministic_command_failed"
    assert (tmp_path / "hook-started").exists()
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved["_command_evidence"][recovery.TASK_ID]["exit_code"] == 124
    assert saved["_command_evidence"][recovery.TASK_ID]["producer_invoked"] is False


def test_actual_hook_descendants_are_cleaned_when_owner_cancels(tmp_path, monkeypatch):
    install_actual_hook(tmp_path, """
const fs=require('fs'),cp=require('child_process');
const child=cp.spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{detached:true,stdio:'ignore'});
fs.writeFileSync('actual-child.pid',String(child.pid));
process.on('SIGTERM',()=>{});setInterval(()=>{},1000);
""")
    monkeypatch.setattr(lane, "_COMMAND_POLL_SECONDS", 0.05)
    monkeypatch.setattr(lane, "_COMMAND_STOP_GRACE_SECONDS", 0.1)
    def heartbeat(**_):
        if (tmp_path / "actual-child.pid").exists():
            raise lane._LaneResumeInterrupted
    with pytest.raises(lane._LaneResumeInterrupted):
        cycle(tmp_path, heartbeat)
    child = int((tmp_path / "actual-child.pid").read_text())
    stop = time.monotonic() + 2
    while time.monotonic() < stop:
        try:
            raw = Path(f"/proc/{child}/stat").read_text()
        except FileNotFoundError:
            break
        if raw[raw.rfind(")") + 2:].split()[0] in ("Z", "X"):
            break
        time.sleep(0.02)
    else:
        pytest.fail("owned detached hook descendant survived cancellation")
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert recovery.TASK_ID not in saved.get("_command_evidence", {})


def test_actual_hook_receives_runtime_environment_before_real_producer_and_validator(tmp_path, monkeypatch):
    install_actual_hook(tmp_path, """
const fs=require('fs');const a=process.argv;const profile=a[a.indexOf('--transport')+1];
fs.appendFileSync('hook-order',profile+':'+process.env.VDG_PRIVATE_ENV_PROBE+':'+process.env.VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT+'\\n');
""")
    wrapper = tmp_path / recovery.PRODUCER[1]
    wrapper.write_text("import fs from 'node:fs';if(fs.readFileSync('hook-order','utf8')!=='http:owned:1\\nlibp2p:owned:1\\n')process.exit(3);fs.writeFileSync('report.json','{\"genuine\":true}');")
    (tmp_path / "validate.py").write_text("import json\nfrom pathlib import Path\nassert json.loads(Path('report.json').read_text())['genuine'] is True\n")
    monkeypatch.setenv("VDG_PRIVATE_ENV_PROBE", "owned")
    monkeypatch.delenv("VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT", raising=False)
    result = cycle(tmp_path)
    assert result["phase"] == "task_command_passed"
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved["_command_evidence"][recovery.TASK_ID]["producer_invoked"] is True


def test_new_deadline_created_by_zero_exit_producer_cannot_reach_validation_or_completion(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path)
    interrupted = [False]; generation = [0]; calls = []
    def execute(argv, **_):
        calls.append(list(argv))
        if "--apply" in argv:
            interrupted[0] = False
        elif list(argv) == list(recovery.PRODUCER):
            generation[0] += 1
            interrupted[0] = generation[0] == 1
            (tmp_path / "report.json").write_text(json.dumps({"actual_generation": generation[0]}))
        elif "--require-clean" in argv:
            return 75 if interrupted[0] else 0
        return 0
    monkeypatch.setattr(lane, "_run_argv", execute)
    first = cycle(tmp_path)
    assert first["phase"] == "deterministic_command_failed"
    assert ["python3", "validate.py"] not in calls
    saved = json.loads((tmp_path / "attempts.json").read_text())
    record = saved["_command_evidence"][recovery.TASK_ID]
    assert saved[recovery.TASK_ID] == 1 and record["exit_code"] == 1
    assert record["producer_invoked"] is True and record["failed_stage"] == "post_command_recovery_guard"
    assert "- Status: pending" in (tmp_path / "board.md").read_text()
    second = cycle(tmp_path)
    assert second["phase"] == "task_command_passed" and generation[0] == 2
    assert json.loads((tmp_path / "attempts.json").read_text())[recovery.TASK_ID] == 2


def test_postguard_time_is_deducted_from_same_budget_and_cannot_approve_after_deadline(tmp_path, monkeypatch):
    policy(tmp_path); task(tmp_path); clock = [100.0]; calls = []
    monkeypatch.setattr(lane.time, "monotonic", lambda: clock[0])
    def execute(argv, **options):
        calls.append((list(argv), options["timeout"]))
        if list(argv) == list(recovery.PRODUCER):
            (tmp_path / "report.json").write_text('{"actual":"fresh"}')
            clock[0] += 99
        elif "--require-clean" in argv:
            clock[0] += 2
        return 0
    monkeypatch.setattr(lane, "_run_argv", execute)
    result = cycle(tmp_path)
    assert result["phase"] == "deterministic_command_failed"
    assert calls[-1][1] == 1.0
    assert ["python3", "validate.py"] not in [argv for argv, _ in calls]
    assert json.loads((tmp_path / "attempts.json").read_text())["_command_evidence"][recovery.TASK_ID]["exit_code"] == 124


def test_actual_zero_exit_producer_with_interrupted_acquisition_fails_readonly_postguard(tmp_path):
    install_actual_hook(tmp_path, """
const fs=require('fs');
if(process.argv.includes('--apply')) { if(fs.existsSync('actual-deadline')) fs.unlinkSync('actual-deadline'); }
else if(process.argv.includes('--require-clean') && fs.existsSync('actual-deadline')) process.exit(1);
""")
    (tmp_path / recovery.PRODUCER[1]).write_text("import fs from 'node:fs';fs.writeFileSync('actual-deadline','real private interruption');fs.writeFileSync('report.json','{\"capture_complete\":true,\"decision\":\"NO_GO\"}');")
    assert cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert (tmp_path / "actual-deadline").read_text() == "real private interruption"
    assert json.loads((tmp_path / "report.json").read_text())["decision"] == "NO_GO"
    saved = json.loads((tmp_path / "attempts.json").read_text())["_command_evidence"][recovery.TASK_ID]
    assert saved["failed_stage"] == "post_command_recovery_guard" and saved["producer_invoked"] is True
    assert "- Status: pending" in (tmp_path / "board.md").read_text()
