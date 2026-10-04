"""Owner validation shares the approved Python used by native pre-merge checks."""
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import _prepare, SOURCE
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


def test_bare_python_validation_does_not_depend_on_os_default_or_ambient_path(scenario, tmp_path, monkeypatch):
    inputs = _prepare(scenario, tmp_path, text=SOURCE.replace("count=2", "amount=2"))
    intent, cid = scenario["intent"], inputs["task_cid"]
    task = intent.get_task(cid)
    intent.cas_task_status(task_cid=cid, expected_revision=task["revision"], new_status="in_progress")
    malicious = tmp_path / "untrusted-bin"
    malicious.mkdir()
    marker = tmp_path / "wrong-interpreter"
    (malicious / "python3").write_text("#!/bin/sh\ntouch '" + str(marker) + "'\nexit 1\n")
    (malicious / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", str(malicious))
    monkeypatch.setattr(local.os, "defpath", "/nonexistent-no-python")
    result = local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="launcher-check")
    assert result["passed"] is True
    assert not marker.exists()
    assert intent.get_task(cid)["status"] == "in_progress"
