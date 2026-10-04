"""Execution imports must not overlap the numerical preparation lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("arm,stop", [
    ("full", "initial_context"), ("full", "planning"), ("full", "doctor"),
    ("no-index", "implementation_setup"), ("full", "native_execution"),
])
def test_execution_dependencies_are_loaded_only_after_admission(tmp_path, arm, stop):
    # A fresh interpreter prevents another test's cached imports from hiding an
    # eager dependency. This is a control-flow test, not native qualification.
    program = r'''
import importlib.abc, json, pathlib, sys, types
from unittest.mock import patch
root, arm, stop = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doctor_name = "benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch"
execution = {
    doctor_name,
    "benchmarks.agent_supervisor.container_coding.native_quack_qualification",
    "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
    "ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy",
}
attempts, calls, cleanup, alarms = [], [], [], []
class ExecutionImportRefused(RuntimeError): pass
class EarlyPhaseStopped(RuntimeError): pass
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in execution:
            attempts.append(fullname)
            raise ExecutionImportRefused(fullname)
sys.meta_path.insert(0, Guard())
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
assert not attempts
assert execution.isdisjoint(sys.modules)
state = root / "state" / "run"
state.mkdir(parents=True)
(state / "admission.json").write_text("{}")
def phase(name, result):
    def invoke(**kwargs):
        calls.append(name)
        assert not attempts
        if name == stop:
            raise EarlyPhaseStopped(name)
        return result
    return invoke
def admitted(*args, **kwargs):
    calls.append("admission")
    assert not attempts
    return {"graph": types.SimpleNamespace(tasks=[types.SimpleNamespace(task_cid="task", task_key="key")])}
if stop == "native_execution":
    # Let the driver finish its Doctor/argv steps, but refuse native startup.
    doctor = types.ModuleType(doctor_name)
    doctor.prepare_terminal_doctor_dispatch = phase("doctor", {"route": "doctor_contract_candidate"})
    doctor.implementation_argv = phase("implementation_setup", ["authored-worker"])
    sys.modules[doctor_name] = doctor
with patch.object(driver, "ROOT", root), \
     patch.object(driver.os, "geteuid", return_value=1000), \
     patch.object(driver.os, "umask"), \
     patch.object(driver.signal, "signal"), \
     patch.object(driver.signal, "setitimer", side_effect=lambda *args: alarms.append(args)), \
     patch.object(driver, "_failure_diagnostics", return_value={}), \
     patch.object(driver, "_final_context_audit"), \
     patch.object(driver.subprocess, "run", side_effect=lambda argv, **kw: cleanup.append(argv) or types.SimpleNamespace(returncode=0)), \
     patch.object(driver.preparation, "prepare", side_effect=phase("prepare", {"intent_preplanning": {}})), \
     patch.object(driver.preparation, "initial_context", side_effect=phase("initial_context", {})), \
     patch.object(driver.preparation, "plan", side_effect=phase("planning", {"qualified": True})), \
     patch.object(driver.preparation, "context", side_effect=phase("context", {"context_bundle": {}})), \
     patch.object(driver, "verify_local_benchmark_admission", side_effect=admitted):
    report = driver.run(instruction=root / "instruction", state=state, arm=arm,
                        source384_config=root / "config" if arm == "full" else None)
assert report["error_phase"] == stop
assert report["error"]["type"] == ("EarlyPhaseStopped" if stop in {"initial_context", "planning"} else "ExecutionImportRefused")
assert report["task_completed"] is False and report["provider_invocations"] == []
assert report["max_total_agent_seconds"] == 285 and report["work_cutoff_seconds"] == 245
assert report["reserved_cleanup_seconds"] == 40 and report["worker_cleanup_returncode"] == 0
assert len(cleanup) == 1 and cleanup[0][-1] == "--cleanup"
assert alarms[-1] == (driver.signal.ITIMER_REAL, 0)
if stop in {"initial_context", "planning"}:
    assert not attempts
else:
    assert "admission" in calls and len(attempts) == 1
    assert attempts[0] == ("benchmarks.agent_supervisor.container_coding.native_quack_qualification" if stop == "native_execution" else doctor_name)
if stop == "doctor":
    assert report["phases"]["doctor_seconds"] >= 0
print(json.dumps({"phase": stop, "calls": calls, "attempts": attempts}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", program, str(tmp_path), arm, stop],
                            cwd=Path(__file__).resolve().parents[3], env=dict(os.environ),
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["phase"] == stop
