"""Public validation contracts and bounded isolated-runner diagnostics."""
import copy
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_public_validation as public
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding.test_terminal_public_acceptance_profile import measured_result, public_profile


@pytest.fixture
def observation():
    states = {name: dict(sha256="a"*64, bytes=20, mode=0o660, uid=1000, gid=1000)
              for name in (profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE, "eval.py", "eigen.py")}
    checks = []
    for name, key in zip(profiles.PUBLIC_CHECKS, profiles.PUBLIC_VALIDATION_KEYS):
        result = (measured_result(candidate=3e-6, reference=2e-6) if name == "performance" else
                  dict(schema=profiles.PUBLIC_ACCEPTANCE_RESULT_SCHEMA, profile=profiles.PUBLIC_VALIDATION_PROFILE,
                       check=name, status="passed", reason="obligations_satisfied", measurements=[]))
        checks.append(dict(validation_key=key, returncode=1 if name == "performance" else 0,
                           result=result, status=result["status"], reason=result["reason"]))
    return dict(schema=public.OBSERVATION_SCHEMA, contract_sha256="b"*64, original_deadline_ns=123,
        actual_returncode=0, round_number=1, session_id="c"*32, descriptor_sha256="d"*64,
        workspace_generation="e"*64, source_hashes={"eigen.py": "a"*64}, workspace_states=states,
        status="failed", failed_obligations=[profiles.PUBLIC_VALIDATION_KEYS[-1]], checks=checks, **public.FLAGS)


def test_correct_but_slow_observation_preserves_real_successful_process_exit(observation):
    restored = public.validate_public_validation_observation(observation)
    assert restored == observation and restored["actual_returncode"] == 0
    assert restored["status"] == "failed"
    feedback = public.public_validation_feedback(restored)
    assert len(json.dumps(feedback)) < 4096
    assert len(feedback["checks"][-1]["measurements"]) == 5
    assert feedback["checks"][-1]["measurement_aggregation"] == "median_of_three_batch_medians"
    assert len(restored["checks"][-1]["result"]["measurements"]) == 15
    assert all(row["candidate_median_seconds"] == 3e-6 for row in feedback["checks"][-1]["measurements"])
    assert "workspace_states" not in feedback and "source_hashes" not in feedback
    restored["workspace_states"]["eigen.py"]["mode"] = 0
    assert observation["workspace_states"]["eigen.py"]["mode"] == 0o660


@pytest.mark.parametrize("change", [
    lambda r: r.update(continuation_authority=True),
    lambda r: r.update(raw_stdout="private text"),
    lambda r: r.update(actual_returncode=True),
    lambda r: r.update(status="passed"),
    lambda r: r.update(failed_obligations=[]),
    lambda r: r.update(session_id="invalid"),
    lambda r: r.update(source_hashes={"outside.py": "a"*64}),
    lambda r: r["checks"][2].update(returncode=0),
    lambda r: r["checks"][2]["result"]["measurements"][0].update(candidate_median_seconds=float("inf")),
    lambda r: r["workspace_states"]["eigen.py"].update(mode=True),
    lambda r: r["checks"].pop(),
])
def test_closed_observation_refuses_forged_or_unjoined_claims(observation, change):
    change(observation)
    with pytest.raises(ValueError):
        public.validate_public_validation_observation(observation)


def test_unavailable_check_never_becomes_pass_or_invented_measurement(observation):
    observation["checks"][2].update(returncode=None, result=None, status="inconclusive", reason="timeout")
    observation.update(status="inconclusive", failed_obligations=[])
    value = public.validate_public_validation_observation(observation)
    assert public.public_validation_feedback(value)["checks"][-1]["measurements"] == []
    with pytest.raises(ValueError):
        public.RetiredCandidateValidation(object(), object(), object(), public.canonical_json(value))


@pytest.mark.parametrize("zero", [False, 0.0])
def test_signed_spec_requires_exact_types_even_for_numerically_equal_exit_codes(zero):
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LOCAL_POLICY
    profile = public_profile()
    spec = profiles.task_profile_spec(profile, policy_cid=content_identity(LOCAL_POLICY))
    assert public._spec(profile, spec) == spec
    spec["validations"][0]["expected_exit_codes"] = [zero]
    with pytest.raises(ValueError, match="commands or acceptance differ"):
        public._spec(profile, spec)


@pytest.fixture
def local_runner(monkeypatch):
    # This exercises the bounded pipe/timeout implementation with real children;
    # the installed root-owned privilege boundary is explicitly a test double.
    monkeypatch.setattr(execution, "verify_candidate_runner", lambda binding: binding)
    return dict(argv=[sys.executable, "-I", "-c"])


def test_bounded_capture_keeps_exact_result_bytes(tmp_path, local_runner):
    result = execution.run_candidate_bounded(local_runner, ["print('result')"], cwd=tmp_path, timeout=3)
    assert result.returncode == 0 and result.stdout == b"result\n"


def test_excess_output_is_stopped_before_unbounded_capture(tmp_path, local_runner):
    with pytest.raises(execution.CandidateOutputLimitError):
        execution.run_candidate_bounded(local_runner,
            ["import os\nwhile True: os.write(1,b'x'*8192)"], cwd=tmp_path, timeout=3, max_output_bytes=100)


def test_bounded_capture_reaps_on_original_check_timeout(tmp_path, local_runner):
    marker = tmp_path / "finished"
    source = "import time;from pathlib import Path;time.sleep(2);Path("+repr(str(marker))+").touch()"
    with pytest.raises(subprocess.TimeoutExpired):
        execution.run_candidate_bounded(local_runner, [source], cwd=tmp_path, timeout=0.05)
    assert not marker.exists()


def test_runner_setup_cannot_renew_check_deadline(tmp_path, local_runner, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def verify(binding):
        clock[0] = 2.0
        return binding
    monkeypatch.setattr(execution, "verify_candidate_runner", verify)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("expired setup launched a process"))
    with pytest.raises(subprocess.TimeoutExpired):
        execution.run_candidate_bounded(local_runner, ["pass"], cwd=tmp_path, timeout=1)


def test_lock_wait_consumes_the_same_check_deadline(tmp_path, local_runner, monkeypatch):
    held, release = threading.Event(), threading.Event()
    lock = threading.RLock()
    monkeypatch.setattr(execution, "_EXECUTION_LOCK", lock)
    def hold():
        with lock:
            held.set()
            release.wait(timeout=5)
    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert held.wait(timeout=2)
        monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("expired lock wait launched a process"))
        with pytest.raises(subprocess.TimeoutExpired):
            execution.run_candidate_bounded(local_runner, ["pass"], cwd=tmp_path, timeout=0.02)
    finally:
        release.set(); thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize("limit", [True, 0, -1, 65537])
def test_bad_capture_bounds_refuse_before_process(tmp_path, local_runner, limit, monkeypatch):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid limit started child"))
    with pytest.raises(ValueError):
        execution.run_candidate_bounded(local_runner, ["pass"], cwd=tmp_path, timeout=1, max_output_bytes=limit)


def test_native_signed_contract_binds_full_checks_and_refuses_immutable_source_drift(tmp_path, monkeypatch):
    from test.api.public_validation_fixture import public_doctor_service
    with public_doctor_service(tmp_path, monkeypatch) as case:
        binding = dict(schema=execution.SCHEMA, argv=[str(execution.ROOT / "bin/validation-worker")],
            files={str(execution.ROOT / path): "b"*64 for path in (
                "bin/validation-worker", "bin/worker-entry", "container-boundary.json")},
            owner_uid=os.getuid(), worker_uid=1001,
            namespaces={name: os.readlink("/proc/self/ns/"+name) for name in ("pid", "mnt", "net")},
            completion_authority=False)
        monkeypatch.setattr(execution, "candidate_runner_from_environment", lambda: copy.deepcopy(binding))
        monkeypatch.setattr(execution, "verify_candidate_runner", lambda selected: selected == binding or
                            (_ for _ in ()).throw(ValueError("changed runner")))
        contract = public.capture_public_validation_contract(source_broker=case.original,
            requirements=case.original.original_requirements,
            deadline_monotonic_ns=time.monotonic_ns()+300_000_000_000)
        assert contract.revalidate()["task_cid"] == case.task_cid
        assert len(contract.metadata["task_spec"]["validations"]) == 3
        assert len(contract.metadata["task_spec"]["acceptance"]) == 3
        assert public.validate_public_validation_contract_metadata(contract.metadata) == contract.metadata
        for target in (case.workspace / "eval.py", case.workspace / profiles.SMOKE,
                       case.repository / profiles.PROFILE, case.public_request["artifact"]):
            raw, mode = target.read_bytes(), target.stat().st_mode & 0o7777
            try:
                target.chmod(0o600); target.write_bytes(raw+b"\n# changed\n"); target.chmod(mode)
                with pytest.raises(ValueError):
                    contract.revalidate()
            finally:
                target.chmod(0o600); target.write_bytes(raw); target.chmod(mode)
        mode = (case.workspace / "eval.py").stat().st_mode & 0o7777
        (case.workspace / "eval.py").chmod(mode ^ 0o100)
        with pytest.raises(ValueError, match="immutable"):
            contract.revalidate()
        (case.workspace / "eval.py").chmod(mode)
        assert contract.revalidate() == contract.metadata
        with monkeypatch.context() as patch:
            original_replay = public._replay_contract
            now = [time.monotonic_ns()]
            def slow_replay(*args, **kwargs):
                result = original_replay(*args, **kwargs)
                now[0] = contract.metadata["deadline_monotonic_ns"]+1
                return result
            patch.setattr(public, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
            patch.setattr(public, "_replay_contract", slow_replay)
            with pytest.raises(ValueError, match="deadline exhausted during replay"):
                contract.revalidate()
