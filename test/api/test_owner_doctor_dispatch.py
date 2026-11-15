"""Explicit Doctor coding allocation, native failed child and fresh continuation.

Actual residuals, source snapshots, owner custody, sealed input and local child
processes are exercised. This single-user fixture authors container identity
and cleanup boundaries; no provider call or token-cap claim is made.
"""
import json
import hashlib
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_lifecycle as lifecycle
from test.api.test_doctor_candidate_context import doctor_service, retained, scenario, capture  # noqa: F401
from test.api.test_owner_candidate_round_context import waited
from test.api.test_owner_candidate_dispatch import original_argv
from test.api.test_candidate_round_lifecycle import row
from test.api.test_terminal_candidate_attempt_accounting import observation


def arguments(service, request):
    return original_argv(service)+["--doctor-residual-artifact", str(request["artifact"]),
        "--doctor-residual-sha256", request["expected_sha256"], "--doctor-residual-task-cid", request["task_cid"]]


@pytest.fixture
def doctor_launch(doctor_service, monkeypatch):
    service, request = doctor_service
    service.original_requirements = capture(service, request)
    def cleanup():
        assert all(signal.getsignal(number) is signal.SIG_IGN for number in
            (signal.SIGTERM, signal.SIGINT, signal.SIGHUP))
        return subprocess.run([sys.executable, "-I", "-c", "pass"], timeout=5)
    monkeypatch.setattr(owner, "_isolated_cleanup", cleanup)
    launch = owner.capture_original_coding_launch(source_broker=service, argv=arguments(service, request),
        absolute_deadline=time.monotonic()+89, doctor_coding_policy=policy.DOCTOR_CODING_POLICY)
    return launch


def retire_first(launch, *, code=3):
    service = launch.source_broker
    command = [*owner._PREFIX, *launch._argv, *service.extra_argv]
    launch.reserve_first_attempt(command)
    waited(service, code=code, close=False)
    if code:
        path = service.workspace / "answer.py"
        path.write_text(path.read_text().replace("process(count=2)", "process(amount=2)"))
    return owner.retire_coding_worker(launch=launch, process=service._process, source_broker=service)


def test_first_attempt_cannot_start_after_final_doctor_validation_crosses_deadline(doctor_launch, monkeypatch):
    launch = doctor_launch
    doctor = launch._original_requirements._doctor_requirements
    native = type(doctor).revalidate_initial
    def cross_deadline(handle):
        result = native(handle)
        monkeypatch.setattr(owner, 'time', SimpleNamespace(monotonic=lambda:launch.absolute_deadline+1))
        return result
    monkeypatch.setattr(type(doctor), 'revalidate_initial', cross_deadline)
    command = [*owner._PREFIX,*launch._argv,*launch.source_broker.extra_argv]
    with pytest.raises(ValueError, match='deadline'):
        launch.reserve_first_attempt(command)
    assert launch._state.reserved_attempts == set() and launch.source_broker._process is None


@pytest.mark.parametrize("selected", [None, "wrong", True])
def test_doctor_continuation_requires_explicit_independent_policy(doctor_service, selected):
    service, request = doctor_service
    service.original_requirements = capture(service, request)
    with pytest.raises(ValueError):
        owner.capture_original_coding_launch(source_broker=service, argv=arguments(service, request),
            absolute_deadline=time.monotonic()+89, doctor_coding_policy=selected)
    assert service._process is None and not service._registered.is_set()


def test_residual_nomination_must_match_captured_policy_even_when_flags_are_omitted(doctor_service):
    service, request = doctor_service
    service.original_requirements = capture(service, request)
    for argv in (original_argv(service), arguments(service, {**request, "expected_sha256": "a"*64})):
        with pytest.raises(ValueError, match="omitted or substituted"):
            owner.capture_original_coding_launch(source_broker=service, argv=argv,
                absolute_deadline=time.monotonic()+89, doctor_coding_policy=policy.DOCTOR_CODING_POLICY)


@pytest.mark.parametrize("already_captured", [False, True])
def test_stale_native_only_handle_cannot_drop_later_captured_doctor_constraints(doctor_service, already_captured):
    service, request = doctor_service
    stale = prompts.capture_native_only_requirements(source_broker=service)
    service.original_requirements = stale
    launch = None
    if already_captured:
        launch = owner.capture_original_coding_launch(source_broker=service, argv=original_argv(service),
            absolute_deadline=time.monotonic()+89)
    doctor = capture(service, request)
    assert doctor._doctor_requirements is not None and service._doctor_request is not None
    service.original_requirements = stale
    with pytest.raises(ValueError, match="Doctor"):
        if launch is None:
            owner.capture_original_coding_launch(source_broker=service, argv=original_argv(service),
                absolute_deadline=time.monotonic()+89)
        else:
            launch.revalidate()
    assert service._process is None and not service._registered.is_set()


def test_policy_pins_original_proposal_budget_without_native_token_or_cost_claim(doctor_launch):
    launch = doctor_launch
    declared = launch.doctor_coding_policy
    assert declared["proposal_budget"] == dict(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1)
    assert declared["max_child_attempts"] == declared["max_outer_router_calls"] == 2
    assert declared["deadline_monotonic_ns"] == int(launch.absolute_deadline*1_000_000_000)
    assert declared["provider_internal_calls_bounded"] is declared["total_tokens_bounded"] is declared["total_cost_bounded"] is False
    declared["proposal_budget"]["max_calls"] = 99
    assert launch.doctor_coding_policy["proposal_budget"]["max_calls"] == 1
    command = [*owner._PREFIX, *launch._argv, *launch.source_broker.extra_argv]
    launch.reserve_first_attempt(command)
    with pytest.raises(ValueError, match="already reserved"):
        launch.reserve_first_attempt(command)
    assert launch._state.reserved_attempts == {1} and launch.source_broker._process is None


def test_first_success_retains_policy_without_spending_or_inventing_second_attempt(doctor_launch):
    launch = doctor_launch
    retired = retire_first(launch, code=0)
    with pytest.raises(ValueError, match="positive first coding failure"):
        owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    saved = lifecycle.persist(source_broker=launch.source_broker,
        observation=observation([row(launch.source_broker, 1)]))
    value = lifecycle.parse_retention((launch.source_broker.descriptor_path.parent/saved["path"]).read_bytes())
    assert value["schema"] == lifecycle.DOCTOR_SCHEMA
    assert value["doctor_coding_policy"] == launch.doctor_coding_policy
    assert value["doctor_attempts_reserved"] == [1]
    assert len(value["observation"]["rounds"]) == 1


def test_native_preparation_preserves_doctor_constraints_and_refuses_a_third_attempt(doctor_launch, monkeypatch, record_property):
    launch = doctor_launch
    retired = retire_first(launch)
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    try:
        wire = json.loads(prepared.prompt.provider_prompt)
        assert wire["schema"] == prompts.DOCTOR_PROMPT_SCHEMA
        context = prepared.prompt._doctor_context
        assert wire["doctor_context"] == dict(schema="supervisor-doctor-candidate-context-binding@1",
            sha256=hashlib.sha256(context.metadata_bytes).hexdigest(), bytes=len(context.metadata_bytes))
        compact_bytes = len(owner.canonical_json(wire["doctor_context"]))
        full_bytes = len(context.metadata_bytes)
        assert compact_bytes < full_bytes // 4
        for name, value in dict(doctor_context_full_bytes=full_bytes,
                doctor_context_binding_bytes=compact_bytes, owner_metadata_bytes_omitted=full_bytes-compact_bytes).items():
            record_property(name, value)
        assert wire["doctor_advisory"].encode() == launch._original_requirements._doctor_requirements.advisory_bytes
        assert context.metadata["candidate_binding"]["workspace_generation"] == prepared.snapshot.metadata["workspace_generation"]
        admitted = prepared.admission.revalidate(snapshot=prepared.snapshot, prompt=prepared.prompt,
            source_broker=prepared.source_broker)
        assert admitted["schema"] == "supervisor-owner-candidate-dispatch-admission@2"
        assert admitted["doctor_coding_policy"] == launch.doctor_coding_policy
        commands = []
        def spawn(command, *, stdin, workspace):
            commands.append(command)
            assert launch._state.reserved_attempts == {1, 2}
            assert "--doctor-residual-artifact" not in command and "--doctor-coding-policy" not in command
            return subprocess.Popen([sys.executable, "-I", "-c", "import sys;sys.stdin.read();sys.exit(4)"],
                stdin=stdin, cwd=workspace)
        monkeypatch.setattr(owner, "_spawn_worker", spawn)
        running = prepared.start()
        assert running.process.wait(timeout=launch.remaining_seconds) == 4
        owner.retire_coding_worker(launch=launch, process=running.process, source_broker=running.source_broker)
        with pytest.raises(ValueError, match="already consumed"):
            prepared.start()
        assert len(commands) == 1
        prepared.close()
        saved = lifecycle.persist(source_broker=launch.source_broker,
            observation=observation([row(launch.source_broker, 1), row(prepared.source_broker, 2)]))
        value = lifecycle.parse_retention((launch.source_broker.descriptor_path.parent/saved["path"]).read_bytes())
        assert value["doctor_attempts_reserved"] == [1, 2]
        assert value["doctor_coding_policy"] == launch.doctor_coding_policy
    finally:
        prepared.close()


def test_candidate_spawn_interruption_consumes_allocation_without_retry(doctor_launch, monkeypatch):
    launch = doctor_launch
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retire_first(launch))
    def interrupted(*args, **kwargs):
        assert launch._state.reserved_attempts == {1, 2}
        raise InterruptedError("authored interruption at potential Popen")
    monkeypatch.setattr(owner, "_spawn_worker", interrupted)
    try:
        with pytest.raises(InterruptedError):
            prepared.start()
        with pytest.raises(ValueError, match="already consumed"):
            prepared.start()
        assert launch._state.reserved_attempts == {1, 2}
    finally:
        prepared.close()


@pytest.mark.parametrize("argv", [
    ["--doctor-coding-policy"], ["--doctor-coding-policy=wrong", "--candidate-rounds=2"],
    ["--doctor-coding-policy="+policy.DOCTOR_CODING_POLICY],
    ["--doctor-coding-policy="+policy.DOCTOR_CODING_POLICY, "--candidate-rounds=1"],
    ["--candidate-rounds=2", "--doctor-coding-policy="+policy.DOCTOR_CODING_POLICY,
     "--doctor-coding-policy", policy.DOCTOR_CODING_POLICY]])
def test_owner_control_refuses_ambiguous_or_unpaired_policy(argv):
    with pytest.raises(ValueError):
        policy.split_owner_coding_controls(argv)


def test_owner_policy_never_reaches_worker_argv():
    argv = ["--model", "authored", "--candidate-rounds=2", "--doctor-coding-policy", policy.DOCTOR_CODING_POLICY]
    assert policy.split_owner_coding_controls(argv) == (2, policy.DOCTOR_CODING_POLICY, ["--model", "authored"])
    assert policy.split_owner_coding_controls(["--model", "authored"]) == (1, None, ["--model", "authored"])


@pytest.mark.parametrize("budget", [None, [], {1: 2}, {"max_calls": True},
    dict(max_calls=2, max_tokens=4096, max_rounds=1, max_cost_units=1)])
def test_historical_policy_refuses_malformed_or_reallocated_proposal_budget(doctor_launch, budget):
    declared = doctor_launch.doctor_coding_policy
    declared["proposal_budget"] = budget
    with pytest.raises(ValueError):
        policy.validate_doctor_coding_policy_shape(declared)
