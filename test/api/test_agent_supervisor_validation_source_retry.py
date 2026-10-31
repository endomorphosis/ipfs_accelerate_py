"""Prospective metadata boundaries only; all genuine execution is separate.

These real private files model immutable history and an existing finite epoch.
They do not claim actual source-lease admission, a source timeout, native or
storage custody, Popen/wait, provider execution, Git, or service qualification.
The original integration still must exercise those actual owners independently.
"""
from __future__ import annotations
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime


PROFILE = {"schema": "agent-supervisor/validation-admission-retry@1",
    "max_additional_invocations": 2, "cooldown_seconds": 60, "source_probe_timeout_seconds": 60}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else owners.canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o400)
    return runtime._pin(path)


@pytest.mark.parametrize("field,bad", [("max_additional_invocations", 3), ("max_additional_invocations", True),
    ("cooldown_seconds", 0), ("source_probe_timeout_seconds", 300), ("schema", "unknown")])
def test_source_retry_cannot_widen_or_reset_its_explicit_fixed_profile(field, bad):
    profile = {**PROFILE, field: bad}
    with pytest.raises(ValueError, match="exact finite"):
        owners.checked_validation_admission_retry(profile)


def test_source_retry_profile_is_copied_without_creating_budget_state():
    checked = owners.checked_validation_admission_retry(PROFILE)
    assert checked == PROFILE and checked is not PROFILE
    assert "attempts" not in checked and "epoch" not in checked


def test_source_absence_accepts_missing_ancestor_and_observes_existing_regular_leaf(tmp_path):
    assert owners.validation_path_absent(tmp_path / "missing/ancestor/leaf") is True
    leaf = tmp_path / "existing.txt"; leaf.write_bytes(b"preserved")
    assert owners.validation_path_absent(leaf) is False
    assert leaf.read_bytes() == b"preserved"


@pytest.mark.parametrize("kind", ["permission", "io"])
def test_source_absence_never_converts_non_ENOENT_to_no_admission(tmp_path, monkeypatch, kind):
    actual = owners.os.stat
    def failing(name, *args, **kwargs):
        if name == "missing-leaf":
            if kind == "permission":
                raise PermissionError(errno.EACCES, "private test denial", str(tmp_path / name))
            raise OSError(errno.EIO, "private test IO failure", str(tmp_path / name))
        return actual(name, *args, **kwargs)
    monkeypatch.setattr(owners.os, "stat", failing)
    with pytest.raises(OSError):
        owners.validation_path_absent(tmp_path / "missing-leaf")


def test_source_absence_refuses_aliased_ancestor(tmp_path):
    original = tmp_path / "original"; original.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        owners.validation_path_absent(alias / "missing")


@pytest.fixture
def unstarted_scope(tmp_path):
    invocation = tmp_path / "metadata/original-validate-invocation"; invocation.mkdir(parents=True)
    policy = {"validation": {"attempt_root": str(tmp_path / "attempts"), "external_fixture_profile": {}}}
    frozen = {"binding": {"namespace": "ptr_lane_0", "board_id": "legal-ir", "attempt_id": "a" * 32}}
    return policy, frozen, invocation


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_any_existing_D_plan_request_or_attempt_prevents_pre_admission_eligibility(unstarted_scope, index):
    policy, frozen, invocation = unstarted_scope
    paths = owners.validation_unstarted_paths(policy, frozen, invocation)
    before = owners.require_validation_unstarted(policy, frozen, invocation)
    assert before == [str(path) for path in paths]
    path = paths[index]
    if index >= 3:
        path.mkdir(parents=True)
    else:
        put(path, {"private_existing_original_artifact": True})
    with pytest.raises(ValueError, match="already has"):
        owners.require_validation_unstarted(policy, frozen, invocation)
    assert path.exists()


@pytest.fixture
def retry_model(tmp_path, monkeypatch):
    """History selection model, never a genuine current owner/cgroup/epoch."""
    token = "a" * 32; metadata = tmp_path / "metadata"; metadata.mkdir()
    owner_source = put(tmp_path / "owner.py", b"# unexecuted private command-source fixture\n")
    engine_source = put(tmp_path / "engine.py", b"# unexecuted private engine-source fixture\n")
    command = SimpleNamespace(argv=("/private/python", "/private/owner", "validate", "{candidate_root}"),
        verify_argv=("/private/python", "/private/owner", "verify-validation", "{candidate_root}"),
        source_descriptors=(owner_source,), timeout_seconds=420)
    obj = object.__new__(runtime.CandidateRuntime)
    obj.config = {"repair_worker": {"candidate_runtime": {"private_model_only": True}}}
    obj.board = {"id": "legal-ir"}; obj.token = token; obj.attempt_number = 3
    obj.job = {"attempts": 3, "repair_unit_token": token, "private_modeled_authority": True}
    obj.policy = SimpleNamespace(repository_root=tmp_path)
    obj.runtime_policy = SimpleNamespace(validation_admission_retry=dict(PROFILE), validation_owner=command,
        sources=(engine_source,))
    binding = {"namespace": "ptr_lane_0", "board_id": "legal-ir", "attempt_id": token}
    obj.session = SimpleNamespace(metadata_root=metadata, candidate_root=tmp_path / "candidate",
        binding=binding, freeze={"binding": binding, "freeze_sha256": "b" * 64, "source_commitment": "c" * 64})
    obj._job = lambda **kwargs: obj.job
    monkeypatch.setattr(runtime, "require_current_authorization", lambda *args, **kwargs: SimpleNamespace(commitment="d" * 64))
    calls = []
    observations = {}
    def original_observation(selected, invocation, terminal, **kwargs):
        calls.append((invocation.name, kwargs))
        return observations.get(invocation.name)
    obj._validation_admission_observation = original_observation
    return SimpleNamespace(runtime=obj, metadata=metadata, calls=calls, observations=observations,
        initial_job=dict(obj.job), command=command)


def record_model_slot(scope, invocation, *, code=75, eligible_after=1, started_at=2):
    obj = scope.runtime
    invocation.mkdir()
    put(invocation / "intent.json", {"validation_admission_retry": obj.validation_admission_selected_context,
        "validation_admission_invocation_started_at": started_at})
    put(invocation / "terminal.json", {"command_returncode": code, "verifier_returncode": None if code else 0})
    if code == 75:
        observed = {"eligible_after": eligible_after, "private_metadata_model": True}
        ref = put(invocation / "source-admission-observation.json", observed)
        scope.observations[invocation.name] = (ref, observed)


def test_first_validation_slot_keeps_original_name_and_spends_no_provider_budget(retry_model):
    s = retry_model
    selected = s.runtime._validation_invocation()
    assert selected == s.metadata / "original-validate-invocation"
    assert not selected.exists() and s.calls == []
    assert s.runtime.validation_admission_selected_context["index"] == 0
    assert s.runtime.validation_admission_selected_context["previous"] is None
    assert s.runtime.job == s.initial_job


def test_deferred_source_cooldown_returns_without_creating_next_invocation(retry_model):
    s = retry_model
    first = s.runtime._validation_invocation()
    record_model_slot(s, first, eligible_after=10**30)
    assert s.runtime._validation_invocation() is None
    assert s.runtime.validation_admission_wait["status"] == "backoff"
    assert not (s.metadata / "original-validate-source-admission-retry-01").exists()
    assert s.runtime.job == s.initial_job


def test_two_fresh_retry_slots_persist_without_reset_and_then_exhaust(retry_model):
    s = retry_model
    names = ["original-validate-invocation", "original-validate-source-admission-retry-01", "original-validate-source-admission-retry-02"]
    for index, name in enumerate(names):
        selected = s.runtime._validation_invocation()
        assert selected == s.metadata / name
        assert s.runtime.validation_admission_selected_context["index"] == index
        record_model_slot(s, selected)
    assert s.runtime._validation_invocation() is None
    assert s.runtime.validation_admission_wait["status"] == "finite_retry_exhausted"
    assert s.runtime.validation_admission_wait["additional_slots_spent"] == 2
    assert {path.name for path in s.metadata.iterdir()} == set(names)
    assert s.runtime.job == s.initial_job and s.runtime.job["attempts"] == 3


@pytest.mark.parametrize("missing", ["intent", "terminal"])
def test_unknown_existing_slot_is_held_without_observation_or_replay(retry_model, missing):
    s = retry_model; selected = s.runtime._validation_invocation(); selected.mkdir()
    if missing == "terminal":
        put(selected / "intent.json", {"validation_admission_retry": s.runtime.validation_admission_selected_context})
    else:
        put(selected / "terminal.json", {"command_returncode": 75, "verifier_returncode": None})
    assert s.runtime._validation_invocation() is None
    assert s.calls == [] and s.runtime.job == s.initial_job
    assert not (s.metadata / "original-validate-source-admission-retry-01").exists()


def test_retry_history_gap_never_authorizes_a_new_slot(retry_model):
    s = retry_model
    (s.metadata / "original-validate-source-admission-retry-02").mkdir()
    with pytest.raises(ValueError, match="gap or fork"):
        s.runtime._validation_invocation()
    assert s.calls == [] and s.runtime.job == s.initial_job


@pytest.mark.parametrize("field", ["epoch_sha256", "freeze_sha256", "attempt_token"])
def test_retry_intent_changed_epoch_freeze_or_token_is_refused(retry_model, field):
    s = retry_model; selected = s.runtime._validation_invocation()
    s.runtime.validation_admission_selected_context[field] = "e" * 64
    record_model_slot(s, selected)
    with pytest.raises(ValueError, match="chain/epoch/candidate"):
        s.runtime._validation_invocation()
    assert s.calls == [] and s.runtime.job == s.initial_job


def test_later_success_uses_historical_predecessor_without_requiring_D_artifacts_absent(retry_model):
    s = retry_model
    first = s.runtime._validation_invocation(); record_model_slot(s, first)
    second = s.runtime._validation_invocation(); record_model_slot(s, second, code=0)
    s.calls.clear()
    assert s.runtime._validation_invocation(create_observation=False) == second
    assert s.calls == [(first.name, {"create": False, "require_absent": False})]
    assert s.runtime.job == s.initial_job


def test_existing_retry_slot_cannot_bypass_the_predecessor_cooldown(retry_model):
    s = retry_model
    first = s.runtime._validation_invocation(); record_model_slot(s, first, eligible_after=10**30)
    observation, _ = s.observations[first.name]
    previous = {"invocation": str(first), "intent": runtime._pin(first / "intent.json"),
        "terminal": runtime._pin(first / "terminal.json"), "observation": observation, "eligible_after": 10**30}
    s.runtime.validation_admission_selected_context = s.runtime._validation_admission_context(1, previous)
    second = s.metadata / "original-validate-source-admission-retry-01"
    record_model_slot(s, second, code=0, started_at=2)
    with pytest.raises(ValueError, match="before its immutable cooldown"):
        s.runtime._validation_invocation(create_observation=False)
    assert s.runtime.job == s.initial_job


@pytest.mark.parametrize("code", [0, 42])
def test_success_or_other_failure_cannot_hide_an_orphan_retry_successor(retry_model, code):
    s = retry_model
    first = s.runtime._validation_invocation(); record_model_slot(s, first, code=code)
    (s.metadata / "original-validate-source-admission-retry-02").mkdir()
    with pytest.raises(ValueError, match="orphan retry successor"):
        s.runtime._validation_invocation(create_observation=False)
    assert s.calls == [] and s.runtime.job == s.initial_job


def test_validation_retry_refuses_replaced_process_runner_before_any_launch(retry_model):
    s = retry_model
    s.runtime.command_runner = lambda *args, **kwargs: pytest.fail("replacement runner was invoked")
    with pytest.raises(ValueError, match="original actual process runner"):
        s.runtime._validation_command_wait(["/private/never-launched"], s.metadata, 420)


def test_default_runtime_policy_retains_no_implicit_validation_retry():
    value = runtime.RuntimePolicy((), None, None, {}, 10)
    assert value.validation_admission_retry is None


def test_foreign_timeout_source_is_refused_before_it_can_produce_a_defer(unstarted_scope, tmp_path, monkeypatch):
    policy, frozen, invocation = unstarted_scope
    policy.update(datasets_root=str(tmp_path / "D"), source_descriptors=[])
    policy["validation"]["validation_admission_retry"] = dict(PROFILE)
    source_path = Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"
    policy["source_descriptors"].append(put(source_path, b"# private origin-refusal metadata, never executed\n"))
    put(invocation / "intent.json", {"private_unexecuted_intent": True})
    def foreign_source(*args, **kwargs):
        pytest.fail("foreign timeout source was invoked")
    monkeypatch.setattr(owners, "public_source_lease", lambda policy: foreign_source)
    args = SimpleNamespace(invocation_root=invocation, candidate_root=tmp_path / "candidate")
    with pytest.raises(ValueError, match="primitive origin differs"):
        owners.probe_validation_source_admission(policy, args, frozen)
    assert not (invocation / "source-admission-deferred.json").exists()


@pytest.mark.parametrize("field,bad", [("actual_Popen_returned", False),
    ("actual_process_waited", False), ("actual_process_returncode", True)])
def test_defer_observation_requires_real_wait_before_any_readonly_observer(retry_model, field, bad):
    s = retry_model; obj = s.runtime
    selected = obj._validation_invocation(); selected.mkdir()
    values = {"{candidate_root}": str(obj.session.candidate_root)}
    intent = {"schema": runtime.SCHEMA, "stage": "validate",
        "argv": [values.get(value, value) for value in s.command.argv],
        "verify_argv": [values.get(value, value) for value in s.command.verify_argv],
        "owner_source_descriptors": list(s.command.source_descriptors),
        "runtime_policy_sha256": runtime.candidate.digest(obj.config["repair_worker"]["candidate_runtime"]),
        "attempt_token": obj.token, "attempt_number": obj.attempt_number,
        "freeze_sha256": obj.session.freeze["freeze_sha256"], "source_commitment": obj.session.freeze["source_commitment"],
        "validation_admission_retry": obj.validation_admission_selected_context}
    intent_ref = put(selected / "intent.json", intent)
    wait = {"schema": "agent-supervisor/validation-owner-wait@1", "argv": intent["argv"],
        "owning_command_timeout_seconds": s.command.timeout_seconds,
        "provider_started": False, "coding_attempts_incremented": False,
        "command_started_monotonic": 1, "command_finished_monotonic": 2,
        "intent": intent_ref, "actual_Popen_returned": True, "actual_process_waited": True,
        "actual_process_returncode": 75, "returncode": 75}
    wait[field] = bad
    wait_ref = put(selected / "original-owner-wait.json", wait)
    terminal = {"command_returncode": 75, "verifier_returncode": None, "original_owner_wait": wait_ref}
    put(selected / "terminal.json", terminal)
    obj.command_runner = lambda *args, **kwargs: pytest.fail("unproven original wait reached an observer")
    with pytest.raises(ValueError, match="genuine original wait"):
        runtime.CandidateRuntime._validation_admission_observation(obj, s.command, selected, terminal, create=True)
    assert not (selected / "source-admission-observation.json").exists()
