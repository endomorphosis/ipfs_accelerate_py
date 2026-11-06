"""Prospective protocol regressions, not real resource/CLI closure evidence.

The fixtures below model immutable original-owner observations. Actual public
D allocation, failed-claim disposition and process waits require their own
original-owner integration qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as auth
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime


def frozen_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runtime._wire(value))
    path.chmod(0o400)
    return runtime._pin(path)


def negative_fixture(tmp_path, attempt_number=1):
    token = "a" * 32
    binding = {"board_id": "legal-ir", "attempt_id": token, "attempt_number": attempt_number}
    invocation = tmp_path / "candidate" / "original-validate-invocation"
    refs = {key: frozen_json(invocation / (key + ".json"), {"fixture": key}) for key in
            ("request", "original_failure_terminal", "original_runner_wait", "trusted_semantic_report")}
    negative = {"schema": runtime.NEGATIVE_VALIDATION_SCHEMA, "passed": False, "semantic_negative": True,
        "negative_profile": "trusted_pytest_call_failure",
        "freeze_sha256": "b" * 64, "source_commitment": "c" * 64, "binding": binding,
        "original_owner_returncode": 1, "original_owner_identity": {"pid": 12345, "birth": "123456"},
        "reservation_id": "d" * 32, "native_lease_id": "e" * 32,
        "source_closed": True, "native_closed": True, "processes_closed": True,
        "storage_closed": False, "cleanup_errors": [], **refs}
    pin = frozen_json(invocation / "negative-validation.json", negative)
    joins = {key: copy.deepcopy(negative[key]) for key in ("negative_profile", "freeze_sha256", "source_commitment", "binding",
        "request", "original_owner_returncode", "original_owner_identity", "reservation_id", "native_lease_id")}
    closed = {"schema": "agent-supervisor/negative-validation-closure-observation@1", "negative_validation": pin,
        **joins, "source_closed": True, "native_closed": True, "processes_closed": True,
        "storage_closed": True, "cleanup_errors": [], "storage_disposition": {"fixture": "original-public-disposition"},
        "observed_at": time.time()}
    observed = {"schema": "agent-supervisor/verified-negative-validation@1", "negative_validation": pin,
                "closure_observation": closed}
    return negative, observed, invocation


def check(observed, negative):
    return runtime._checked_negative_observation(observed, binding=negative["binding"],
        freeze_sha256=negative["freeze_sha256"], source_commitment=negative["source_commitment"],
        original_returncode=negative["original_owner_returncode"])


def epoch_runtime(tmp_path, number=1):
    negative, observed, invocation = negative_fixture(tmp_path, number)
    root = tmp_path / "state"
    old = {"board_id": "legal-ir", "status": "verified_healthy", "escalation_reason": "repair_budget_exhausted",
           "attempts": 3, "latest_incident": {"signature": "old"}}
    prior = frozen_json(root / "repairs/legal-ir/job.json", old)
    policy = {"namespace": "legal-ir", "state_root": str(root), "base_commit": "0" * 40}
    now = time.time()
    epoch = {"schema": auth.SCHEMA, "epoch_id": "explicit-once", "objective_id": "Repair the original supervisor",
        "board_id": "legal-ir", "board_namespace": "legal-ir", "candidate_policy_sha256": auth.digest(policy),
        "max_attempts": 3, "prior_job": prior, "prior_incident_signature_sha256": hashlib.sha256(b"old").hexdigest(),
        "not_before_observed_at": now - 10, "expires_at": now + 3600}
    board = {"id": "legal-ir", "max_repair_attempts": 3, "repair_authorization_epoch": epoch}
    config = {"state_dir": str(root), "boards": [board],
              "repair_worker": {"candidate_policy": policy, "candidate_runtime": {"fixture": True}}}
    job = {"board_id": "legal-ir", "status": "validation_pending", "attempts": number,
        "repair_unit_token": "a" * 32, "repair_authorization_sha256": auth.digest(epoch),
        "latest_incident": {"board_id": "legal-ir", "signature": "new", "observed_at": now - 1},
        "next_attempt_at": now + 1800, "untouched_history": {"prior": ["must", "remain"]}}
    job_path = auth.repair_job_directory(root, board) / "job.json"
    job_path.parent.mkdir(parents=True)
    fleet.write_json(job_path, job)
    owner = object.__new__(runtime.CandidateRuntime)
    owner.config, owner.board, owner.job_path = config, board, job_path
    owner.token, owner.attempt_number = "a" * 32, number
    owner.checkout_lease_closed = owner.provider_closed = True
    owner.fleet = fleet
    owner.session = SimpleNamespace(metadata_root=invocation.parent, binding=negative["binding"],
        candidate_root=tmp_path / "worktree", freeze={"freeze_sha256": negative["freeze_sha256"],
                                                    "source_commitment": negative["source_commitment"]})
    observation = {"returncode": 0, "stdout": runtime._wire(observed).decode()}
    witness = frozen_json(tmp_path / "observer-wait.json", observation)
    owner.known_negative_validation = observed, witness, negative
    return owner, job, negative, observed, invocation


def test_retained_original_storage_requires_distinct_fully_closed_observation(tmp_path):
    negative, observed, _ = negative_fixture(tmp_path)
    before = runtime._pin(Path(observed["negative_validation"]["path"]))
    observed["closure_observation"]["storage_closed"] = False
    with pytest.raises(ValueError):
        check(observed, negative)
    observed["closure_observation"]["storage_closed"] = True
    assert check(observed, negative)["storage_closed"] is False
    assert runtime._pin(Path(observed["negative_validation"]["path"])) == before


@pytest.mark.parametrize("field,value", [
    ("source_closed", False), ("native_closed", False), ("processes_closed", False),
    ("cleanup_errors", ["unknown"]), ("passed", True), ("semantic_negative", False),
])
def test_unknown_original_negative_custody_never_authorizes_retry(tmp_path, field, value):
    negative, observed, _ = negative_fixture(tmp_path)
    changed = dict(negative, **{field: value})
    pin = frozen_json(tmp_path / "changed-negative.json", changed)
    observed["negative_validation"] = pin
    observed["closure_observation"]["negative_validation"] = pin
    with pytest.raises(ValueError):
        check(observed, negative)


@pytest.mark.parametrize("field,value", [
    ("reservation_id", "f" * 32), ("native_lease_id", "f" * 32),
    ("freeze_sha256", "f" * 64), ("source_commitment", "f" * 64),
    ("original_owner_identity", {"pid": 999, "birth": "999"}), ("original_owner_returncode", 2),
])
def test_foreign_closure_cannot_close_this_original_negative(tmp_path, field, value):
    negative, observed, _ = negative_fixture(tmp_path)
    observed["closure_observation"][field] = value
    with pytest.raises(ValueError):
        check(observed, negative)


def test_mutable_original_negative_proof_is_held(tmp_path):
    negative, observed, _ = negative_fixture(tmp_path)
    Path(observed["negative_validation"]["path"]).chmod(0o600)
    with pytest.raises(ValueError):
        check(observed, negative)


@pytest.mark.parametrize("terminal", [
    {"command_returncode": None, "verifier_returncode": None},
    {"command_returncode": 0, "verifier_returncode": 1},
    {"command_returncode": 0, "verifier_returncode": None},
    {"command_returncode": 1, "verifier_returncode": 1},
])
def test_positive_or_ambiguous_terminal_does_not_launch_negative_observer(terminal):
    owner = object.__new__(runtime.CandidateRuntime)
    assert owner._closed_negative_validation(None, None, terminal) is None


@pytest.mark.parametrize("observer_returncode", [0, 1])
def test_negative_path_only_observes_original_owner_without_replaying_it(tmp_path, observer_returncode):
    owner, job, negative, observed, invocation = epoch_runtime(tmp_path)
    del owner.known_negative_validation
    owner.counter = 0
    owner.directory = tmp_path / "runtime"
    owner.directory.mkdir()
    owner.policy = SimpleNamespace(repository_root=tmp_path)
    owner._remaining_seconds = lambda: 1000
    owner._job = lambda: job
    guards = []
    owner.authority = SimpleNamespace(require_current=lambda stage, binding: guards.append((stage, binding)))
    source = frozen_json(tmp_path / "trusted-owner-source.json", {"fixture": "source-pinned original owner"})
    command = SimpleNamespace(argv=("python", "owner.py", "validate", "{candidate_root}", "{invocation_root}"),
        verify_argv=("python", "owner.py", "verify-validation", "{candidate_root}", "{invocation_root}"),
        timeout_seconds=30, source_descriptors=(source,))
    expected_original = ["python", "owner.py", "validate", str(owner.session.candidate_root), str(invocation)]
    expected_verify = ["python", "owner.py", "verify-validation", str(owner.session.candidate_root), str(invocation)]
    frozen_json(invocation / "intent.json", {"schema": runtime.SCHEMA, "stage": "validate",
        "argv": expected_original, "verify_argv": expected_verify, "owner_source_descriptors": [source],
        "runtime_policy_sha256": runtime.candidate.digest(owner.config["repair_worker"]["candidate_runtime"]),
        "attempt_token": owner.token, "attempt_number": owner.attempt_number,
        "freeze_sha256": negative["freeze_sha256"], "source_commitment": negative["source_commitment"]})
    terminal = {"command_returncode": 1, "verifier_returncode": None}
    frozen_json(invocation / "terminal.json", terminal)
    calls = []
    def observe(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=observer_returncode, stdout=runtime._wire(observed).decode(), stderr="")
    owner.command_runner = observe
    result = owner._closed_negative_validation(command, invocation, terminal)
    assert len(calls) == 1
    assert calls[0][0] == ["python", "owner.py", "verify-negative-validation",
                          str(owner.session.candidate_root), str(invocation)]
    assert calls[0][1] == {"cwd": tmp_path, "timeout_seconds": 30}
    assert "validate" not in calls[0][0]
    if observer_returncode == 0:
        assert result == negative and len(guards) == 2
        assert owner.known_negative_validation[0] == observed
    else:
        assert result is None and not hasattr(owner, "known_negative_validation")


@pytest.mark.parametrize("closed", ["checkout_lease_closed", "provider_closed"])
def test_failure_cannot_requeue_before_original_owner_closes(tmp_path, closed):
    owner, job, *_ = epoch_runtime(tmp_path)
    setattr(owner, closed, False)
    before = copy.deepcopy(job)
    with pytest.raises(ValueError):
        owner._negative_failure_transition(job, {"fixture": "bootstrap"})
    assert job == before


def test_closed_failure_preserves_same_epoch_attempt_and_original_cooldown(tmp_path):
    owner, job, *_ = epoch_runtime(tmp_path)
    before = copy.deepcopy(job)
    bootstrap = frozen_json(tmp_path / "bootstrap.json", {"original_checkout_lease_closed": True})
    result = owner._negative_failure_transition(job, bootstrap)
    assert result["status"] == "queued"
    for field in ("attempts", "repair_unit_token", "repair_authorization_sha256", "latest_incident",
                  "next_attempt_at", "untouched_history"):
        assert result[field] == before[field]
    assert result["candidate_validation_failures"][0]["closed_bootstrap"] == bootstrap
    assert result["candidate_validation_failures"][0]["coding_attempts_incremented"] is False
    assert auth.require_current_authorization(owner.config, owner.board, job=result).value["max_attempts"] == 3


def test_final_spent_attempt_uses_original_escalation_without_fourth_retry(tmp_path):
    owner, job, *_ = epoch_runtime(tmp_path, number=3)
    result = owner._negative_failure_transition(job, frozen_json(tmp_path / "bootstrap.json", {}))
    assert result["attempts"] == 3
    assert result["status"] == "escalated"
    assert result["escalation_reason"] == "repair_budget_exhausted"
    assert result["max_repair_attempts"] == 3


def test_failed_attempt_history_cannot_be_replayed_or_erased(tmp_path):
    owner, job, *_ = epoch_runtime(tmp_path)
    original = {"attempt_number": 1, "attempt_token": owner.token, "marker": "retain"}
    job["candidate_validation_failures"] = [original]
    before = copy.deepcopy(job)
    with pytest.raises(ValueError):
        owner._negative_failure_transition(job, {"fixture": "bootstrap"})
    assert job == before


def test_expired_epoch_stays_held_without_changing_spent_history(tmp_path):
    owner, job, *_ = epoch_runtime(tmp_path)
    epoch = owner.board["repair_authorization_epoch"]
    epoch["not_before_observed_at"] = time.time() - 3600
    epoch["expires_at"] = time.time() - 1
    job["repair_authorization_sha256"] = auth.digest(epoch)
    before = copy.deepcopy(job)
    with pytest.raises(ValueError):
        owner._negative_failure_transition(job, {"fixture": "bootstrap"})
    assert job == before


@pytest.mark.parametrize("profile", ["trusted_pytest_call_failure", "changed_candidate_source_syntax_error"])
def test_only_explicit_semantic_negative_profiles_allow_retry(tmp_path, profile):
    negative, observed, _ = negative_fixture(tmp_path)
    negative["negative_profile"] = profile
    pin = frozen_json(tmp_path / "selected-profile.json", negative)
    observed["negative_validation"] = pin
    observed["closure_observation"].update(negative_validation=pin, negative_profile=profile)
    assert check(observed, negative)["negative_profile"] == profile


@pytest.mark.parametrize("profile", [None, "infrastructure_error"])
def test_missing_or_infrastructure_negative_profile_stays_held(tmp_path, profile):
    negative, observed, _ = negative_fixture(tmp_path)
    negative["negative_profile"] = profile
    pin = frozen_json(tmp_path / "unsupported-profile.json", negative)
    observed["negative_validation"] = pin
    observed["closure_observation"].update(negative_validation=pin, negative_profile=profile)
    with pytest.raises(ValueError):
        check(observed, negative)


def test_closure_cannot_substitute_another_semantic_negative_profile(tmp_path):
    negative, observed, _ = negative_fixture(tmp_path)
    observed["closure_observation"]["negative_profile"] = "changed_candidate_source_syntax_error"
    with pytest.raises(ValueError):
        check(observed, negative)
