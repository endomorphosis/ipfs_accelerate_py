"""Actual private Git candidates stay separate from runtime and adoption authority.

These tests exercise the normal canonical module and original checkout/lifecycle
owners. Their source/resource callbacks are explicit private fixture owners;
they are not host admission, provider execution, or live runtime qualification.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate as candidate
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    acquire_checkout_mutation_lease, checkout_lock_metadata,
    checkout_mutation_lock_path, release_checkout_mutation_lease,
)


def git(root, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
        "-c", "user.name=Candidate Fixture", "-c", "user.email=candidate@example.invalid", *args],
        cwd=root, env=env, text=True, capture_output=True, timeout=15, check=True)
    return result.stdout.strip()


@pytest.fixture
def fixture_owner(tmp_path, monkeypatch):
    # Keep optional derived metadata outside the real user's orchestration tree.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_HOME", str(tmp_path / "agent-home"))
    repository = tmp_path / "repository"
    state = tmp_path / "repair-state"
    repository.mkdir(); state.mkdir(mode=0o700)
    git(repository, "init", "--initial-branch=main")
    (repository / "app.py").write_text("VALUE = 1\n")
    (repository / "test_app.py").write_text("def test_value():\n    assert True\n")
    git(repository, "add", "app.py", "test_app.py")
    git(repository, "commit", "-m", "private candidate baseline")
    base = git(repository, "rev-parse", "HEAD")
    policy_value = dict(schema=candidate.POLICY_SCHEMA, repository_root=str(repository),
        state_root=str(state), namespace="ptr_lane_0", base_commit=base,
        source_paths=["app.py", "test_app.py"], allowed_paths=["app.py", "new.py"],
        max_source_bytes=65536, max_candidate_bytes=131072, max_files=8,
        git_timeout_seconds=15)
    policy = candidate.parse_candidate_policy({"candidate_policy": policy_value})
    metadata = checkout_lock_metadata(kind="repair-candidate-fixture", repo_root=repository,
        task_id="fixture-board", attempt=1)
    lease, reason, _, _ = acquire_checkout_mutation_lease(
        checkout_mutation_lock_path(repository), metadata, owner_active=lambda _: True)
    assert reason == "acquired" and lease is not None
    calls = []

    def original_guard(kind):
        def guard(stage, binding):
            calls.append((kind, stage))
            return candidate.OwnerGuardReceipt(kind, "private-fixture:" + kind, candidate.digest(binding))
        return guard

    def result_guard(kind):
        def guard(binding, receipt):
            # The independent fixture owner validates an exact preselected
            # original receipt reference; arbitrary success JSON is refused.
            if receipt.get("original_receipt_reference") != "private-original:" + kind:
                raise candidate.CandidateError("fixture original receipt not selected")
            return candidate.OwnerGuardReceipt(kind, receipt["original_receipt_reference"],
                candidate.digest(dict(binding=binding, receipt=receipt)))
        return guard

    authority = candidate.ExistingRepairAuthority(checkout_lease=lease,
        source_guard=original_guard("source"), resource_guard=original_guard("resource"),
        validation_guard=result_guard("validate"), adoption_guard=result_guard("adopt"))
    value = dict(repository=repository, state=state, policy=policy, policy_value=policy_value,
                 lease=lease, authority=authority, calls=calls)
    yield value
    assert release_checkout_mutation_lease(lease, require_exact_metadata=True)


def prepare(owner, **kwargs):
    values = dict(policy=owner["policy"], board_id="fixture-board", attempt_id="attempt_1",
                  attempt_number=1, active_root=owner["repository"], authority=owner["authority"])
    values.update(kwargs)
    return candidate.prepare_candidate(**values)


def frozen(owner):
    session = prepare(owner)
    (session.candidate_root / "app.py").write_text("VALUE = 2\n")
    candidate.freeze_candidate(session)
    return session


def validation(session, **changes):
    value = dict(schema=candidate.VALIDATION_SCHEMA, freeze_sha256=session.freeze["freeze_sha256"],
        source_commitment=session.freeze["source_commitment"], passed=True,
        source_closed=True, native_closed=True, processes_closed=True, storage_closed=True,
        cleanup_errors=[], original_receipt_reference="private-original:validate")
    value.update(changes)
    return value


def adoption(session, **changes):
    value = dict(schema=candidate.ADOPTION_SCHEMA, freeze_sha256=session.freeze["freeze_sha256"],
        validation_sha256=candidate.digest(session.validation), adopted=True,
        source_closed=True, native_closed=True, processes_closed=True, storage_closed=True,
        cleanup_errors=[], original_receipt_reference="private-original:adopt",
        runtime_release="private-qualified-release", loaded_source_commitment=session.freeze["source_commitment"])
    value.update(changes)
    return value


def test_unconfigured_policy_is_opt_in_only():
    assert candidate.parse_candidate_policy({}) is None


@pytest.mark.parametrize("field,value", [
    ("max_files", True), ("max_source_bytes", 0), ("max_candidate_bytes", 1.0),
    ("base_commit", "HEAD"), ("namespace", "../other"),
    ("source_paths", ["../foreign.py"]), ("allowed_paths", [".git/config"]),
])
def test_policy_rejects_unbounded_or_ambiguous_scope(fixture_owner, field, value):
    payload = dict(fixture_owner["policy_value"])
    payload[field] = value
    with pytest.raises(candidate.CandidateError):
        candidate.parse_candidate_policy({"candidate_policy": payload})
    assert not (fixture_owner["state"] / "worktrees").exists()


def test_policy_refuses_unknown_keys_and_source_overlap(fixture_owner):
    payload = dict(fixture_owner["policy_value"], extra=True)
    with pytest.raises(candidate.CandidateError):
        candidate.parse_candidate_policy({"candidate_policy": payload})
    payload.pop("extra")
    payload["state_root"] = str(fixture_owner["repository"])
    with pytest.raises(candidate.CandidateError, match="disjoint"):
        candidate.parse_candidate_policy({"candidate_policy": payload})


def test_real_private_sparse_candidate_is_claimed_before_git(fixture_owner):
    original_runner = candidate.run_command
    observed = []

    def runner(argv, **kwargs):
        if "worktree" in argv and argv[argv.index("worktree"):argv.index("worktree") + 2] == ["worktree", "add"]:
            path = Path(argv[-2])
            store = candidate.WorktreeLifecycleStore(repo_root=fixture_owner["repository"])
            record = store.load_workspace(path)
            assert record is not None and record.state.value == "preparing"
            assert not path.exists()
            observed.append(record.record_id)
        return original_runner(argv, **kwargs)

    session = prepare(fixture_owner, run_command_fn=runner)
    assert len(observed) == 1
    assert session.candidate_root != fixture_owner["repository"]
    assert set(session.baseline) == {"app.py", "test_app.py"}
    assert session.record.state.value == "active"
    assert (session.metadata_root / "prepared.json").exists()
    assert git(fixture_owner["repository"], "status", "--porcelain") == ""


def test_boolean_authority_cannot_create_any_candidate(fixture_owner):
    with pytest.raises(candidate.CandidateError, match="typed original"):
        prepare(fixture_owner, authority=True)
    assert not (fixture_owner["state"] / "worktrees").exists()


def test_original_lease_loss_refuses_before_git_or_provider(fixture_owner):
    assert release_checkout_mutation_lease(fixture_owner["lease"], require_exact_metadata=True)
    with pytest.raises(candidate.CandidateError, match="lease"):
        prepare(fixture_owner)
    assert not (fixture_owner["state"] / "worktrees").exists()


def test_current_guard_rejects_serialized_or_misbinding_success(fixture_owner):
    authority = candidate.ExistingRepairAuthority(checkout_lease=fixture_owner["lease"],
        source_guard=lambda stage, binding: {"current": True},
        resource_guard=lambda stage, binding: candidate.OwnerGuardReceipt("resource", "fixture", "0" * 64))
    with pytest.raises(candidate.CandidateError, match="guard refused"):
        prepare(fixture_owner, authority=authority)
    assert not (fixture_owner["state"] / "worktrees").exists()


def test_provider_environment_arms_existing_boundary_and_candidate_freeze(fixture_owner):
    session = prepare(fixture_owner)
    env = candidate.provider_environment(session)
    assert env[candidate.PROTECTED_STATE_ENV] == str(fixture_owner["state"])
    assert env["IPFS_PROOF_REUSE_STATE_ROOT"] == ""
    assert session.candidate_root.relative_to(fixture_owner["state"]).parts == (
        "worktrees", "ptr_lane_0", "attempt_1")
    (session.candidate_root / "app.py").write_text("VALUE = 2\n")
    frozen_report = candidate.freeze_candidate(session)
    assert frozen_report["stage"] == "candidate_ready"
    assert frozen_report["changed_paths"] == ["app.py"]
    assert frozen_report["runtime_adopted"] is False
    assert frozen_report["coding_attempts_incremented"] is False
    assert (fixture_owner["repository"] / "app.py").read_text() == "VALUE = 1\n"
    assert session.record.state.value == "settling"
    with pytest.raises(candidate.CandidateError):
        candidate.provider_environment(session)


def test_existing_inherited_boundary_denies_active_and_shared_git_writes(fixture_owner, tmp_path):
    """One real private Python child tests the reused boundary, never an LLM."""
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import provider_readonly_state_command

    session = prepare(fixture_owner)
    private_home = tmp_path / "private-provider-home"
    private_home.mkdir(mode=0o700)
    shared_git = Path(git(fixture_owner["repository"], "rev-parse", "--absolute-git-dir"))
    child = """import json, pathlib, sys
candidate, active, shared = map(pathlib.Path, sys.argv[1:])
(candidate / 'app.py').write_text('VALUE = 2\\n')
denied = []
for target in (active / 'app.py', shared / 'config'):
    try:
        with target.open('a') as handle: handle.write('unauthorized')
    except PermissionError:
        denied.append(str(target))
assert len(denied) == 2, denied
print(json.dumps({'candidate_write': True, 'protected_denials': len(denied)}))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    wrapped, receipt = provider_readonly_state_command(
        [sys.executable, "-I", "-B", "-c", child, str(session.candidate_root),
         str(fixture_owner["repository"]), str(shared_git)],
        state_root_path=fixture_owner["state"], workspace_path=session.candidate_root,
        private_home_path=private_home, checkpoint_path=None,
        provider_profile_paths=(), environment=env)
    observed = subprocess.run(wrapped, cwd=session.candidate_root, env=env,
                              text=True, capture_output=True, timeout=20, check=True)
    assert json.loads(observed.stdout) == {"candidate_write": True, "protected_denials": 2}
    assert receipt.landlock_abi >= 1
    assert (fixture_owner["repository"] / "app.py").read_text() == "VALUE = 1\n"
    assert candidate.freeze_candidate(session)["changed_paths"] == ["app.py"]


@pytest.mark.parametrize("damage", ["outside_path", "delete", "symlink", "hardlink", "overrun", "git_binding", "active_source"])
def test_candidate_damage_is_retained_and_never_qualifies(fixture_owner, damage):
    session = prepare(fixture_owner)
    source = session.candidate_root / "app.py"
    source.write_text("VALUE = 2\n")
    if damage == "outside_path":
        (session.candidate_root / "foreign.py").write_text("unauthorized\n")
    elif damage == "delete":
        (session.candidate_root / "test_app.py").unlink()
    elif damage == "symlink":
        source.unlink(); source.symlink_to(fixture_owner["repository"] / "app.py")
    elif damage == "hardlink":
        source.unlink(); os.link(fixture_owner["repository"] / "app.py", source)
    elif damage == "overrun":
        source.write_bytes(b"x" * (fixture_owner["policy"].max_candidate_bytes + 1))
    elif damage == "git_binding":
        (session.candidate_root / ".git").write_text("gitdir: /unrelated\n")
    else:
        (fixture_owner["repository"] / "app.py").write_text("concurrent active change\n")
    with pytest.raises(candidate.CandidateError):
        candidate.freeze_candidate(session)
    assert session.candidate_root.exists() and session.metadata_root.exists()
    assert not (session.metadata_root / "candidate-ready.json").exists()
    # These are fixture-authored corrupt aliases. Preserve the rejection and
    # the candidate above, then remove only the alias so terminal fixture
    # inventory is ordinary; the product itself performs no deletion.
    if damage in {"symlink", "hardlink"}:
        source.unlink()


def test_noop_and_duplicate_attempt_cannot_recode_or_delete_evidence(fixture_owner):
    session = prepare(fixture_owner)
    with pytest.raises(candidate.CandidateError, match="no candidate source"):
        candidate.freeze_candidate(session)
    prepared = (session.metadata_root / "prepared.json").read_bytes()
    with pytest.raises(candidate.CandidateError, match="already exists"):
        prepare(fixture_owner)
    assert (session.metadata_root / "prepared.json").read_bytes() == prepared


@pytest.mark.parametrize("changes", [
    {"passed": False}, {"passed": 1}, {"freeze_sha256": "0" * 64},
    {"source_closed": False}, {"processes_closed": "unknown"},
    {"cleanup_errors": ["incomplete"]}, {"original_receipt_reference": "agent-claimed-success"},
])
def test_failed_stale_or_agent_asserted_validation_cannot_admit_adoption(fixture_owner, changes):
    session = frozen(fixture_owner)
    with pytest.raises(candidate.CandidateError):
        candidate.record_validation(session, validation(session, **changes))
    assert session.stage == "candidate_ready"
    assert not (session.metadata_root / "adoption-pending.json").exists()


def test_missing_original_verifier_holds_without_new_coding_attempt(fixture_owner):
    session = frozen(fixture_owner)
    authority = session.authority
    authority.validation_guard = None
    with pytest.raises(candidate.CandidateError, match="not configured"):
        candidate.record_validation(session, validation(session))
    assert session.stage == "candidate_ready"
    assert json.loads((session.metadata_root / "candidate-ready.json").read_text())["coding_attempts_incremented"] is False


def test_validated_candidate_waits_for_actual_independent_adoption(fixture_owner):
    session = frozen(fixture_owner)
    pending = candidate.record_validation(session, validation(session))
    assert pending["stage"] == "adoption_pending" and pending["runtime_adopted"] is False
    assert pending["coding_attempts_incremented"] is False
    with pytest.raises(candidate.CandidateError):
        candidate.record_adoption(session, adoption(session, adopted=False))
    with pytest.raises(candidate.CandidateError):
        candidate.record_adoption(session, adoption(session, loaded_source_commitment="0" * 64))
    with pytest.raises(candidate.CandidateError):
        candidate.record_adoption(session, adoption(session, original_receipt_reference="agent report"))
    accepted = candidate.record_adoption(session, adoption(session))
    assert accepted["stage"] == "adopted_pending_probe"
    assert accepted["fresh_observer_probe_required"] is True
    assert accepted["observer_verified"] is False
    assert accepted["completion_authority"] is False
    assert session.candidate_root.exists()


def test_postfreeze_change_invalidates_even_a_positive_validation(fixture_owner):
    session = frozen(fixture_owner)
    (session.candidate_root / "app.py").write_text("VALUE = 3\n")
    with pytest.raises(candidate.CandidateError, match="frozen candidate source changed"):
        candidate.record_validation(session, validation(session))
    assert session.stage == "candidate_ready"


def test_direct_policy_constructor_cannot_bypass_finite_scope(fixture_owner):
    bad = replace(fixture_owner["policy"], max_files=True)
    with pytest.raises(candidate.CandidateError):
        prepare(fixture_owner, policy=bad)
    assert not (fixture_owner["state"] / "worktrees").exists()


_FROZEN_CHILD = """import json, pathlib, sys
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate as c
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import read_checkout_mutation_lease
policy = c.parse_candidate_policy({'candidate_policy': json.loads(pathlib.Path(sys.argv[1]).read_text())})
lease = read_checkout_mutation_lease(pathlib.Path(sys.argv[2]))
def guard(kind):
    return lambda stage, binding: c.OwnerGuardReceipt(kind, 'private-child-fixture:' + kind, c.digest(binding))
def validate(binding, receipt):
    assert receipt['original_receipt_reference'] == 'private-original:validate'
    return c.OwnerGuardReceipt('validate', 'private-original:validate', c.digest(dict(binding=binding, receipt=receipt)))
authority = c.ExistingRepairAuthority(checkout_lease=lease, source_guard=guard('source'),
    resource_guard=guard('resource'), validation_guard=validate)
session = c.prepare_candidate(policy=policy, board_id='fixture-board', attempt_id='attempt_1',
    attempt_number=1, active_root=pathlib.Path(sys.argv[3]), authority=authority)
(session.candidate_root / 'app.py').write_text('VALUE = 2\\n')
frozen = c.freeze_candidate(session)
if sys.argv[4] == 'validated':
    receipt = dict(schema=c.VALIDATION_SCHEMA, freeze_sha256=frozen['freeze_sha256'],
        source_commitment=frozen['source_commitment'], passed=True, source_closed=True,
        native_closed=True, processes_closed=True, storage_closed=True, cleanup_errors=[],
        original_receipt_reference='private-original:validate')
    c.record_validation(session, receipt)
print(json.dumps({'freeze_sha256': frozen['freeze_sha256'], 'stage': session.stage}), flush=True)
if sys.argv[4] == 'wait':
    sys.stdin.readline()
"""


def frozen_child(owner, *, wait=False, validated=False):
    """Real private owner exits; no PID fabrication or mocked dead-owner guard."""
    policy_file = owner["state"] / "private-child-policy.json"
    policy_file.write_text(json.dumps(owner["policy_value"]))
    # A normal child imports the actual selected canonical source context.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    root = str(Path(candidate.__file__).resolve().parents[3])
    env["PYTHONPATH"] = root
    process = subprocess.Popen([sys.executable, "-B", "-c", _FROZEN_CHILD,
        str(policy_file), str(owner["lease"].lock_path), str(owner["repository"]),
        "wait" if wait else "validated" if validated else "closed"],
        cwd=owner["repository"], env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if wait:
        try:
            line = process.stdout.readline()
            assert line, "private child failed before frozen receipt"
            result = json.loads(line)
        except BaseException:
            process.communicate("close\n", timeout=20)
            raise
        return process, result
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    return process, json.loads(stdout)


def resume(owner):
    return candidate.resume_candidate(policy=owner["policy"], board_id="fixture-board",
        attempt_id="attempt_1", attempt_number=1, active_root=owner["repository"],
        authority=owner["authority"])


def test_resume_same_owner_preserves_freeze_and_spends_no_coding_attempt(fixture_owner):
    original = frozen(fixture_owner)
    before = (original.metadata_root / "candidate-ready.json").read_bytes()
    resumed = resume(fixture_owner)
    assert resumed.stage == "candidate_ready" and resumed.record == original.record
    assert resumed.freeze == original.freeze
    assert (original.metadata_root / "candidate-ready.json").read_bytes() == before
    receipts = list(original.metadata_root.glob("resumed-fence-*.json"))
    assert len(receipts) == 1
    value = json.loads(receipts[0].read_text())
    assert value["dead_owner_transferred"] is False
    assert value["coding_attempts_incremented"] is False and value["provider_dispatched"] is False
    assert resume(fixture_owner).record == resumed.record


@pytest.mark.parametrize("validated", [False, True])
def test_resume_exact_dead_private_owner_uses_original_fenced_transfer(fixture_owner, validated):
    process, original = frozen_child(fixture_owner, validated=validated)
    assert process.returncode == 0
    metadata = fixture_owner["state"] / "repair-candidates/fixture-board/attempt_1"
    preserved = {path.name: path.read_bytes() for path in metadata.iterdir()}
    resumed = resume(fixture_owner)
    assert resumed.stage == ("adoption_pending" if validated else "candidate_ready")
    assert resumed.freeze["freeze_sha256"] == original["freeze_sha256"]
    assert resumed.record.owner.pid == os.getpid() and resumed.record.state.value == "settling"
    assert all((metadata / name).read_bytes() == payload for name, payload in preserved.items())
    receipts = list(metadata.glob("resumed-fence-*.json"))
    value = json.loads(receipts[0].read_text())
    assert value["dead_owner_transferred"] is True
    assert value["original_lifecycle"]["owner"]["pid"] == process.pid
    assert value["resumed_lifecycle"]["fence"] == value["original_lifecycle"]["fence"] + 2
    assert value["resumed_lifecycle"]["lease_id"] != value["original_lifecycle"]["lease_id"]
    assert value["provider_dispatched"] is False and value["coding_attempts_incremented"] is False


def test_resume_cannot_adopt_a_live_original_private_owner(fixture_owner):
    process, _ = frozen_child(fixture_owner, wait=True)
    try:
        metadata = fixture_owner["state"] / "repair-candidates/fixture-board/attempt_1"
        before = {path.name: path.read_bytes() for path in metadata.iterdir()}
        with pytest.raises(Exception, match="alive|dead|owner"):
            resume(fixture_owner)
        assert {path.name: path.read_bytes() for path in metadata.iterdir()} == before
    finally:
        _, stderr = process.communicate("close\n", timeout=20)
        assert process.returncode == 0, stderr


@pytest.mark.parametrize("damage", ["policy", "freeze_digest", "frozen_source", "validation"])
def test_resume_refuses_stale_or_changed_retained_evidence_without_transfer(fixture_owner, damage):
    process, _ = frozen_child(fixture_owner, validated=damage == "validation")
    assert process.returncode == 0
    metadata = fixture_owner["state"] / "repair-candidates/fixture-board/attempt_1"
    if damage == "policy":
        changed = dict(fixture_owner["policy_value"], max_candidate_bytes=262144)
        fixture_owner["policy"] = candidate.parse_candidate_policy({"candidate_policy": changed})
    elif damage == "freeze_digest":
        path = metadata / "candidate-ready.json"
        value = json.loads(path.read_text()); value["freeze_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    elif damage == "frozen_source":
        (fixture_owner["state"] / "worktrees/ptr_lane_0/attempt_1/app.py").write_text("VALUE = 3\n")
    else:
        path = metadata / "adoption-pending.json"
        value = json.loads(path.read_text()); value["validation"]["passed"] = False
        path.write_text(json.dumps(value))
    store = candidate.WorktreeLifecycleStore(repo_root=fixture_owner["repository"])
    old_record = store.load_workspace(fixture_owner["state"] / "worktrees/ptr_lane_0/attempt_1")
    with pytest.raises(candidate.CandidateError):
        resume(fixture_owner)
    assert store.load_workspace(fixture_owner["state"] / "worktrees/ptr_lane_0/attempt_1") == old_record
    assert not list(metadata.glob("resumed-fence-*.json"))
