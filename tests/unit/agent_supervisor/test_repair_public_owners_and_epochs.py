"""Private protocol fixtures; no live service, provider or native resource API.

These are prospective unit regressions. Genuine public-D execution, Git source
publication and service activation require distinct original-owner integration
qualification; modeled observer data below grants no runtime authority.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as auth
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(auth.canonical(value) + b"\n")
    return auth.descriptor(path)


@pytest.fixture
def scope(tmp_path):
    state = tmp_path / "state"; state.mkdir()
    prior = {"board_id": "legal-ir", "attempts": 3, "status": "verified_healthy",
        "escalation_reason": "repair_budget_exhausted", "repair_unit_token": "a" * 32,
        "latest_incident": {"signature": "old-incident"}}
    prior_pin = put(state / "repairs/legal-ir/job.json", prior)
    policy = {"schema": "agent-supervisor/isolated-repair-candidate-policy@1",
        "repository_root": str(tmp_path / "source"), "state_root": str(state), "namespace": "repair_test",
        "base_commit": "a" * 40, "source_paths": ["subject.py"], "allowed_paths": ["subject.py"],
        "max_source_bytes": 1024, "max_candidate_bytes": 4096, "max_files": 4, "git_timeout_seconds": 10}
    epoch = {"schema": auth.SCHEMA, "epoch_id": "explicit-test-epoch", "objective_id": "private-test-objective",
        "board_id": "legal-ir", "board_namespace": "repair_test", "candidate_policy_sha256": auth.digest(policy),
        "max_attempts": 3, "prior_job": prior_pin,
        "prior_incident_signature_sha256": hashlib.sha256(b"old-incident").hexdigest(),
        "not_before_observed_at": 1000, "expires_at": 1200}
    board = {"id": "legal-ir", "max_repair_attempts": 3, "repair_authorization_epoch": epoch}
    config = {"state_dir": str(state), "boards": [board], "repair_worker": {"candidate_policy": policy}}
    incident = {"board_id": "legal-ir", "observed_at": 1100, "signature": "new-incident"}
    put(state / "legal-ir/state.json", {"observed_at": 1100, "incident_signature": "new-incident"})
    return config, board, incident, prior_pin


def test_legacy_job_path_and_spent_history_are_never_reset(scope):
    config, board, incident, before = scope
    plain = {"id": "legal-ir", "max_repair_attempts": 3}
    assert auth.repair_job_directory(Path(config["state_dir"]), plain) == Path(before["path"]).parent
    epoch = auth.require_current_authorization(config, board, incident=incident, now=1100)
    assert epoch.directory == Path(config["state_dir"]) / "repair-epochs/explicit-test-epoch/repairs/legal-ir"
    assert auth.descriptor(Path(before["path"])) == before
    assert not (epoch.directory / "job.json").exists()


@pytest.mark.parametrize("failure", ["board", "old_signature", "old_sample", "expired", "budget", "namespace", "prior_changed", "boolean_budget"])
def test_epoch_refuses_before_queue_or_coding_charge(scope, failure):
    config, board, incident, prior = scope
    now = 1100
    if failure == "board":
        board["repair_authorization_epoch"]["board_id"] = "foreign"
    elif failure == "old_signature":
        incident["signature"] = "old-incident"
    elif failure == "old_sample":
        incident["observed_at"] = 999
    elif failure == "expired":
        now = 1200
    elif failure == "budget":
        board["repair_authorization_epoch"]["max_attempts"] = 4
    elif failure == "namespace":
        config["repair_worker"]["candidate_policy"]["namespace"] = "foreign"
    elif failure == "prior_changed":
        Path(prior["path"]).write_bytes(b"{}\n")
    else:
        board["repair_authorization_epoch"]["max_attempts"] = True
    with pytest.raises(ValueError):
        auth.require_current_authorization(config, board, incident=incident, now=now)
    assert not (Path(config["state_dir"]) / "repair-epochs").exists()
    if failure != "prior_changed":
        assert auth.descriptor(Path(prior["path"])) == prior


def test_same_epoch_cap_and_job_binding_cannot_be_replenished(scope):
    config, board, incident, before = scope
    epoch = auth.parse_authorization_epoch(Path(config["state_dir"]), board)
    job = {"attempts": 3, "latest_incident": incident, "repair_authorization_sha256": epoch.commitment}
    assert auth.require_current_authorization(config, board, job=job, now=1100) == epoch
    job["attempts"] = 4
    with pytest.raises(ValueError, match="attempt"):
        auth.require_current_authorization(config, board, job=job, now=1100)
    assert auth.descriptor(Path(before["path"])) == before


def source_link(scope, tmp_path):
    config, board, incident, _ = scope
    epoch = auth.parse_authorization_epoch(Path(config["state_dir"]), board)
    prior = config["repair_worker"]["candidate_policy"]
    successor = {**prior, "base_commit": "b" * 40}
    frozen = {"freeze_sha256": "c" * 64, "source_commitment": "d" * 64}
    validation = {"schema": owners.VALIDATION_SCHEMA, "passed": True, **frozen,
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True, "cleanup_errors": []}
    activation = {"completed": True, "private_fixture_observer": True}
    link = {"schema": "agent-supervisor/repair-source-successor@1", "epoch_sha256": epoch.commitment,
        "attempt_number": 1, "previous_link": None, "prior_policy": prior, "successor_policy": successor,
        "freeze": put(tmp_path / "frozen.json", frozen), "validation": put(tmp_path / "validated.json", validation),
        "activation": put(tmp_path / "activation.json", activation), "git_snapshot": {"fixture": True},
        "external_terminal": {"private_fixture_observer": True}}
    return epoch, successor, link


def fixture_activation_observer(link, frozen, activation):
    # Models only an authenticated existing observer's return. Separate owning
    # integration must exercise real Git/units/public-D custody before adoption.
    if activation != {"completed": True, "private_fixture_observer": True}:
        raise ValueError("unknown original activation")


def test_source_successor_keeps_epoch_paths_budget_and_old_job(scope, tmp_path, monkeypatch):
    epoch, successor, link = source_link(scope, tmp_path)
    original = auth.canonical(epoch.value)
    monkeypatch.setattr(owners, "verify_source_successor_activation", fixture_activation_observer)
    auth.validate_source_successors(epoch, successor, [put(tmp_path / "link.json", link)], 1)
    assert auth.canonical(epoch.value) == original
    assert epoch.value["max_attempts"] == 3 and epoch.value["expires_at"] == 1200
    assert auth.descriptor(Path(scope[3]["path"])) == scope[3]


@pytest.mark.parametrize("failure", ["fork", "replay", "paths", "bounds", "anchor", "unknown"])
def test_source_successor_refuses_fork_widening_and_unknown(scope, tmp_path, monkeypatch, failure):
    epoch, successor, link = source_link(scope, tmp_path)
    monkeypatch.setattr(owners, "verify_source_successor_activation", fixture_activation_observer)
    if failure == "fork":
        link["previous_link"] = link["activation"]
    elif failure == "replay":
        link["attempt_number"] = 0
    elif failure == "paths":
        successor["allowed_paths"] = ["subject.py", "foreign.py"]
    elif failure == "bounds":
        successor["max_candidate_bytes"] += 1
    elif failure == "anchor":
        link["prior_policy"] = {**link["prior_policy"], "base_commit": "e" * 40}
    else:
        link["activation"] = put(tmp_path / "unknown.json", {"completed": False})
    with pytest.raises(ValueError):
        auth.validate_source_successors(epoch, successor, [put(tmp_path / "bad-link.json", link)], 1)


def test_loader_uses_broken_candidate_instead_of_passing_release(tmp_path):
    candidate, trusted, datasets = [tmp_path / name for name in ("candidate", "trusted", "datasets")]
    for root in (candidate, trusted, datasets):
        root.mkdir()
    target = candidate / "subject.py"; target.write_text("healthy = False\n")
    old = trusted / "subject.py"; old.write_text("healthy = True\n")
    pins = {str(target): worker.descriptor(target)["sha256"], str(old): worker.descriptor(old)["sha256"]}
    guard = worker.PinnedSources(pins, [candidate, datasets, trusted])
    spec = guard.find_spec("ipfs_accelerate_py.subject", [str(candidate)])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(AssertionError):
        assert module.healthy is True
    assert guard.executed == {str(target): pins[str(target)]}
    with pytest.raises(ImportError):
        guard.find_spec("ipfs_accelerate_py.subject", [str(trusted)])
    assert guard.denials


def test_caught_undeclared_source_refusal_remains_a_permanent_denial(tmp_path):
    root = tmp_path / "candidate"; root.mkdir()
    subject = root / "subject.py"; subject.write_text("healthy = True\n")
    hidden = root / "hidden.py"; hidden.write_text("healthy = True\n")
    guard = worker.PinnedSources({str(subject): worker.descriptor(subject)["sha256"]}, [root, tmp_path / "d", tmp_path / "trusted"])
    caught = False
    try:
        guard.find_spec("ipfs_accelerate_py.hidden", [str(root)])
    except ImportError:
        caught = True
    assert caught and guard.denials == [{"path": str(hidden), "reason": "undeclared first-party source body"}]
    # Even if the candidate catches the import exception and every natural
    # assertion passes, the worker's final acceptance predicate must refuse.
    with pytest.raises(ValueError):
        worker.need(not guard.denials, "full natural-phase/source evidence required")


def test_worker_rejects_cached_release_before_any_project_test_or_lease(tmp_path):
    policy = put(tmp_path / "policy.json", {"datasets_root": str(tmp_path / "datasets")})
    plan = put(tmp_path / "plan.json", {"schema": owners.WORK_SCHEMA, "policy": policy,
        "candidate_root": str(tmp_path / "candidate"), "attempt_directory": str(tmp_path / "attempt")})
    # This owning test deliberately already has the real project package loaded.
    # The absolute-script bootstrap must reject this state, not clear/fabricate
    # package modules or accidentally test the release instead of the candidate.
    with pytest.raises(ValueError, match="cached project"):
        worker.validation_worker(SimpleNamespace(plan=Path(plan["path"]), plan_sha256=plan["sha256"]))


def test_original_wait_refuses_replaced_process_primitive_before_launch(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    calls = []
    def replaced(*args, **kwargs):
        calls.append(args)
        raise AssertionError("replacement must not be called")
    monkeypatch.setattr(engine.subprocess, "Popen", replaced)
    policy = {"source_descriptors": [owners.descriptor(Path(engine.__file__).absolute())]}
    with pytest.raises(ValueError, match="process primitive"):
        owners.original_runner_wait(policy, ["never-launched"], tmp_path, 1)
    assert calls == []


@pytest.mark.parametrize("stage", ["initial_read", "before_compile"])
def test_caught_nonordinary_read_failure_is_a_permanent_denial(tmp_path, monkeypatch, stage):
    root = tmp_path / "candidate"; root.mkdir()
    source = root / "subject.py"; source.write_text("healthy = True\n")
    guard = worker.PinnedSources({str(source): worker.descriptor(source)["sha256"]}, [root, tmp_path / "d", tmp_path / "trusted"])
    spec = guard.find_spec("ipfs_accelerate_py.subject", [str(root)]) if stage == "before_compile" else None
    def changed_identity(*args, **kwargs):
        raise RuntimeError("owned file changed while observed")
    monkeypatch.setattr(worker, "ordinary_bytes", changed_identity)
    with pytest.raises(ImportError):
        if spec is None:
            guard.find_spec("ipfs_accelerate_py.subject", [str(root)])
        else:
            spec.loader.get_code("ipfs_accelerate_py.subject")
    assert len(guard.denials) == 1


def test_project_namespace_cannot_fall_through_to_a_later_finder(tmp_path):
    root = tmp_path / "candidate"; root.mkdir()
    # A real directory without __init__.py would otherwise return a normal
    # namespace spec. Missing ordinary origin is a permanent refusal, before
    # any later finder can synthesize a project module or use the old release.
    (root / "subject").mkdir()
    guard = worker.PinnedSources({}, [root, tmp_path / "d", tmp_path / "trusted"])
    with pytest.raises(ImportError):
        guard.find_spec("ipfs_accelerate_py.subject", [str(root)])
    assert guard.denials == [{"path": "ipfs_accelerate_py.subject",
                             "reason": "project package lacks a declared ordinary normal source origin"}]



def terminal_fixture():
    # Explicit private observations for pure checks; no original manager or
    # process is replaced, and these values are never runtime evidence.
    owner = {"unit": "ipfs-taskboard-legal-ir-cutover-" + "a" * 32 + ".service",
        "owner": {"pid": 12345, "birth": "100"}, "invocation_id": "b" * 32}
    properties = {"MainPID": "0", "InvocationID": "b" * 32, "Result": "success",
        "ExecMainStatus": "0", "SubState": "exited", "ControlGroup": "",
        "TasksCurrent": "0", "KillMode": "control-group"}
    return owner, properties


@pytest.mark.parametrize("process", [None, {"pid": 12345, "birth": "101"}])
def test_original_external_owner_closes_when_absent_or_pid_reused(process):
    owner, properties = terminal_fixture()
    assert owners.validate_external_activation_terminal(owner, properties, process) == properties


def test_original_external_owner_same_pid_and_birth_remains_live():
    owner, properties = terminal_fixture()
    with pytest.raises(ValueError, match="remain live"):
        owners.validate_external_activation_terminal(owner, properties, dict(owner["owner"]))


def test_external_history_requires_a_sealed_original_terminal_proof():
    owner, _ = terminal_fixture()
    with pytest.raises(ValueError, match="immutable authenticated"):
        owners.verify_external_activation_closed(owner, None, None)


def historical_terminal_fixture(tmp_path):
    owner, properties = terminal_fixture()
    activation_pin = put(tmp_path / "actual-activation.json", {"schema": owners.ACTIVATION_SCHEMA,
        "completed": True, "external_owner": owner})
    proof = {"schema": "agent-supervisor/repair-external-owner-terminal@1",
        "activation": activation_pin, "external_owner": owner, "terminal_properties": properties,
        "process_observation": {"pid": 12345, "birth": "101"}, "observed_at": 1100,
        "original_manager_terminal_verified": True}
    proof_pin = put(tmp_path / "external-owner-terminal.json", proof)
    Path(proof_pin["path"]).chmod(0o400)
    return owner, activation_pin, proof_pin, proof


def test_authenticated_historical_terminal_does_not_query_current_manager(tmp_path, monkeypatch):
    owner, activation, terminal, proof = historical_terminal_fixture(tmp_path)
    def unavailable_manager(*args, **kwargs):
        raise AssertionError("historical verification must not query vanished transient manager state")
    monkeypatch.setattr(owners, "observe_external_activation_closed", unavailable_manager)
    assert owners.verify_external_activation_closed(owner, terminal, activation) == proof["terminal_properties"]


@pytest.mark.parametrize("failure", ["foreign_activation", "foreign_owner", "unsealed", "same_live_identity",
    "incomplete_terminal", "unknown_result", "nonzero_tasks", "unknown_tasks", "foreign_kill_mode"])
def test_historical_external_terminal_refuses_unproved_or_mutable_closure(tmp_path, failure):
    owner, activation, terminal, proof = historical_terminal_fixture(tmp_path)
    if failure == "foreign_activation":
        proof["activation"] = {**activation, "sha256": "f" * 64}
    elif failure == "foreign_owner":
        proof["external_owner"] = {**owner, "invocation_id": "c" * 32}
    elif failure == "same_live_identity":
        proof["process_observation"] = dict(owner["owner"])
    elif failure == "incomplete_terminal":
        proof["terminal_properties"].pop("InvocationID")
    elif failure == "unknown_result":
        proof["terminal_properties"]["Result"] = "unknown"
    elif failure == "nonzero_tasks":
        proof["terminal_properties"]["TasksCurrent"] = "1"
    elif failure == "unknown_tasks":
        proof["terminal_properties"]["TasksCurrent"] = "unknown"
    elif failure == "foreign_kill_mode":
        proof["terminal_properties"]["KillMode"] = "process"
    path = Path(terminal["path"]); path.chmod(0o600)
    path.write_bytes(auth.canonical(proof) + b"\n")
    terminal = auth.descriptor(path)
    if failure != "unsealed":
        path.chmod(0o400)
    with pytest.raises(ValueError):
        owners.verify_external_activation_closed(owner, terminal, activation)


@pytest.mark.parametrize("declared", [False, True])
def test_normal_import_of_external_helper_requires_exact_source_pin(tmp_path, monkeypatch, declared):
    import sys
    # This is an ordinary external source file and a genuine normal import.
    # Adding its directory to sys.path cannot give an unpinned helper authority.
    external = tmp_path / "external"; external.mkdir()
    name = "repair_ordinary_external_helper_fixture"
    source = external / (name + ".py")
    source.write_text("loaded_from_raw_external_source = True\n")
    pin = worker.descriptor(source)["sha256"]
    guard = worker.PinnedSources({str(source): pin} if declared else {},
        [tmp_path / "candidate", tmp_path / "datasets", tmp_path / "trusted"])
    monkeypatch.syspath_prepend(str(external))
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    assert name not in sys.modules
    try:
        if declared:
            actual = importlib.import_module(name)
            assert actual.loaded_from_raw_external_source is True
            assert guard.executed == {str(source): pin} and guard.denials == []
            assert Path(actual.__file__) == source
        else:
            with pytest.raises(ImportError):
                importlib.import_module(name)
            assert name not in sys.modules and guard.executed == {}
            assert guard.denials == [{"path": str(source),
                "reason": "undeclared source origin outside authenticated interpreter roots"}]
            with pytest.raises(ValueError):
                worker.need(not guard.denials, "caught external import remains a permanent denial")
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("failure", ["missing", "invalid", "changed"])
def test_normal_external_import_resolution_failure_is_permanent(tmp_path, monkeypatch, failure):
    import sys
    external = tmp_path / "unreadable-external"; external.mkdir()
    name = "repair_unreadable_external_helper_fixture"
    source = external / (name + ".py")
    source.write_text("raise AssertionError('unproved external body must never execute')\n")
    guard = worker.PinnedSources({}, [tmp_path / "candidate", tmp_path / "datasets", tmp_path / "trusted"])
    original_resolve = Path.resolve
    def changed_origin(path, *args, **kwargs):
        if path == source:
            error = {"missing": OSError, "invalid": ValueError, "changed": RuntimeError}[failure]
            raise error("private fixture origin cannot be authenticated")
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", changed_origin)
    monkeypatch.syspath_prepend(str(external))
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    assert name not in sys.modules
    try:
        with pytest.raises(ImportError):
            importlib.import_module(name)
        assert name not in sys.modules and guard.executed == {}
        assert guard.denials == [{"path": str(source), "reason": "source origin could not be authenticated"}]
        with pytest.raises(ValueError):
            worker.need(not guard.denials, "caught origin failure remains a permanent denial")
    finally:
        sys.modules.pop(name, None)
