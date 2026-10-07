"""Read-only STOP observation of authored private cleanup records.

No Docker or provider runs. The producer builds real canonical records and
resource retirement; independent Docker/process observations remain doubles.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleProfile, ProcessIdentity, ProcessTreeSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as producer
from ipfs_accelerate_py.agent_supervisor.runtime import durable_cleanup_observer as observation


def _filesystem(root):
    result = {}
    for path in sorted(root.rglob("*")):
        meta = path.lstat()
        result[str(path.relative_to(root))] = (
            meta.st_dev, meta.st_ino, meta.st_mode, meta.st_nlink,
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
    return result


@pytest.fixture
def cleanup(tmp_path, monkeypatch):
    state = tmp_path / "state"
    run = state / "run"
    run.mkdir(parents=True, mode=0o700)
    profile = LifecycleProfile(
        target_id="supervisor-track:private-cleanup-observer", run_id="private-cleanup-run",
        configuration_root="private-cleanup-configuration", repository_root=str(tmp_path),
        state_root=str(state), run_root=str(run), argv=(sys.executable, "owned-supervisor.py"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(7).items():
        if name in producer._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
            monkeypatch.setenv(name, value)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    lease = Path(tempfile.mkdtemp(prefix="asref-codex-container-"))
    home = Path(tempfile.mkdtemp(prefix="asref-codex-home-"))
    fd, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(fd)
    prompt = Path(prompt_name)
    config = lease / "docker-config"
    config.mkdir(mode=0o700)
    name = "ipfs-accelerate-codex-900001-" + "a" * 32
    anchor = observation.CleanupDirectoryAnchor.open_before_launch(run)
    binding_path = producer._docker_cleanup_binding_path(name)
    paths = {"lease_root": lease, "provider_home": home, "prompt_path": prompt,
             "docker_config": config}
    record = producer._docker_cleanup_binding_value(
        binding_state="prepared_no_dispatch", provider="codex", docker_bin="/usr/bin/docker",
        container_name=name, lease_root=lease, docker_config=config,
        cidfile=lease / "container.cid", provider_home=home, prompt_path=prompt,
        effect_observation={}, path_identities={key: producer._cleanup_path_identity(
            value, directory=key != "prompt_path") for key, value in paths.items()},
        binding_path=binding_path, runner_pid=900001, runner_start_ticks=100001,
        watchdog_pid=900002, watchdog_start_ticks=100002,
    )
    producer._write_private_control_record(binding_path.parent, binding_path.name,
                                          record, replace_existing=False)

    def identity(pid, start, parent, argv):
        return ProcessIdentity(
            pid=pid, start_time_ticks=start, parent_pid=parent, process_group_id=pid,
            session_id=pid, boot_id=record["boot_id"], argv=argv, cwd=profile.cwd,
            executable=str(Path(sys.executable).resolve()), run_id=profile.run_id, profile_id=profile.profile_id,
            target_id=profile.target_id, repository_root=profile.repository_root,
            state_root=profile.state_root, run_root=profile.run_root, fencing_epoch=7,
            configuration_root=profile.configuration_root,
        )

    runner = identity(900001, 100001, 1, profile.argv)
    watchdog_arguments = [
        sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--internal-docker-cleanup-watchdog-launcher", "--internal-docker-cleanup-watchdog",
    ]
    for flag, field in (
        ("--provider", "provider"), ("--docker-bin", "docker_bin"),
        ("--container-name", "container_name"), ("--lease-root", "lease_root"),
        ("--cidfile", "cidfile"), ("--provider-home", "provider_home"),
        ("--prompt-path", "prompt_path"), ("--cleanup-binding-record", "binding_path"),
        ("--runner-pid", "runner_pid"), ("--runner-start-ticks", "runner_start_ticks"),
    ):
        watchdog_arguments.extend((flag, str(record[field])))
    watchdog = identity(900002, 100002, 1, tuple(watchdog_arguments))
    tree = ProcessTreeSnapshot(profile_id=profile.profile_id, run_id=profile.run_id,
                               members=(runner, watchdog))
    observer = observation.ManagedCleanupObserver(profile, anchor, fencing_epoch=7)
    try:
        yield SimpleNamespace(profile=profile, anchor=anchor, observer=observer,
                              record=record, path=binding_path, tree=tree,
                              watchdog=watchdog, runner=runner, private=private, paths=paths)
    finally:
        anchor.close()


def test_empty_missing_cleanup_custody_probe_is_read_only(tmp_path):
    run = tmp_path / "missing-run"
    before = _filesystem(tmp_path)
    assert observation.has_cleanup_custody(run) is False
    assert _filesystem(tmp_path) == before


def test_observer_retains_only_exact_private_watchdog_birth(cleanup, tmp_path):
    before = _filesystem(tmp_path)
    protected = cleanup.observer.observe(cleanup.tree)
    assert protected == frozenset({(cleanup.watchdog.pid, cleanup.watchdog.start_time_ticks,
                                   cleanup.watchdog.boot_id)})
    assert _filesystem(tmp_path) == before


def test_active_prepared_record_is_not_cleanup_completion(cleanup, tmp_path):
    cleanup.observer.observe(cleanup.tree)
    before = _filesystem(tmp_path)
    assert cleanup.observer.complete(deadline=time.monotonic() + .1) is False
    assert _filesystem(tmp_path) == before


@pytest.mark.parametrize("mutation", ["deleted", "symlink", "foreign_lifecycle", "checksum"])
def test_changed_binding_never_converts_process_absence_to_completion(cleanup, tmp_path, mutation):
    cleanup.observer.observe(cleanup.tree)
    if mutation == "deleted":
        cleanup.path.unlink()
    elif mutation == "symlink":
        target = tmp_path / "unrelated-private-record"
        target.write_bytes(cleanup.path.read_bytes())
        target.chmod(0o600)
        cleanup.path.unlink()
        cleanup.path.symlink_to(target)
    else:
        value = copy.deepcopy(cleanup.record)
        if mutation == "foreign_lifecycle":
            value["run_id"] = "unrelated-run"
            value.pop("record_id")
            value["record_id"] = producer._effect_receipt_identity(value)
        else:
            value["record_id"] = "sha256:" + "0" * 64
        producer._write_private_control_record(cleanup.path.parent, cleanup.path.name,
                                              value, replace_existing=True)
    before = _filesystem(tmp_path)
    assert cleanup.observer.complete(deadline=time.monotonic() + .1) is False
    assert _filesystem(tmp_path) == before


def test_replaced_cleanup_directory_cannot_be_readmitted(cleanup, tmp_path):
    cleanup.observer.observe(cleanup.tree)
    directory = cleanup.path.parent
    retained = directory.with_name("retained-original-bindings")
    directory.rename(retained)
    directory.mkdir(mode=0o700)
    before = _filesystem(tmp_path)
    assert cleanup.observer.complete(deadline=time.monotonic() + .1) is False
    assert _filesystem(tmp_path) == before


def test_cleanup_anchor_rejects_symlink_before_launch(tmp_path):
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    target = tmp_path / "unrelated"
    target.mkdir(mode=0o700)
    (run / producer._DOCKER_CLEANUP_BINDING_DIRECTORY).symlink_to(target, target_is_directory=True)
    before = _filesystem(tmp_path)
    with pytest.raises((OSError, ValueError)):
        observation.CleanupDirectoryAnchor.open_before_launch(run)
    assert _filesystem(tmp_path) == before


def _finish_prepared(cleanup):
    identity = producer._cleanup_path_identity(cleanup.path, directory=False)
    # The fixture explicitly supplies the producer's absence precondition.
    # This finalizer performs real private-file retirement, no Docker effect.
    assert producer._finalize_verified_cleanup_completion(
        binding_path=cleanup.path, binding_identity=identity,
        binding_record=cleanup.record,
    )
    assert not cleanup.path.exists()
    assert cleanup.path.with_suffix(".authority").is_file()
    assert cleanup.path.with_suffix(".complete").is_file()


def test_completed_protocol_requires_fresh_read_only_absence(cleanup, tmp_path, monkeypatch):
    cleanup.observer.observe(cleanup.tree)
    _finish_prepared(cleanup)
    calls = []
    deadline = time.monotonic() + .5

    def independently_observe(**kwargs):
        assert kwargs["issue_removal"] is False
        assert kwargs["settle_for_creation"] is False
        assert kwargs["deadline"] == deadline
        assert kwargs["container_name"] == cleanup.record["container_name"]
        assert kwargs["pass_fds"] == (cleanup.anchor.descriptor,)
        calls.append(kwargs)

    monkeypatch.setattr(producer, "_remove_exact_docker_container", independently_observe)
    before = _filesystem(tmp_path)
    assert cleanup.observer.complete(deadline=deadline) is True
    assert len(calls) == 1
    assert _filesystem(tmp_path) == before


def test_completed_protocol_unknown_docker_absence_is_incomplete(cleanup, tmp_path, monkeypatch):
    _finish_prepared(cleanup)
    calls = []

    def unavailable(**kwargs):
        assert kwargs["issue_removal"] is False
        calls.append(kwargs)
        raise ValueError("Docker engine unavailable")

    monkeypatch.setattr(producer, "_remove_exact_docker_container", unavailable)
    before = _filesystem(tmp_path)
    assert cleanup.observer.complete(deadline=time.monotonic() + .5) is False
    assert len(calls) == 1
    assert _filesystem(tmp_path) == before


def test_new_observer_recovers_completed_record_after_all_markers_disappear(cleanup, monkeypatch):
    _finish_prepared(cleanup)
    recovered = observation.ManagedCleanupObserver(cleanup.profile, cleanup.anchor, fencing_epoch=7)
    empty = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id)
    assert recovered.observe(empty) == frozenset()
    monkeypatch.setattr(producer, "_remove_exact_docker_container", lambda **kwargs: None)
    assert recovered.complete(deadline=time.monotonic() + .5) is True


@pytest.mark.parametrize("mutation", ["authority_inode", "completion", "resource_reappeared"])
def test_completed_protocol_rejects_later_drift_before_docker_observation(cleanup, monkeypatch, mutation):
    cleanup.observer.observe(cleanup.tree)
    _finish_prepared(cleanup)
    if mutation == "authority_inode":
        authority = cleanup.path.with_suffix(".authority")
        raw = authority.read_bytes()
        authority.rename(authority.with_suffix(".retained-original"))
        authority.write_bytes(raw)
        authority.chmod(0o600)
        # Keep the retained unrelated inode outside the protocol namespace.
        authority.with_suffix(".retained-original").rename(cleanup.private / "retained-original")
    elif mutation == "completion":
        path = cleanup.path.with_suffix(".complete")
        value = producer._read_private_control_record(path.parent, path.name)
        value["docker_absence"]["observation"] = "forged-absence"
        value.pop("completion_id")
        value["completion_id"] = producer._effect_receipt_identity(value)
        producer._write_private_control_record(path.parent, path.name, value, replace_existing=True)
    else:
        cleanup.paths["provider_home"].mkdir(mode=0o700)
    monkeypatch.setattr(producer, "_remove_exact_docker_container", lambda **kwargs: pytest.fail(
        "invalid completion must not reach Docker observation"))
    assert cleanup.observer.complete(deadline=time.monotonic() + .5) is False


def test_expired_stop_deadline_never_launches_docker_observation(cleanup, monkeypatch):
    _finish_prepared(cleanup)
    monkeypatch.setattr(producer, "_remove_exact_docker_container", lambda **kwargs: pytest.fail(
        "expired deadline must not start Docker observation"))
    assert cleanup.observer.complete(deadline=time.monotonic() - 1) is False


def test_reused_watchdog_pid_fails_exact_private_record_join(cleanup):
    from dataclasses import replace
    reused = replace(cleanup.watchdog, start_time_ticks=cleanup.watchdog.start_time_ticks + 1,
                     identity_id="")
    tree = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id,
                               members=(cleanup.runner, reused))
    with pytest.raises(ValueError, match="exact durable birth"):
        cleanup.observer.observe(tree)


def test_observer_retains_observed_cleanup_child_after_watchdog_exit(cleanup):
    from dataclasses import replace
    child = replace(cleanup.runner, pid=900003, start_time_ticks=100003,
                    parent_pid=cleanup.watchdog.pid, identity_id="")
    tree = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id,
                               members=(*cleanup.tree.members, child))
    protected = cleanup.observer.observe(tree)
    assert (child.pid, child.start_time_ticks, child.boot_id) in protected
    reparented = replace(child, parent_pid=1, identity_id="")
    later = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id,
                                members=(reparented,))
    assert cleanup.observer.observe(later) == protected


@pytest.mark.parametrize("first_record", ["known_prepared", "malformed"])
def test_second_stop_cannot_forget_failed_cleanup_after_record_disappears(
    cleanup, tmp_path, monkeypatch, first_record,
):
    from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as owner

    if first_record == "malformed":
        cleanup.path.write_bytes(b'{"invalid":"cleanup"}\n')
    empty = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id)
    process = SimpleNamespace(
        pid=cleanup.runner.pid, poll=lambda: 137,
        _agent_supervisor_lifecycle_profile=cleanup.profile,
        _agent_supervisor_process_identity=cleanup.runner,
        _agent_supervisor_cleanup_directory_anchor=cleanup.anchor,
        _agent_supervisor_fencing_epoch=7,
    )
    adapter = SimpleNamespace(snapshot=lambda profile: empty,
                              terminate=lambda *args, **kwargs: pytest.fail("no process remains to signal"))
    monkeypatch.setattr(owner, "LinuxProcessAdapter", lambda: adapter)
    before = _filesystem(tmp_path)
    try:
        fenced, _ = owner._terminate_managed_process(process, grace_seconds=0)
    except (ValueError, owner.ProcessIdentityMismatch):
        fenced = False
    assert fenced is False
    assert _filesystem(tmp_path) == before
    cleanup.path.unlink()
    after = _filesystem(tmp_path)
    try:
        fenced_again, _ = owner._terminate_managed_process(process, grace_seconds=0)
    except (ValueError, owner.ProcessIdentityMismatch):
        fenced_again = False
    assert fenced_again is False
    assert _filesystem(tmp_path) == after


@pytest.fixture
def signed_cleanup(tmp_path, monkeypatch):
    from test.api import test_terminal_cleanup_observer as current

    monkeypatch.setattr(current.local_profile_module, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
                        tmp_path / "isolated-lifecycle-registry")
    repository, _key, route, invocation = current._reviewed_route(tmp_path)
    original_publish = current._publish_prepared
    selected = {}

    def publish_with_current_owner(case):
        run = case.binding_path.parent.parent
        profile = LifecycleProfile(
            target_id="supervisor-track:signed-cleanup", run_id="signed-cleanup-run",
            configuration_root="signed-cleanup-configuration", repository_root=str(repository),
            state_root=str(run.parent), run_root=str(run),
            argv=(sys.executable, "owned-supervisor.py"), cwd=str(repository),
        )
        for name, value in profile.launch_environment(1).items():
            if name in producer._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
                monkeypatch.setenv(name, value)
        selected["anchor"] = observation.CleanupDirectoryAnchor.open_before_launch(run)
        selected["profile"] = profile
        return original_publish(case)

    monkeypatch.setattr(current, "_publish_prepared", publish_with_current_owner)
    case = current._native_cleanup(tmp_path, monkeypatch, repository=repository,
                                   route=route, invocation=invocation, complete=True)
    anchor = selected["anchor"]
    try:
        observer = observation.ManagedCleanupObserver(selected["profile"], anchor, fencing_epoch=1)
        yield SimpleNamespace(case=case, observer=observer, anchor=anchor, profile=selected["profile"])
    finally:
        anchor.close()


@pytest.mark.parametrize("store_state", ["complete", "missing", "replaced"])
def test_current_signed_terminal_cas_must_join_read_only_owner_completion(
    signed_cleanup, tmp_path, monkeypatch, store_state,
):
    case, observer = signed_cleanup.case, signed_cleanup.observer
    store = case.store.directory
    if store_state != "complete":
        store.rename(store.with_name("retained-original-attempts"))
        if store_state == "replaced":
            store.mkdir(mode=0o700)
    observed = []

    def absence(**kwargs):
        assert kwargs["issue_removal"] is False
        assert kwargs["termination_fence"]
        observed.append(kwargs)

    monkeypatch.setattr(producer, "_remove_exact_docker_container", absence)
    before = _filesystem(tmp_path)
    assert observer.complete(deadline=time.monotonic() + .5) is (store_state == "complete")
    assert len(observed) == (1 if store_state == "complete" else 0)
    assert _filesystem(tmp_path) == before


@pytest.mark.parametrize("field", ["container_id", "image_id", "cidfile", "provider_home", "prompt_path"])
def test_terminal_join_rejects_different_fence_or_resource_claim(signed_cleanup, tmp_path, field):
    observer = signed_cleanup.observer
    path = signed_cleanup.case.binding_path
    record = producer._read_private_control_record(path.parent, path.with_suffix(".authority").name)
    completion = producer._read_private_control_record(path.parent, path.with_suffix(".complete").name)
    changed = copy.deepcopy(record)
    if field in {"container_id", "image_id"}:
        changed["termination_fence"][field] = ("sha256:" if field == "image_id" else "") + "f" * 64
    else:
        changed[field] = str(tmp_path / ("unrelated-" + field))
    before = _filesystem(tmp_path)
    with pytest.raises(ValueError, match="does not join exact completion"):
        observer._terminal(changed, completion)
    assert _filesystem(tmp_path) == before


@pytest.mark.parametrize("issuer_state", ["exact", "reused_pid", "missing_journal", "foreign_binding"])
def test_detached_removal_issuer_requires_exact_dispatch_birth(cleanup, issuer_state):
    from dataclasses import replace
    from test.api.test_terminal_cleanup_observer import _created_docker_termination_fence

    record = copy.deepcopy(cleanup.record)
    fence = _created_docker_termination_fence(
        container_name=record["container_name"], container_id="6" * 64,
        image_id=producer._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
    )
    record.update(binding_state="command_bound", create_command_id="sha256:" + "1" * 64,
                  create_cwd=str(cleanup.private), create_environment_id="sha256:" + "2" * 64,
                  termination_fence=fence)
    record.pop("record_id")
    record["record_id"] = producer._effect_receipt_identity(record)
    producer._write_private_control_record(cleanup.path.parent, cleanup.path.name,
                                          record, replace_existing=True)
    issuer = replace(cleanup.runner, pid=900004, start_time_ticks=100004, parent_pid=1,
                     argv=(sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
                           "--internal-docker-removal-issuer-launcher", "--internal-docker-removal-issuer",
                           "--binding-path", str(cleanup.path)), identity_id="")
    dispatch = producer._docker_removal_dispatch_value(
        binding_path=cleanup.path, binding_record=record, termination_fence=fence,
        issuer_process_birth={"pid": issuer.pid, "start_time_ticks": issuer.start_time_ticks,
                              "boot_id": issuer.boot_id, "parent_pid": 1},
        state="prepared", generation=1, previous_dispatch_id="", docker_returncode=None,
        failure_kind="",
    )
    if issuer_state != "missing_journal":
        dispatch_path = cleanup.path.with_suffix(".remove-dispatched")
        producer._write_private_control_record(dispatch_path.parent, dispatch_path.name,
                                              dispatch, replace_existing=False)
    if issuer_state == "reused_pid":
        issuer = replace(issuer, start_time_ticks=issuer.start_time_ticks + 1, identity_id="")
    elif issuer_state == "foreign_binding":
        issuer = replace(issuer, argv=(*issuer.argv[:-1], str(cleanup.path.with_name("f" * 64 + ".json"))),
                         identity_id="")
    tree = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id,
                               members=(issuer,))
    if issuer_state == "exact":
        assert cleanup.observer.observe(tree) == frozenset({(issuer.pid, issuer.start_time_ticks, issuer.boot_id)})
    else:
        with pytest.raises(ValueError):
            cleanup.observer.observe(tree)


def test_prior_cleanup_custody_prevents_next_process_birth(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as owner

    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    state = tmp_path / "state"
    run = state / "lifecycle-runs/owned"
    directory = run / producer._DOCKER_CLEANUP_BINDING_DIRECTORY
    directory.mkdir(parents=True, mode=0o700)
    retained = directory / ("a" * 64 + ".json")
    retained.write_bytes(b'{"unresolved":"prior custody"}\n')
    retained.chmod(0o600)
    track = owner.SupervisorTrack(name="owned", script_path=script, log_path=tmp_path / "worker.log",
                                  supervisor_pid_path=state / "worker.pid",
                                  daemon_pid_path=state / "daemon.pid")
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail(
        "prior cleanup custody must prevent process birth"))
    with pytest.raises(ValueError, match="unresolved prior cleanup"):
        owner.start_track(track, repo_root=tmp_path, common_args=(), output=lambda _: None)
    assert retained.read_bytes() == b'{"unresolved":"prior custody"}\n'


def test_cleanup_anchor_close_is_idempotent_and_does_not_close_reused_descriptor(tmp_path):
    anchor = observation.CleanupDirectoryAnchor.open_before_launch(tmp_path / "run")
    old_fd = anchor.descriptor
    anchor.close()
    # Obtain a fresh descriptor at the just-retired numeric fd to model reuse.
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert descriptor == old_fd
        anchor.close()
        assert os.fstat(descriptor).st_ino == tmp_path.stat().st_ino
    finally:
        os.close(descriptor)


def test_command_bound_name_only_completion_never_certifies_issued_effect(cleanup, monkeypatch):
    record = copy.deepcopy(cleanup.record)
    record.update(binding_state="command_bound", create_command_id="sha256:" + "3" * 64,
                  create_cwd=str(cleanup.private), create_environment_id="sha256:" + "4" * 64)
    record.pop("record_id")
    record["record_id"] = producer._effect_receipt_identity(record)
    producer._write_private_control_record(cleanup.path.parent, cleanup.path.name,
                                          record, replace_existing=True)
    identity = producer._cleanup_path_identity(cleanup.path, directory=False)
    assert producer._finalize_verified_cleanup_completion(
        binding_path=cleanup.path, binding_identity=identity, binding_record=record)
    monkeypatch.setattr(producer, "_remove_exact_docker_container", lambda **kwargs: pytest.fail(
        "an issued effect without exact fence cannot use name-only absence"))
    assert cleanup.observer.complete(deadline=time.monotonic() + .5) is False


def test_final_empty_process_snapshot_cannot_hide_last_cleanup_publication(
    cleanup, tmp_path, monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as owner

    retained = cleanup.private / "not-yet-published-binding"
    cleanup.path.rename(retained)
    empty = ProcessTreeSnapshot(profile_id=cleanup.profile.profile_id, run_id=cleanup.profile.run_id)
    process = SimpleNamespace(
        pid=cleanup.runner.pid, poll=lambda: 137,
        _agent_supervisor_lifecycle_profile=cleanup.profile,
        _agent_supervisor_process_identity=cleanup.runner,
        _agent_supervisor_cleanup_directory_anchor=cleanup.anchor,
        _agent_supervisor_fencing_epoch=7,
    )
    completions = []
    original_complete = observation.ManagedCleanupObserver.complete

    def complete(observer, *, deadline):
        result = original_complete(observer, deadline=deadline)
        completions.append((deadline, result))
        return result

    snapshots = []
    published_filesystem = []

    def snapshot(profile):
        assert profile is cleanup.profile
        snapshots.append(True)
        if len(snapshots) == 3:
            # The exiting descendant publishes its valid durable binding
            # after the owner's first completion check, then disappears from
            # the final process snapshot. The record is unresolved.
            assert completions == [(completions[0][0], True)]
            retained.rename(cleanup.path)
            published_filesystem.append(_filesystem(tmp_path))
        return empty

    monkeypatch.setattr(observation.ManagedCleanupObserver, "complete", complete)
    monkeypatch.setattr(owner, "LinuxProcessAdapter", lambda: SimpleNamespace(
        snapshot=snapshot,
        terminate=lambda *args, **kwargs: pytest.fail("no authored process remains to signal"),
    ))
    monkeypatch.setattr(producer, "_remove_exact_docker_container", lambda **kwargs: pytest.fail(
        "an unfinished binding cannot reach Docker absence observation"))

    fenced, _ = owner._terminate_managed_process(process, grace_seconds=0)

    assert fenced is False
    assert len(completions) >= 2
    assert completions[0][1] is True
    assert all(result is False for _, result in completions[1:])
    assert len({deadline for deadline, _ in completions}) == 1
    assert len(published_filesystem) == 1
    assert _filesystem(tmp_path) == published_filesystem[0]
