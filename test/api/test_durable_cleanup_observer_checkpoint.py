"""Read-only cleanup checkpoint tests with the canonical lifecycle/producer.

All records and inode retirement are real private fixture files. Docker
absence is an explicit authored double. No process, Docker or provider runs.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleProfile, ProcessIdentity, ProcessTreeSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.runtime import durable_cleanup_observer as observation
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as producer


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("observer checkpoint tests must not start Docker, providers or subprocesses")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    # Current finalizer rechecks Docker before retiring an inode even for a
    # prepared, never-dispatched record. That precondition is authored here;
    # tests of the owner's fresh observation replace only the guarded reader.
    monkeypatch.setattr(producer, "_reobserve_cleanup_binding_docker_absence", lambda _: True)
    monkeypatch.setattr(producer, "_remove_exact_docker_container",
                        lambda **_: pytest.fail("undeclared Docker observation is forbidden"))


@pytest.fixture(params=("grok", "codex"))
def retained(tmp_path, monkeypatch, request):
    provider = request.param
    state = tmp_path / "state"
    run = state / "run"
    run.mkdir(parents=True, mode=0o700)
    profile = LifecycleProfile(
        target_id="supervisor-track:observer-checkpoint", run_id="observer-checkpoint-run",
        configuration_root="observer-checkpoint-configuration", repository_root=str(tmp_path),
        state_root=str(state), run_root=str(run), argv=(sys.executable, "owned-supervisor.py"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(7).items():
        if name in producer._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
            monkeypatch.setenv(name, value)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    lease = Path(tempfile.mkdtemp(prefix=f"asref-{provider}-container-"))
    home = Path(tempfile.mkdtemp(prefix=f"asref-{provider}-home-"))
    descriptor, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(descriptor)
    prompt = Path(prompt_name)
    config = lease / "docker-config"
    config.mkdir(mode=0o700)
    anchor = observation.CleanupDirectoryAnchor.open_before_launch(run)
    name = f"ipfs-accelerate-{provider}-900001-" + "a" * 32
    path = producer._docker_cleanup_binding_path(name)
    paths = {"lease_root": lease, "provider_home": home, "prompt_path": prompt,
             "docker_config": config}
    record = producer._docker_cleanup_binding_value(
        binding_state="prepared_no_dispatch", provider=provider, docker_bin="/usr/bin/docker",
        container_name=name, lease_root=lease, docker_config=config, cidfile=lease / "container.cid",
        provider_home=home, prompt_path=prompt, effect_observation={},
        path_identities={key: producer._cleanup_path_identity(value, directory=key != "prompt_path")
                         for key, value in paths.items()},
        binding_path=path, runner_pid=900001, runner_start_ticks=100001,
        watchdog_pid=900002, watchdog_start_ticks=100002,
    )
    producer._write_private_control_record(path.parent, path.name, record, replace_existing=False)

    def identity(pid, start, parent, argv):
        return ProcessIdentity(
            pid=pid, start_time_ticks=start, parent_pid=parent, process_group_id=pid,
            session_id=pid, boot_id=record["boot_id"], argv=argv, cwd=profile.cwd,
            executable=str(Path(sys.executable).resolve()), run_id=profile.run_id,
            profile_id=profile.profile_id, target_id=profile.target_id,
            repository_root=profile.repository_root, state_root=profile.state_root,
            run_root=profile.run_root, fencing_epoch=7, configuration_root=profile.configuration_root,
        )

    runner = identity(900001, 100001, 1, profile.argv)
    watchdog_argv = [sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
                     "--internal-docker-cleanup-watchdog-launcher", "--internal-docker-cleanup-watchdog"]
    for flag, field in (
        ("--provider", "provider"), ("--docker-bin", "docker_bin"),
        ("--container-name", "container_name"), ("--lease-root", "lease_root"),
        ("--cidfile", "cidfile"), ("--provider-home", "provider_home"),
        ("--prompt-path", "prompt_path"), ("--cleanup-binding-record", "binding_path"),
        ("--runner-pid", "runner_pid"), ("--runner-start-ticks", "runner_start_ticks"),
    ):
        watchdog_argv.extend((flag, str(record[field])))
    watchdog = identity(900002, 100002, 1, tuple(watchdog_argv))
    tree = ProcessTreeSnapshot(profile_id=profile.profile_id, run_id=profile.run_id,
                               members=(runner, watchdog))
    observer = observation.ManagedCleanupObserver(profile, anchor, fencing_epoch=7)
    observer.observe(tree)
    value = SimpleNamespace(profile=profile, anchor=anchor, observer=observer, record=record,
                            path=path, tree=tree, watchdog=watchdog, runner=runner,
                            private=private, paths=paths)
    try:
        yield value
    finally:
        anchor.close()


def _restore(retained, checkpoint=None):
    return observation.ManagedCleanupObserver.restore_checkpoint(
        retained.profile, retained.observer.checkpoint() if checkpoint is None else checkpoint,
        fencing_epoch=7,
    )


def _finish(retained):
    assert producer._finalize_verified_cleanup_completion(
        binding_path=retained.path,
        binding_identity=producer._cleanup_path_identity(retained.path, directory=False),
        binding_record=retained.record,
    )


def _files(root):
    return {str(path.relative_to(root)): (path.lstat().st_ino, path.lstat().st_mode,
            path.read_bytes() if path.is_file() and not path.is_symlink() else None)
            for path in root.rglob('*')}


def test_checkpoint_restore_retains_exact_directory_records_and_protected_births(retained):
    checkpoint = retained.observer.checkpoint()
    recovered = _restore(retained, checkpoint)
    try:
        assert recovered.anchor is not retained.anchor
        assert recovered.anchor.descriptor != retained.anchor.descriptor
        assert recovered.anchor.identity == retained.anchor.identity
        assert recovered.records == retained.observer.records
        assert recovered.identities == retained.observer.identities
        assert recovered.protected == retained.observer.protected
        assert recovered.checkpoint() == checkpoint
        assert recovered.failed is False
    finally:
        recovered.anchor.close()


def test_checkpoint_and_restore_are_deep_copies(retained):
    checkpoint = retained.observer.checkpoint()
    checkpoint['records'][retained.path.stem]['path_identities']['lease_root']['inode'] += 1
    assert checkpoint != retained.observer.checkpoint()
    clean = retained.observer.checkpoint()
    recovered = _restore(retained, clean)
    try:
        clean['records'][retained.path.stem]['provider'] = 'altered'
        clean['identities'][retained.path.stem]['inode'] += 1
        clean['protected_births'].clear()
        assert recovered.checkpoint() == retained.observer.checkpoint()
    finally:
        recovered.anchor.close()


@pytest.mark.parametrize('replacement', ('missing', 'fresh', 'symlink'))
def test_recovery_never_creates_or_adopts_missing_replaced_or_aliased_directory(retained, replacement):
    directory = retained.anchor.path
    original = directory.with_name('retained-original-cleanup')
    directory.rename(original)
    if replacement == 'fresh':
        directory.mkdir(mode=0o700)
    elif replacement == 'symlink':
        directory.symlink_to(original, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        _restore(retained)
    if replacement == 'missing':
        assert not directory.exists()


def test_known_binding_deletion_cannot_become_empty_success_after_restore(retained):
    checkpoint = retained.observer.checkpoint()
    retained.path.unlink()
    recovered = _restore(retained, checkpoint)
    try:
        assert not recovered.complete(deadline=time.monotonic() + .2)
        assert recovered.failed
        assert retained.path.stem in recovered.records
    finally:
        recovered.anchor.close()


def test_unknown_failure_latch_survives_recovery_even_after_exact_record_returns(retained):
    retained.path.unlink()
    assert not retained.observer.complete(deadline=time.monotonic() + .2)
    failed = retained.observer.checkpoint()
    producer._write_private_control_record(retained.path.parent, retained.path.name,
                                          retained.record, replace_existing=False)
    recovered = _restore(retained, failed)
    try:
        assert recovered.failed
        assert not recovered.complete(deadline=time.monotonic() + .2)
        assert recovered.checkpoint()['failed'] is True
    finally:
        recovered.anchor.close()


@pytest.mark.parametrize('mutation', ('symlink', 'inode', 'foreign_record'))
def test_current_binding_drift_remains_unknown_after_checkpoint_restore(retained, mutation):
    checkpoint = retained.observer.checkpoint()
    old = retained.private / 'original-binding'
    retained.path.rename(old)
    if mutation == 'symlink':
        retained.path.symlink_to(old)
    elif mutation == 'inode':
        retained.path.write_bytes(old.read_bytes())
        retained.path.chmod(0o600)
    else:
        changed = copy.deepcopy(retained.record)
        changed['run_id'] = 'foreign-run'
        changed.pop('record_id')
        changed['record_id'] = producer._effect_receipt_identity(changed)
        producer._write_private_control_record(retained.path.parent, retained.path.name,
                                              changed, replace_existing=False)
    recovered = _restore(retained, checkpoint)
    try:
        assert not recovered.complete(deadline=time.monotonic() + .2)
        assert recovered.failed
    finally:
        recovered.anchor.close()


@pytest.mark.parametrize('mutation', ('boot', 'profile', 'fence', 'record_checksum', 'identity_uid',
                                     'identity_kind', 'duplicate_birth', 'foreign_birth_boot', 'extra_field'))
def test_malformed_or_foreign_checkpoint_cannot_rebind_custody(retained, mutation):
    checkpoint = retained.observer.checkpoint()
    if mutation == 'boot':
        checkpoint['boot_id'] = 'foreign-boot'
        for birth in checkpoint['protected_births']:
            birth[2] = 'foreign-boot'
    elif mutation == 'profile':
        checkpoint['profile_id'] = 'foreign-profile'
    elif mutation == 'fence':
        checkpoint['fencing_epoch'] += 1
    elif mutation == 'record_checksum':
        checkpoint['records'][retained.path.stem]['record_id'] = 'sha256:' + '0' * 64
    elif mutation == 'identity_uid':
        checkpoint['identities'][retained.path.stem]['uid'] += 1
    elif mutation == 'identity_kind':
        checkpoint['identities'][retained.path.stem]['mode'] = stat.S_IFDIR
    elif mutation == 'duplicate_birth':
        checkpoint['protected_births'] *= 2
    elif mutation == 'foreign_birth_boot':
        checkpoint['protected_births'][0][2] = 'foreign-boot'
    else:
        checkpoint['dispatch_authorized'] = True
    with pytest.raises((OSError, ValueError)):
        _restore(retained, checkpoint)


def test_protected_child_births_roundtrip_and_pid_reuse_does_not_inherit_them(retained):
    child = replace(retained.watchdog, pid=900010, start_time_ticks=100010,
                    parent_pid=retained.watchdog.pid, argv=('/usr/bin/docker', 'inspect'), identity_id='')
    tree = replace(retained.tree, members=(*retained.tree.members, child), tree_id='')
    protected = retained.observer.observe(tree)
    recovered = _restore(retained)
    try:
        assert (child.pid, child.start_time_ticks, child.boot_id) in protected
        assert recovered.protected == set(protected)
        reused = replace(child, start_time_ticks=child.start_time_ticks + 1,
                         parent_pid=1, argv=('unrelated-fixture',), identity_id='')
        recovered_tree = replace(retained.tree, members=(reused,), tree_id='')
        still_protected = recovered.observe(recovered_tree)
        assert (reused.pid, reused.start_time_ticks, reused.boot_id) not in still_protected
        assert (child.pid, child.start_time_ticks, child.boot_id) in still_protected
    finally:
        recovered.anchor.close()


def test_legacy_prepared_observer_is_read_only_and_incomplete(retained, tmp_path):
    before = _files(tmp_path)
    assert not retained.observer.complete(deadline=time.monotonic() + .2)
    assert _files(tmp_path) == before
    assert retained.observer.failed is False


def test_completed_checkpoint_requires_fresh_independent_read_only_absence(retained, monkeypatch, tmp_path):
    checkpoint = retained.observer.checkpoint()
    _finish(retained)
    recovered = _restore(retained, checkpoint)
    calls = []
    deadline = time.monotonic() + .5

    def absence(**kwargs):
        assert kwargs['issue_removal'] is False
        assert kwargs['settle_for_creation'] is False
        assert kwargs['deadline'] == deadline
        assert kwargs['container_name'] == retained.record['container_name']
        assert kwargs['pass_fds'] == (recovered.anchor.descriptor,)
        calls.append(kwargs)

    monkeypatch.setattr(producer, '_remove_exact_docker_container', absence)
    try:
        before = _files(tmp_path)
        assert recovered.complete(deadline=deadline)
        assert len(calls) == 1
        assert _files(tmp_path) == before
    finally:
        recovered.anchor.close()


def test_unknown_docker_observation_is_incomplete_and_can_be_retried(retained, monkeypatch):
    _finish(retained)
    recovered = _restore(retained)
    def unavailable(**kwargs):
        assert kwargs['issue_removal'] is False
        raise ValueError('authored Docker observation unavailable')
    monkeypatch.setattr(producer, '_remove_exact_docker_container', unavailable)
    try:
        assert not recovered.complete(deadline=time.monotonic() + .2)
        assert recovered.failed is False
        monkeypatch.setattr(producer, '_remove_exact_docker_container', lambda **_: None)
        assert recovered.complete(deadline=time.monotonic() + .2)
    finally:
        recovered.anchor.close()


def test_empty_legacy_namespace_requires_no_docker_query(tmp_path):
    run = tmp_path / 'run'
    state = tmp_path / 'state'
    profile = LifecycleProfile(target_id='empty-owner', run_id='empty-run', configuration_root='empty-config',
                               repository_root=str(tmp_path), state_root=str(state),
                               run_root=str(state / 'run'), argv=(sys.executable, 'empty.py'), cwd=str(tmp_path))
    anchor = observation.CleanupDirectoryAnchor.open_before_launch(Path(profile.run_root))
    try:
        observer = observation.ManagedCleanupObserver(profile, anchor)
        assert observer.complete(deadline=time.monotonic() + .2)
        assert observer.checkpoint()['records'] == {}
    finally:
        anchor.close()


def test_missing_custody_probe_is_read_only(tmp_path):
    before = _files(tmp_path)
    assert observation.has_cleanup_custody(tmp_path / 'missing-run') is False
    assert _files(tmp_path) == before
