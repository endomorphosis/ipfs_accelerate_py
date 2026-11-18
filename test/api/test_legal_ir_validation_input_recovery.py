"""Filesystem and authority attacks on the isolated missing-input restorer."""
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'ops/restore_validation_inputs.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parents[2] / 'ipfs_accelerate_py/agent_supervisor/rescue/legal_ir_validation_input_recovery.py'
spec = importlib.util.spec_from_file_location('recovery_proposal', SOURCE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def pin(path, content):
    return {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


@pytest.fixture
def case(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    contents = [b'authentic source corpus\n', b'authentic frozen checkpoint\n']
    paths = [source / 'corpus.bin', source / 'checkpoint.json']
    for path, content in zip(paths, contents):
        path.write_bytes(content)
    pins = [pin(path, content) for path, content in zip(paths, contents)]
    tool = r.Recovery(str(tmp_path / 'mirror'), pins)
    return tool, paths, contents


def initialized(case):
    tool, paths, contents = case
    assert tool.initialize(True)['status'] == 'verified_snapshot_initialized'
    return tool, paths, contents


def results(report):
    return {row['path']: row for row in report['results']}


def tamper(path, content):
    path.chmod(0o600)
    path.write_bytes(content)


def test_literal_inventory_is_exact_protected_helper_authority():
    protected = SOURCE.parents[3] / 'width-replay-proposal/proposed/scripts/ops/legal_ir/replay_born_alive_width.py'
    if not protected.exists():
        protected = Path('/home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1/scripts/ops/legal_ir/replay_born_alive_width.py')
    helper_spec = importlib.util.spec_from_file_location('protected_pins_only', protected)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    expected = {p['path']: p for d in (384, 768) for p in helper.required_input_file_pins(d)}
    assert r.normalize_pins(r.LITERAL_PINS) == sorted(expected.values(), key=lambda p: p['path'])
    assert len(expected) == 23
    assert sum(p['bytes'] for p in expected.values()) == 1082445442
    assert r.AUTHORITY.endswith(hashlib.sha256(protected.read_bytes()).hexdigest())


def test_initialize_dry_run_verifies_without_creating_mirror(case):
    tool, paths, contents = case
    assert tool.initialize()['status'] == 'sources_verified_no_writes'
    assert not Path(tool.mirror).exists()
    assert [p.read_bytes() for p in paths] == contents


def test_snapshot_is_exact_and_content_addressed_and_manifest_literal(case):
    tool, paths, contents = initialized(case)
    manifest = json.loads((Path(tool.mirror) / 'manifest.json').read_bytes())
    assert manifest == tool.manifest
    for row, content in zip(tool.pins, sorted(zip(paths, contents), key=lambda pair: str(pair[0]))):
        blob = Path(tool.mirror) / 'objects' / row['sha256']
        assert blob.read_bytes() == content[1]
        assert blob.stat().st_nlink == 1
        assert stat.S_IMODE(blob.stat().st_mode) == 0o444


def test_missing_restore_and_identical_retry_are_exact_no_clobber(case):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    report = tool.recover(True, now=1000)
    assert report['restored_files'] == 1
    assert report['restored_bytes'] == len(contents[0])
    assert paths[0].read_bytes() == contents[0]
    inode = paths[0].stat().st_ino
    assert tool.recover(True, now=1001)['restored_files'] == 0
    assert paths[0].stat().st_ino == inode
    assert not list(paths[0].parent.glob('.input-recovery-*'))


def test_dry_run_of_missing_file_does_not_create_state_or_destination(case):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    report = tool.recover()
    assert results(report)[str(paths[0])]['status'] == 'verified_missing_restore_available'
    assert not paths[0].exists()
    assert not (Path(tool.mirror) / 'state').exists()


def test_present_wrong_file_is_never_hashed_repaired_or_overwritten(case):
    tool, paths, _ = initialized(case)
    paths[0].write_bytes(b'changed operator data')
    before = paths[0].stat()
    assert results(tool.recover(True))[str(paths[0])]['status'] == 'present_untouched'
    assert paths[0].read_bytes() == b'changed operator data'
    assert paths[0].stat().st_ino == before.st_ino


@pytest.mark.parametrize('kind', ['symlink', 'dangling', 'hardlink', 'fifo', 'directory'])
def test_present_alias_or_special_file_is_refused_unchanged(case, tmp_path, kind):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    other = tmp_path / 'other'
    other.write_bytes(contents[0])
    if kind == 'symlink':
        paths[0].symlink_to(other)
    elif kind == 'dangling':
        paths[0].symlink_to(tmp_path / 'absent')
    elif kind == 'hardlink':
        os.link(other, paths[0])
    elif kind == 'fifo':
        os.mkfifo(paths[0])
    else:
        paths[0].mkdir()
    before = paths[0].lstat()
    report = tool.recover(True)
    assert results(report)[str(paths[0])]['status'] == 'refused'
    assert paths[0].lstat().st_ino == before.st_ino
    assert other.read_bytes() == contents[0]


@pytest.mark.parametrize('mutation', ['wrong_hash', 'extra_path', 'missing_pin', 'extra_field', 'duplicate_path', 'bool_bytes'])
def test_manifest_cannot_grant_or_change_authority(case, tmp_path, mutation):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    manifest = json.loads(json.dumps(tool.manifest))
    if mutation == 'wrong_hash':
        manifest['pins'][0]['sha256'] = '0' * 64
    elif mutation == 'extra_path':
        manifest['pins'].append(pin(tmp_path / 'unauthorized', b'x'))
    elif mutation == 'missing_pin':
        manifest['pins'].pop()
    elif mutation == 'extra_field':
        manifest['command'] = 'arbitrary code'
    elif mutation == 'duplicate_path':
        manifest['pins'].append(manifest['pins'][0])
    else:
        manifest['pins'][0]['bytes'] = True
    tamper(Path(tool.mirror) / 'manifest.json', r.wire(manifest))
    with pytest.raises(r.Denied):
        tool.recover(True)
    assert not paths[0].exists()
    assert not (Path(tool.mirror) / 'state').exists()


def test_duplicate_json_key_manifest_refused_before_any_restore(case):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    payload = r.wire(tool.manifest).rstrip()[:-1] + b',"schema":"' + r.SCHEMA.encode() + b'"}'
    tamper(Path(tool.mirror) / 'manifest.json', payload)
    with pytest.raises(r.Denied, match='duplicate_json_key'):
        tool.recover(True)
    assert not paths[0].exists()


@pytest.mark.parametrize('corruption', ['wrong', 'truncated', 'larger', 'symlink', 'fifo', 'hardlink'])
def test_corrupt_or_aliased_backup_never_publishes_destination(case, tmp_path, corruption):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    row = next(p for p in tool.pins if p['path'] == str(paths[0]))
    blob = Path(tool.mirror) / 'objects' / row['sha256']
    if corruption in ('wrong', 'truncated', 'larger'):
        data = blob.read_bytes()
        tamper(blob, b'x' * len(data) if corruption == 'wrong' else data[:-1] if corruption == 'truncated' else data + b'x')
    else:
        blob.unlink()
        other = tmp_path / 'backup-other'
        other.write_bytes(b'a' * row['bytes'])
        if corruption == 'symlink':
            blob.symlink_to(other)
        elif corruption == 'fifo':
            os.mkfifo(blob)
        else:
            os.link(other, blob)
    report = tool.recover(True)
    assert results(report)[str(paths[0])]['status'] == 'refused'
    assert not paths[0].exists()
    assert not list(paths[0].parent.glob('.input-recovery-*'))


def test_corrupt_backup_attempt_also_reserves_cooldown_before_io(case, monkeypatch):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    row = next(p for p in tool.pins if p['path'] == str(paths[0]))
    blob = Path(tool.mirror) / 'objects' / row['sha256']
    tamper(blob, b'x' * row['bytes'])
    assert results(tool.recover(True, now=1000))[str(paths[0])]['status'] == 'refused'
    original = r.checked_read
    def no_backup_read(directory, name, *args, **kwargs):
        assert name != row['sha256'], 'cooldown must prevent repeated corrupt-backup I/O'
        return original(directory, name, *args, **kwargs)
    monkeypatch.setattr(r, 'checked_read', no_backup_read)
    report = tool.recover(True, now=1001)
    assert results(report)[str(paths[0])]['status'] == 'cooldown_deferred'
    assert report['attempted_files'] == 0


def test_initializer_rejects_changed_source_and_no_usable_manifest(case):
    tool, paths, _ = case
    paths[0].write_bytes(b'wrong')
    with pytest.raises(r.Denied):
        tool.initialize(True)
    assert not (Path(tool.mirror) / 'manifest.json').exists()


def test_initializer_rejects_source_symlink_and_hardlink(case, tmp_path):
    tool, paths, contents = case
    alias = tmp_path / 'alias'
    os.link(paths[0], alias)
    with pytest.raises(r.Denied, match='unaliased'):
        tool.initialize(True)
    alias.unlink()
    paths[0].unlink()
    alias.write_bytes(contents[0])
    paths[0].symlink_to(alias)
    with pytest.raises(OSError):
        tool.initialize(True)


def test_parent_symlink_is_not_followed(case, tmp_path):
    tool, paths, _ = initialized(case)
    old = paths[0].parent
    relocated = tmp_path / 'relocated'
    old.rename(relocated)
    old.symlink_to(relocated, target_is_directory=True)
    (relocated / paths[0].name).unlink()
    assert results(tool.recover(True))[str(paths[0])]['status'] == 'refused'
    assert not (relocated / paths[0].name).exists()


def test_directory_rebound_during_publish_cleans_only_own_inode(case, tmp_path, monkeypatch):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    old_parent = paths[0].parent
    relocated = tmp_path / 'renamed-parent'
    original = r.os.link
    def race(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if dst == paths[0].name:
            old_parent.rename(relocated)
            old_parent.mkdir()
            (old_parent / paths[0].name).write_bytes(b'foreign replacement')
        return result
    monkeypatch.setattr(r.os, 'link', race)
    report = tool.recover(True)
    assert results(report)[str(paths[0])]['reason'] == 'directory_custody_changed'
    assert paths[0].read_bytes() == b'foreign replacement'
    # Already-published exact bytes are never unlinked through a public name.
    assert (relocated / paths[0].name).read_bytes() == contents[0]
    assert not list(relocated.glob('.input-recovery-*'))


def test_no_clobber_race_preserves_foreign_file(case, monkeypatch):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    original = r.os.link
    def race(src, dst, **kwargs):
        if dst == paths[0].name:
            paths[0].write_bytes(b'concurrent foreign writer')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(r.os, 'link', race)
    assert results(tool.recover(True))[str(paths[0])]['reason'] == 'EEXIST'
    assert paths[0].read_bytes() == b'concurrent foreign writer'
    assert not list(paths[0].parent.glob('.input-recovery-*'))


def test_fsync_failure_after_publication_retains_exact_bytes_and_reservation(case, monkeypatch):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    original = r.os.fsync
    def fail_after_link(fd):
        if paths[0].exists() and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, 'injected fsync failure')
        return original(fd)
    monkeypatch.setattr(r.os, 'fsync', fail_after_link)
    assert results(tool.recover(True))[str(paths[0])]['status'] == 'refused'
    assert paths[0].read_bytes() == contents[0]
    assert not list(paths[0].parent.glob('.input-recovery-*'))


def test_state_fsync_failure_retains_existing_retry_history(case, monkeypatch):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    assert tool.recover(True, now=1000)['restored_files'] == 1
    paths[0].unlink()
    original_replace = r.os.replace
    original_fsync = r.os.fsync
    state_replaced = [False]
    def replace(*args, **kwargs):
        result = original_replace(*args, **kwargs)
        if args[1] == 'attempts.json':
            state_replaced[0] = True
        return result
    def fsync(fd):
        if state_replaced[0]:
            raise OSError(5, 'injected state commit durability failure')
        return original_fsync(fd)
    monkeypatch.setattr(r.os, 'replace', replace)
    monkeypatch.setattr(r.os, 'fsync', fsync)
    assert tool.recover(True, now=1600)['restored_files'] == 0
    state = json.loads((Path(tool.mirror) / 'state/attempts.json').read_bytes())
    assert state['attempts'][str(paths[0])] == [1000, 1600]
    assert not paths[0].exists()


def test_crash_reservation_cools_down_without_looping(case, monkeypatch):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    original = r.publish
    def fail_destination(directory, name, **kwargs):
        if name == paths[0].name:
            raise OSError(5, 'injected prepublication crash')
        return original(directory, name, **kwargs)
    monkeypatch.setattr(r, 'publish', fail_destination)
    assert tool.recover(True, now=1000)['restored_files'] == 0
    monkeypatch.setattr(r, 'publish', original)
    assert results(tool.recover(True, now=1001))[str(paths[0])]['status'] == 'cooldown_deferred'
    assert not paths[0].exists()
    assert tool.recover(True, now=1301)['restored_files'] == 1


def test_four_per_hour_bound_and_eventual_window_expiry(case):
    tool, paths, _ = initialized(case)
    for now in (1000, 1600, 2200, 2800):
        paths[0].unlink()
        assert tool.recover(True, now=now)['restored_files'] == 1
    paths[0].unlink()
    assert results(tool.recover(True, now=3400))[str(paths[0])]['status'] == 'cooldown_deferred'
    assert tool.recover(True, now=4601)['restored_files'] == 1


def test_future_state_clock_defers_fail_closed(case):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    assert tool.recover(True, now=10000)['restored_files'] == 1
    paths[0].unlink()
    assert results(tool.recover(True, now=5000))[str(paths[0])]['status'] == 'cooldown_deferred'


def test_per_run_byte_and_count_budgets_allow_other_small_ready_inputs(case):
    tool, paths, _ = initialized(case)
    for path in paths:
        path.unlink()
    report = tool.recover(True, max_restores=1)
    assert report['restored_files'] == 1
    assert sum(p.exists() for p in paths) == 1
    remaining = next(p for p in paths if not p.exists())
    row = next(p for p in tool.pins if p['path'] == str(remaining))
    report = tool.recover(True, budget_bytes=row['bytes'] - 1)
    assert results(report)[str(remaining)]['status'] == 'run_budget_deferred'
    assert not remaining.exists()


def test_two_recovery_processes_are_serialized_and_idempotent(case):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    context = multiprocessing.get_context('fork')
    barrier = context.Event()
    queue = context.Queue()
    def recover():
        barrier.wait(5)
        queue.put(tool.recover(True, now=1000))
    processes = [context.Process(target=recover) for _ in range(2)]
    for process in processes:
        process.start()
    barrier.set()
    reports = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert sum(report.get('restored_files', 0) for report in reports) == 1
    assert paths[0].read_bytes() == contents[0]
    assert paths[0].stat().st_nlink == 1
    assert not list(paths[0].parent.glob('.input-recovery-*'))


@pytest.mark.parametrize('after_link', [False, True])
def test_process_death_during_atomic_publication_leaves_no_orphan_or_alias(case, after_link):
    tool, paths, contents = initialized(case)
    paths[0].unlink()
    context = multiprocessing.get_context('fork')
    original = r.os.link
    def child():
        def kill(src, dst, **kwargs):
            if dst == paths[0].name:
                if after_link:
                    original(src, dst, **kwargs)
                os._exit(78 if after_link else 77)
            return original(src, dst, **kwargs)
        r.os.link = kill
        tool.recover(True, now=1000)
    process = context.Process(target=child)
    process.start()
    process.join(10)
    assert process.exitcode == (78 if after_link else 77)
    assert not list(paths[0].parent.glob('.input-recovery-*'))
    if after_link:
        assert paths[0].read_bytes() == contents[0]
        assert paths[0].stat().st_nlink == 1
        assert results(tool.recover(True, now=1001))[str(paths[0])]['status'] == 'present_untouched'
    else:
        assert not paths[0].exists()
        assert tool.recover(True, now=1301)['restored_files'] == 1


def test_unsupported_anonymous_staging_fails_closed_without_named_fallback(case, monkeypatch):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    original = r.os.open
    def unsupported(path, flags, *args, **kwargs):
        if flags & os.O_TMPFILE == os.O_TMPFILE:
            raise OSError(95, 'anonymous staging unsupported')
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(r.os, 'open', unsupported)
    assert results(tool.recover(True))[str(paths[0])]['status'] == 'refused'
    assert not paths[0].exists()
    assert not list(paths[0].parent.glob('.input-recovery-*'))


def test_failed_contended_copy_charges_file_and_byte_budgets(case, monkeypatch):
    tool, paths, _ = initialized(case)
    for path in paths:
        path.unlink()
    first = tool.pins[0]
    original = r.os.link
    def contention(src, dst, **kwargs):
        if dst == Path(first['path']).name:
            Path(first['path']).write_bytes(b'foreign winner')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(r.os, 'link', contention)
    report = tool.recover(True, max_restores=1, budget_bytes=first['bytes'])
    assert report['attempted_files'] == 1
    assert report['attempted_bytes'] == first['bytes']
    assert report['restored_files'] == 0
    assert results(report)[first['path']]['reason'] == 'EEXIST'
    other = next(p for p in tool.pins if p['path'] != first['path'])
    assert results(report)[other['path']]['status'] == 'run_budget_deferred'
    assert not Path(other['path']).exists()


def test_identical_bytes_at_two_paths_use_one_verified_blob(tmp_path):
    paths = [tmp_path / 'first', tmp_path / 'second']
    for path in paths:
        path.write_bytes(b'one identical authenticated object')
    tool = r.Recovery(str(tmp_path / 'mirror'), [pin(path, path.read_bytes()) for path in paths])
    tool.initialize(True)
    assert len(list((Path(tool.mirror) / 'objects').iterdir())) == 1
    for path in paths:
        path.unlink()
    assert tool.recover(True)['restored_files'] == 2
    assert paths[0].stat().st_ino != paths[1].stat().st_ino


@pytest.mark.parametrize('bad', [True, 0, -1, 2**30 + 1])
def test_pin_size_types_and_bounds_rejected(case, bad):
    tool, _, _ = case
    pins = json.loads(json.dumps(tool.pins))
    pins[0]['bytes'] = bad
    with pytest.raises(r.Denied):
        r.Recovery(tool.mirror, pins)


@pytest.mark.parametrize('path', ['relative', '/x/../y', '/x//y', '/x/./y', '/x/evil\nname'])
def test_noncanonical_pin_paths_rejected(case, path):
    tool, _, _ = case
    pins = json.loads(json.dumps(tool.pins))
    pins[0]['path'] = path
    with pytest.raises(r.Denied):
        r.Recovery(tool.mirror, pins)


def test_malformed_state_blocks_all_restoration(case):
    tool, paths, _ = initialized(case)
    tool.recover(True)
    paths[0].unlink()
    (Path(tool.mirror) / 'state/attempts.json').write_bytes(b'{"schema": "wrong"}')
    with pytest.raises(r.Denied):
        tool.recover(True)
    assert not paths[0].exists()


def test_cli_reports_partial_refusal_as_failed_service_status(case, monkeypatch, capsys):
    tool, paths, _ = initialized(case)
    paths[0].unlink()
    row = next(p for p in tool.pins if p['path'] == str(paths[0]))
    tamper(Path(tool.mirror) / 'objects' / row['sha256'], b'bad')
    monkeypatch.setattr(r, 'LITERAL_PINS', tuple(tool.pins))
    assert r.main(['recover', '--apply', '--mirror', tool.mirror]) == 1
    assert json.loads(capsys.readouterr().out)['results'][1]['status'] == 'refused'


def test_default_mirror_is_inside_ignored_board_data():
    assert r.DEFAULT_MIRROR == ('/home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1/'
                                'data/legal-ir-autoencoder-v1/validation-input-mirror')


@pytest.mark.parametrize('mode', ['dry_run', 'byte_budget', 'file_budget', 'cooldown', 'refused'])
def test_unresolved_missing_inputs_report_attention_and_cli_failure(case, monkeypatch, capsys, mode):
    tool, paths, _ = initialized(case)
    monkeypatch.setattr(r, 'LITERAL_PINS', tuple(tool.pins))
    paths[0].unlink()
    args = ['recover', '--mirror', tool.mirror]
    if mode != 'dry_run':
        args.append('--apply')
    if mode == 'byte_budget':
        args += ['--budget-bytes', '1']
    elif mode == 'file_budget':
        paths[1].unlink()
        args += ['--max-restores', '1']
    elif mode == 'cooldown':
        assert tool.recover(True, now=time.time())['restored_files'] == 1
        paths[0].unlink()
    elif mode == 'refused':
        row = next(p for p in tool.pins if p['path'] == str(paths[0]))
        tamper(Path(tool.mirror) / 'objects' / row['sha256'], b'bad')
    assert r.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'attention_required'
    assert report['unresolved_inputs'] == 1
    assert report['unresolved_paths'] == [str(paths[0])]
    assert report['present_files_sha_verified'] is False
    wanted = {'dry_run': 'verified_missing_restore_available', 'byte_budget': 'run_budget_deferred',
              'file_budget': 'run_budget_deferred', 'cooldown': 'cooldown_deferred', 'refused': 'refused'}
    assert results(report)[str(paths[0])]['status'] == wanted[mode]


def test_successful_recovery_reports_no_unresolved_inputs_and_exit_zero(case, monkeypatch, capsys):
    tool, paths, contents = initialized(case)
    monkeypatch.setattr(r, 'LITERAL_PINS', tuple(tool.pins))
    paths[0].unlink()
    assert r.main(['recover', '--apply', '--mirror', tool.mirror]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'complete'
    assert report['unresolved_inputs'] == 0 and report['unresolved_paths'] == []
    assert report['restored_files'] == 1
    assert paths[0].read_bytes() == contents[0]


def test_present_changed_input_stays_explicitly_unverified_with_exit_zero(case, monkeypatch, capsys):
    tool, paths, _ = initialized(case)
    monkeypatch.setattr(r, 'LITERAL_PINS', tuple(tool.pins))
    paths[0].write_bytes(b'present operator-authored change')
    original = r.checked_read
    def no_present_content_read(directory, name, *args, **kwargs):
        assert name not in [p.name for p in paths]
        return original(directory, name, *args, **kwargs)
    monkeypatch.setattr(r, 'checked_read', no_present_content_read)
    assert r.main(['recover', '--apply', '--mirror', tool.mirror]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'complete' and report['unresolved_inputs'] == 0
    assert report['present_files_sha_verified'] is False
    assert all(item['status'] == 'present_untouched' for item in report['results'])
    assert paths[0].read_bytes() == b'present operator-authored change'


def test_busy_peer_is_safe_and_does_not_claim_completed_inspection(case, monkeypatch, capsys):
    tool, _, _ = initialized(case)
    monkeypatch.setattr(r, 'LITERAL_PINS', tuple(tool.pins))
    tool.recover(True)  # Establish the private state/lock without any input write.
    with (Path(tool.mirror) / 'state/recovery.lock').open('rb') as lock:
        r.fcntl.flock(lock.fileno(), r.fcntl.LOCK_EX | r.fcntl.LOCK_NB)
        assert r.main(['recover', '--apply', '--mirror', tool.mirror]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'busy' and report['results'] == []
    assert 'unresolved_inputs' not in report


def test_cli_has_no_arbitrary_pin_config_or_command_option():
    process = subprocess.run([sys.executable, str(SOURCE), 'recover', '--pins', '/tmp/untrusted'],
                             capture_output=True, text=True, timeout=5)
    assert process.returncode == 2
    assert 'unrecognized arguments' in process.stderr
