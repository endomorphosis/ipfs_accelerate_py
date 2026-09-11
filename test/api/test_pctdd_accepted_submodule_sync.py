"""Actual Git staging/ref races for accepted detached submodule maintenance."""
import subprocess
import pytest
from ipfs_accelerate_py.agent_supervisor.merge import accepted_submodule_sync as m


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()


def init(repo):
    repo.mkdir(parents=True)
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'config', 'user.name', 'Fixture')


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'parent'; init(root)
    child = root / 'external' / 'datasets'; init(child)
    (child / 'original').write_text('before\n'); git(child, 'add', '.'); git(child, 'commit', '-qm', 'old')
    old = git(child, 'rev-parse', 'HEAD')
    (child / 'original').write_text('after\n'); (child / 'added').write_text('new\n')
    git(child, 'add', '.'); git(child, 'commit', '-qm', 'accepted child')
    target = git(child, 'rev-parse', 'HEAD')
    git(root, 'add', 'external/datasets'); git(root, 'commit', '-qm', 'accepted parent')
    parent = git(root, 'rev-parse', 'HEAD'); git(child, 'checkout', '--detach', old)
    archive = tmp_path / 'archive'; archive.mkdir()
    return dict(root=root, child=child, old=old, target=target, parent=parent, archive=archive)


def run(s, phase=lambda *a, **k: None, guard=lambda: None, prepare=None):
    return m.synchronize_accepted_submodule(s['root'], 'external/datasets',
        expected_parent=s['parent'], expected_old=s['old'], archive=s['archive'],
        phase=phase, custody_guard=guard, prepare_index_lock=prepare)


def primitive(s, phase=lambda *a, **k: None, revalidate=lambda: None, prepare=None):
    index = m.git_index_path(s['child'])
    return m._advance_detached_head(s['child'], expected_head=s['old'], target=s['target'],
        expected_index_sha256=m.digest(index.read_bytes()), archive=s['archive'], label='trial',
        phase=phase, revalidate=revalidate, prepare_index_lock=prepare)


def test_actual_native_lease_syncs_exact_accepted_gitlink(setup):
    s=setup; original=(s['child']/'.git/index').read_bytes(); events=[]
    result=run(s, lambda name, **kw: events.append(name))
    assert result['old']==s['target'] and result['target']==s['target']
    assert m.detached_at(s['child'],s['target'])
    assert (s['child']/'original').read_text()=='after\n'
    assert (s['child']/'added').read_text()=='new\n'
    assert git(s['root'],'status','--porcelain')==''
    assert (s['archive']/'submodule-original.index').read_bytes()==original
    assert events[-1]=='accepted_submodule_synchronized'
    assert not (s['child']/'.git/index.lock').exists()


@pytest.mark.parametrize('change', ['dirty','untracked','attached','parent','staging','lease','provider'])
def test_refuses_unadmitted_checkout_or_custody_before_effect(setup, change):
    s=setup; old_index=(s['child']/'.git/index').read_bytes()
    if change=='dirty':(s['child']/'original').write_text('user work')
    if change=='untracked':(s['child']/'user').write_text('user work')
    if change=='attached':git(s['child'],'checkout','-b','working')
    if change=='parent':git(s['root'],'commit','--allow-empty','-qm','concurrent')
    if change=='staging':
        (s['root']/'new').write_text('user');git(s['root'],'add','new')
    if change=='lease':
        from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import checkout_mutation_lock_path
        checkout_mutation_lock_path(s['root']).write_text('{"pid":1,"kind":"merge"}')
    def guard():
        if change=='provider':raise m.Refused('current_provider')
    with pytest.raises(m.Refused):run(s,guard=guard)
    assert (s['child']/'.git/index').read_bytes()==old_index
    assert not list(s['archive'].iterdir())


@pytest.mark.parametrize('kind', ['head','symbolic'])
def test_expected_old_detached_head_transaction_rejects_concurrent_change(setup, kind):
    s=setup; calls=0
    def revalidate():
        nonlocal calls
        calls+=1
        if calls==1:
            if kind=='head':git(s['child'],'update-ref','HEAD',s['target'],s['old'])
            else:git(s['child'],'symbolic-ref','HEAD','refs/heads/master')
    with pytest.raises(m.Refused):primitive(s,revalidate=revalidate)
    assert not (s['child']/'added').exists()
    assert not (s['child']/'.git/index.lock').exists()


def test_prepared_head_lock_excludes_ref_and_symbolic_writers(setup):
    s=setup; outcomes=[]
    def phase(name, **kw):
        if name=='trial_ref_prepared':
            for args in [('update-ref','HEAD',s['target']),('symbolic-ref','HEAD','refs/heads/foreign')]:
                outcomes.append(subprocess.run(['git','-C',str(s['child']),*args],capture_output=True).returncode)
    primitive(s,phase=phase)
    assert len(outcomes)==2 and all(outcomes)
    assert m.detached_at(s['child'],s['target'])


def test_foreign_index_lock_is_preserved_without_archive_effect(setup):
    s=setup; lock=s['child']/'.git/index.lock';lock.write_bytes(b'foreign')
    with pytest.raises(FileExistsError):run(s)
    assert lock.read_bytes()==b'foreign' and not list(s['archive'].iterdir())


def test_foreign_lock_after_explicit_quarantine_is_preserved(setup):
    s=setup; lock=s['child']/'.git/index.lock';lock.write_bytes(b'authorized old')
    def prepare(path, archive, phase):
        path.rename(archive/'preserved.lock'); path.write_bytes(b'foreign new')
    with pytest.raises(FileExistsError):primitive(s,prepare=prepare)
    assert lock.read_bytes()==b'foreign new'
    assert (s['archive']/'preserved.lock').read_bytes()==b'authorized old'
    assert m.detached_at(s['child'],s['old'])


def test_parent_drift_after_ref_prepare_denied_and_original_index_preserved(setup):
    s=setup; before=(s['child']/'.git/index').read_bytes()
    def phase(name, **kw):
        if name=='submodule_ref_prepared':git(s['root'],'commit','--allow-empty','-qm','uncooperative parent writer')
    with pytest.raises(m.Refused,match='accepted_parent_or_index_changed'):run(s,phase=phase)
    assert (s['child']/'.git/index').read_bytes()==before
    assert not (s['child']/'added').exists()


def test_failure_after_worktree_effects_retains_index_lock_and_old_head(setup):
    s=setup; count=0; events=[]
    def revalidate():
        nonlocal count
        count+=1
        if count==3:raise m.Refused('injected_after_read_tree')
    with pytest.raises(m.Refused):primitive(s,revalidate=revalidate,phase=lambda name,**kw:events.append((name,kw)))
    assert (s['child']/'added').exists() and (s['child']/'.git/index.lock').exists()
    assert m.detached_at(s['child'],s['old'])
    assert events[-1][0]=='trial_incomplete_preserved'
    assert events[-1][1]['automatic_rollback'] is False


def test_archive_collision_preserves_existing_evidence(setup):
    s=setup;(s['archive']/'trial-original.index').write_bytes(b'prior')
    with pytest.raises(FileExistsError):primitive(s)
    assert (s['archive']/'trial-original.index').read_bytes()==b'prior'
    assert not (s['child']/'added').exists()


def test_index_directory_fsync_failure_records_actual_publication_but_not_durability(
    setup, monkeypatch
):
    import os

    s = setup
    index = s["child"] / ".git/index"
    original = index.read_bytes()
    events = []
    fsync = m.os.fsync

    def fail(fd):
        if (
            os.readlink(f"/proc/self/fd/{fd}") == str(index.parent)
            and index.read_bytes() != original
        ):
            raise OSError("injected index directory fsync failure")
        return fsync(fd)

    monkeypatch.setattr(m.os, "fsync", fail)
    with pytest.raises(OSError):
        primitive(s, phase=lambda name, **kw: events.append((name, kw)))
    assert index.read_bytes() != original and (index.parent / "index.lock").exists()
    assert m.detached_at(s["child"], s["old"])
    assert events[-1][1]["index_published"] is True
    assert events[-1][1]["index_publication_durable"] is False
