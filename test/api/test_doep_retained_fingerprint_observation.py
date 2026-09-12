"""Retained-workspace observations must never refresh a private Git index."""
from pathlib import Path
import os
import pytest
from test.api.test_agent_supervisor_implementation_protected_paths import (
    PortalImplementationDaemon,
    test_ephemeral_verification_lock_deferral_does_not_consume_attempt as full_recovery,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnershipError


def test_rejected_retained_recovery_preserves_private_index_bytes(tmp_path, monkeypatch):
    original = PortalImplementationDaemon.recover_retained_verification_deferred_candidate
    observations = []
    def observed(self, *, task, retained_candidate_receipt):
        workspace = Path(retained_candidate_receipt['workspace_path'])
        current = self.worktree_lifecycle.load_task_attempt(
            canonical_task_cid=self._canonical_ref(task), task_id=task.task_id,
            attempt=retained_candidate_receipt['portal_attempt'],
        )
        # Only the original test's genuine different-workspace native successor.
        if current.workspace_path == str(workspace):
            return original(self,task=task,retained_candidate_receipt=retained_candidate_receipt)
        import subprocess
        def git(*args):
            return subprocess.check_output(['git','-C',str(workspace),*args])
        tracked = [workspace/os.fsdecode(p) for p in git('ls-files','-z').split(b'\0') if p]
        target = next(p for p in tracked if p.is_file() and not p.is_symlink())
        st=target.stat();os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000_000))
        index=Path(git('rev-parse','--git-path','index').decode().strip())
        if not index.is_absolute():index=workspace/index
        before=index.read_bytes()
        try:
            return original(self,task=task,retained_candidate_receipt=retained_candidate_receipt)
        except OwnershipError:
            observations.append((before,index.read_bytes()))
            raise
    monkeypatch.setattr(PortalImplementationDaemon,'recover_retained_verification_deferred_candidate',observed)
    full_recovery(tmp_path,monkeypatch,acquire_successor=True)
    assert observations
    assert all(before==after for before,after in observations)


def _git_observe(workspace, *arguments):
    import subprocess
    return subprocess.check_output(
        ['git', '--no-optional-locks', '-C', str(workspace), *arguments],
        stderr=subprocess.PIPE,
    )


def _index_evidence(workspace):
    index = Path(os.fsdecode(_git_observe(workspace, 'rev-parse', '--git-path', 'index')).strip())
    if not index.is_absolute():
        index = workspace / index
    value = index.stat()
    return (index.read_bytes(), value.st_dev, value.st_ino,
            value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
            index.with_name(index.name + '.lock').exists())


@pytest.mark.parametrize('with_submodule', [False, True])
def test_fingerprint_preserves_native_parent_and_submodule_index(tmp_path, monkeypatch, with_submodule):
    import subprocess

    def git(repo, *args):
        return subprocess.check_output(
            ['git', '-c', 'protocol.file.allow=always', '-C', str(repo), *args],
            stderr=subprocess.PIPE,
        )

    def initialized(path):
        path.mkdir()
        git(path, 'init', '-b', 'main')
        git(path, 'config', 'user.name', 'Native fixture')
        git(path, 'config', 'user.email', 'native-fixture@example.invalid')
        (path/'tracked.py').write_text('original = True\n')
        git(path, 'add', 'tracked.py')
        git(path, 'commit', '-m', 'fixture')
        return path

    repo = initialized(tmp_path/'source')
    if with_submodule:
        child = initialized(tmp_path/'child-source')
        git(repo, 'submodule', 'add', str(child), 'nested')
        git(repo, 'commit', '-am', 'add actual nested repository')
    workspace = tmp_path/'retained'
    git(repo, 'worktree', 'add', '-b', 'retained', str(workspace), 'HEAD')
    if with_submodule:
        git(workspace, 'submodule', 'update', '--init')
    baseline = os.fsdecode(git(workspace, 'rev-parse', 'HEAD')).strip()
    members = [workspace] + ([workspace/'nested'] if with_submodule else [])
    # Prime actual Git stat caches, then change only tracked-file metadata.
    for member in members:
        git(member, 'status', '--porcelain')
        tracked = member/'tracked.py'
        st = tracked.stat()
        os.utime(tracked, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    # The explicit command option must override an inherited request for
    # optional writes, and propagate through Git's nested status queries.
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS', '1')
    before = {str(member): _index_evidence(member) for member in members}
    first = PortalImplementationDaemon._retained_workspace_content_fingerprint(
        workspace, baseline_ref=baseline,
    )
    assert {str(member): _index_evidence(member) for member in members} == before
    second = PortalImplementationDaemon._retained_workspace_content_fingerprint(
        workspace, baseline_ref=baseline,
    )
    assert first == second
    assert {str(member): _index_evidence(member) for member in members} == before
    assert first['head'] == first['baseline_commit'] == baseline
    assert first['branch'] == 'retained'
    assert first['status_bytes'] == 0
