import subprocess

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.verification.retained_workspace import fingerprint_retained_tree


def test_retained_callback_fingerprints_bundled_toolchain_past_old_limit(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path).decode().strip()
    git('init', '-b', 'retained')
    git('-c', 'user.email=test@example.invalid', '-c', 'user.name=Test',
        'commit', '--allow-empty', '-m', 'baseline')
    head = git('rev-parse', 'HEAD')
    with (tmp_path / 'toolchain.bin').open('wb') as stream:
        stream.truncate(512 * 1024 * 1024 + 1)
    expected = fingerprint_retained_tree(tmp_path)
    result = PortalImplementationDaemon._retained_workspace_content_fingerprint(
        tmp_path, baseline_ref=head)
    for name in ('content_digest', 'content_bytes', 'entry_count'):
        assert result[name] == expected[name]
    assert result['head'] == result['baseline_commit'] == head
    assert result['branch'] == 'retained'
    assert result['fingerprint_id'].startswith('sha256:')
