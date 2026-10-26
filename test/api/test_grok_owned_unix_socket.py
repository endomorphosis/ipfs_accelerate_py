"""Unexecuted physical listener controls; no task/native authority implied."""
import os
import socket

import pytest

import ipfs_accelerate_py.cli_runtime.grok_owned_unix_socket as consumer


def sample(listener, endpoint, parent, path, descriptor):
    return consumer._sample_created_listener(listener, endpoint, parent, path, descriptor, 65536)


def test_exact_real_private_listener_sample(tmp_path):
    endpoint = tmp_path/'s'
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint)); endpoint.chmod(0o600); listener.listen(1)
        observed = sample(listener, endpoint, consumer.signature(tmp_path.lstat()),
            consumer.signature(endpoint.lstat()), consumer.signature(os.fstat(listener.fileno())))
        assert observed['exact_owned_socket_count'] == observed['entry_count'] == 1
        assert observed['apparent_bytes'] == endpoint.lstat().st_size
        assert observed['filesystem_quota'] is False
    endpoint.unlink()


@pytest.mark.parametrize('change', ['file', 'socket', 'fifo', 'symlink', 'replacement', 'mode', 'parent_mode'])
def test_physical_sampler_refuses_every_unowned_or_changed_entry(tmp_path, change):
    endpoint = tmp_path/'s'; extra = None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint)); endpoint.chmod(0o600); listener.listen(1)
        parent = consumer.signature(tmp_path.lstat()); path = consumer.signature(endpoint.lstat())
        descriptor = consumer.signature(os.fstat(listener.fileno()))
        try:
            if change == 'file':
                (tmp_path/'unexpected').write_bytes(b'not-an-allowed-output')
            elif change == 'socket':
                extra = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); extra.bind(str(tmp_path/'other'))
            elif change == 'fifo':
                os.mkfifo(tmp_path/'other')
            elif change == 'symlink':
                (tmp_path/'other').symlink_to(endpoint)
            elif change == 'replacement':
                endpoint.unlink(); extra = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); extra.bind(str(endpoint))
                endpoint.chmod(0o600)
            elif change == 'mode':
                endpoint.chmod(0o660)
            else:
                tmp_path.chmod(0o750)
            with pytest.raises(consumer.OwnedUnixSocketError):
                sample(listener, endpoint, parent, path, descriptor)
        finally:
            if extra is not None:
                extra.close()
            tmp_path.chmod(0o700)
            for item in tmp_path.iterdir():
                item.unlink()


def test_arbitrary_disk_object_cannot_create_a_listener_or_grant(tmp_path):
    parent = tmp_path/'new'
    with pytest.raises(consumer.OwnedUnixSocketError):
        consumer.open_owned_unix_listener(disk=object(), parent=parent)
    assert not parent.exists()
