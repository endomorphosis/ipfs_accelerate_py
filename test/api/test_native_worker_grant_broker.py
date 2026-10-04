"""A pathname-bound live socket cannot be mistaken for a stale broker."""
import socket

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerControlError, TypedStateOwnerGrantBroker,
)


def test_bound_socket_before_listen_is_preserved_then_closed_socket_is_recovered(tmp_path):
    path = tmp_path / "grants.sock"
    tmp_path.chmod(0o700)
    broker = TypedStateOwnerGrantBroker(
        socket_path=path, bootstrap_secret="1" * 64, store_id="test-store",
        resolve_credential=lambda *_: "unused_credential",
    )
    live = socket.socket(socket.AF_UNIX)
    try:
        live.bind(str(path))
        before = path.lstat()
        with pytest.raises(QuackStateServerControlError, match="bound socket"):
            broker.start()
        assert path.lstat().st_ino == before.st_ino
        # Closing the actual kernel socket, rather than ECONNREFUSED alone,
        # establishes that this retained filesystem pathname is stale.
        live.close()
        broker.start()
        assert broker.alive()
    finally:
        live.close()
        broker.stop()
    assert not path.exists()
