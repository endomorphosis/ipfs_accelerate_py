import json
import multiprocessing as mp
import os
import socket
import time

import pytest


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401

        return True
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_for_announce(path: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return data if isinstance(data, dict) else {}
            except Exception:
                pass
        time.sleep(0.1)
    raise TimeoutError(f"announce file not written: {path}")


def _run_worker_with_service(
    *,
    queue_path: str,
    listen_port: int,
    announce_file: str,
    cache_dir: str,
    identity: str,
) -> None:
    # Deterministic local-only setup.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file

    # Keep discovery mechanisms off inside the service; the *client* test will
    # rely on announce-file dialing using peer-id only.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

    # Ensure worker claims tool.call.
    os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "tool.call"

    class _Accel:
        def call_tool(self, tool_name: str, args: dict):
            return {
                "ok": True,
                "identity": str(identity),
                "tool": str(tool_name),
                "args": dict(args or {}),
            }

    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    run_worker(
        queue_path=queue_path,
        worker_id=f"unit-test-worker:{identity}",
        poll_interval_s=0.05,
        once=False,
        p2p_service=True,
        p2p_listen_port=int(listen_port),
        accelerate_instance=_Accel(),
        supported_task_types=["tool.call"],
    )


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_peer_id_only_uses_announce_file(tmp_path, monkeypatch):
    """E2E: dial using only peer_id (multiaddr empty).

    The client discovers a dialable multiaddr via announce-file, so the caller
    never needs to pass/share the multiaddr explicitly.
    """

    from ipfs_accelerate_py.p2p_tasks.client import (
        RemoteQueue,
        cache_get_sync,
        cache_set_sync,
        discover_status_sync,
        submit_task_sync,
        wait_task,
    )

    import anyio

    listen_port = _free_port()
    queue_path = str(tmp_path / "queue.duckdb")
    announce_file = str(tmp_path / "announce.json")
    cache_dir = str(tmp_path / "cache")

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_run_worker_with_service,
        kwargs={
            "queue_path": queue_path,
            "listen_port": listen_port,
            "announce_file": announce_file,
            "cache_dir": cache_dir,
            "identity": "B",
        },
        daemon=True,
    )

    proc.start()
    try:
        ann = _wait_for_announce(announce_file)
        peer_id = str(ann.get("peer_id") or "").strip()
        assert peer_id

        # Point the client at this announce file so it can discover the address.
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE", announce_file)

        # The critical part: no multiaddr supplied.
        remote = RemoteQueue(peer_id=peer_id, multiaddr="")

        trace = discover_status_sync(remote=remote, timeout_s=10.0)
        assert trace.get("ok") is True
        attempts = trace.get("attempts")
        assert isinstance(attempts, list) and attempts
        assert any(a.get("method") == "announce-file" and a.get("ok") is True for a in attempts)

        set_resp = cache_set_sync(remote=remote, key="k", value={"from": "peer_id_only"}, ttl_s=5.0, timeout_s=10.0)
        assert set_resp.get("ok") is True

        hit = cache_get_sync(remote=remote, key="k", timeout_s=10.0)
        assert hit.get("ok") is True
        assert hit.get("hit") is True
        assert hit.get("value") == {"from": "peer_id_only"}

        task_id = submit_task_sync(
            remote=remote,
            task_type="tool.call",
            model_name="demo",
            payload={"tool": "unit_test.identity", "args": {"ping": True}},
        )
        assert isinstance(task_id, str) and task_id

        async def _wait() -> dict | None:
            return await wait_task(remote=remote, task_id=task_id, timeout_s=25.0)

        task = anyio.run(_wait, backend="trio")
        assert isinstance(task, dict)
        assert task.get("status") == "completed"
        result = task.get("result")
        assert isinstance(result, dict)

        inner = result.get("result")
        assert isinstance(inner, dict)
        assert inner.get("ok") is True
        assert inner.get("identity") == "B"
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=5.0)
        except Exception:
            pass
