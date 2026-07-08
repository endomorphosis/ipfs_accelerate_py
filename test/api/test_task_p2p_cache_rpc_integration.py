import json
import os
import socket
import time
import multiprocessing as mp

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


def _run_service(queue_path: str, listen_port: int, announce_file: str, cache_dir: str) -> None:
	# Configure cache + announce file for the service process.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	# mDNS isn't needed for direct dialing in tests.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

	from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
	import anyio

	async def _main() -> None:
		await serve_task_queue(queue_path=queue_path, listen_port=listen_port, accelerate_instance=None)

	anyio.run(_main, backend="trio")


def _wait_for_announce(path: str, timeout_s: float = 15.0) -> dict:
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		if os.path.exists(path) and os.path.getsize(path) > 0:
			try:
				with open(path, "r", encoding="utf-8") as handle:
					return json.load(handle)
			except Exception:
				pass
		time.sleep(0.1)
	raise TimeoutError(f"announce file not written: {path}")


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_cache_rpc_roundtrip(tmp_path, monkeypatch):
	from ipfs_accelerate_py.p2p_tasks.client import (
		RemoteQueue,
		cache_get_sync,
		cache_has_sync,
		cache_set_sync,
		cache_delete_sync,
	)

	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")
	cache_dir = str(tmp_path / "p2p_cache")
	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service, args=(queue_path, port, announce_file, cache_dir), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		remote = RemoteQueue(peer_id=str(ann.get("peer_id") or ""), multiaddr=str(ann.get("multiaddr") or ""))

		# get/has miss
		miss = cache_get_sync(remote=remote, key="k1", timeout_s=10.0)
		assert miss.get("ok") is True
		assert miss.get("hit") is False

		has_miss = cache_has_sync(remote=remote, key="k1", timeout_s=10.0)
		assert has_miss.get("ok") is True
		assert has_miss.get("hit") is False

		# set + get hit
		set_resp = cache_set_sync(remote=remote, key="k1", value={"a": 1}, ttl_s=10.0, timeout_s=10.0)
		assert set_resp.get("ok") is True

		hit = cache_get_sync(remote=remote, key="k1", timeout_s=10.0)
		assert hit.get("ok") is True
		assert hit.get("hit") is True
		assert hit.get("value") == {"a": 1}

		has_hit = cache_has_sync(remote=remote, key="k1", timeout_s=10.0)
		assert has_hit.get("ok") is True
		assert has_hit.get("hit") is True

		# delete
		del_resp = cache_delete_sync(remote=remote, key="k1", timeout_s=10.0)
		assert del_resp.get("ok") is True

		after = cache_get_sync(remote=remote, key="k1", timeout_s=10.0)
		assert after.get("ok") is True
		assert after.get("hit") is False
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_cache_rpc_ttl_expires(tmp_path):
	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, cache_get_sync, cache_set_sync

	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")
	cache_dir = str(tmp_path / "p2p_cache")
	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service, args=(queue_path, port, announce_file, cache_dir), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		remote = RemoteQueue(peer_id=str(ann.get("peer_id") or ""), multiaddr=str(ann.get("multiaddr") or ""))

		cache_set_sync(remote=remote, key="k2", value="v", ttl_s=0.2, timeout_s=10.0)
		assert cache_get_sync(remote=remote, key="k2", timeout_s=10.0).get("hit") is True

		time.sleep(0.35)
		assert cache_get_sync(remote=remote, key="k2", timeout_s=10.0).get("hit") is False
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
