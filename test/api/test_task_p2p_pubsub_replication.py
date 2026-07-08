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


def _have_crypto() -> bool:
	try:
		import cryptography  # noqa: F401
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
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
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
@pytest.mark.skipif(not _have_crypto(), reason="cryptography not installed")
def test_task_p2p_pubsub_like_replication_fanout(tmp_path):
	"""Validates the pubsub-like replication fanout in BaseAPICache.

	Mechanism under test: BaseAPICache._task_p2p_set() replicates cache writes to
	configured bootstrap peers (fanout) when enable_pubsub is true.
	"""

	from ipfs_accelerate_py.common.base_cache import BaseAPICache
	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
	import ipfs_accelerate_py.p2p_tasks.client as p2p_client

	class DummyCache(BaseAPICache):
		def get_cache_namespace(self) -> str:
			return "dummy_api_pubsub"

		def extract_validation_fields(self, operation: str, data):
			return None

	# Start two independent services.
	port_a = _free_port()
	port_b = _free_port()

	queue_a = str(tmp_path / "queue_a.duckdb")
	queue_b = str(tmp_path / "queue_b.duckdb")
	announce_a = str(tmp_path / "announce_a.json")
	announce_b = str(tmp_path / "announce_b.json")
	cache_dir_a = str(tmp_path / "p2p_cache_a")
	cache_dir_b = str(tmp_path / "p2p_cache_b")

	ctx = mp.get_context("spawn")
	proc_a = ctx.Process(target=_run_service, args=(queue_a, port_a, announce_a, cache_dir_a), daemon=True)
	proc_b = ctx.Process(target=_run_service, args=(queue_b, port_b, announce_b, cache_dir_b), daemon=True)
	proc_a.start()
	proc_b.start()

	try:
		ann_a = _wait_for_announce(announce_a)
		ann_b = _wait_for_announce(announce_b)
		remote_a = RemoteQueue(peer_id=str(ann_a.get("peer_id") or ""), multiaddr=str(ann_a.get("multiaddr") or ""))
		remote_b = RemoteQueue(peer_id=str(ann_b.get("peer_id") or ""), multiaddr=str(ann_b.get("multiaddr") or ""))

		secret = "test-provider-secret"

		# Configure replication fanout to B.
		os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = remote_b.multiaddr

		writer = DummyCache(
			cache_dir=str(tmp_path / "local_writer"),
			enable_persistence=False,
			enable_p2p=True,
			p2p_shared_secret=secret,
			p2p_secret_salt=b"dummy-task-p2p-cache-pubsub",
			enable_pubsub=True,
			default_ttl=60,
		)
		# Primary remote for the write.
		writer._task_p2p_remote = lambda: remote_a

		writer.put("op", {"value": 456}, "b", y=2, ttl=60)

		cache_key = writer.make_cache_key("op", "b", y=2)
		remote_key = writer._task_p2p_key(cache_key)

		# Ensure the replicated entry exists on B.
		resp_b = p2p_client.cache_get_sync(remote=remote_b, key=remote_key, timeout_s=10.0)
		assert resp_b.get("ok") is True
		assert resp_b.get("hit") is True

		wrapped = resp_b.get("value")
		assert isinstance(wrapped, dict)
		assert wrapped.get("enc") == "fernet-v1"
		assert isinstance(wrapped.get("ct"), str)
	finally:
		try:
			proc_a.terminate()
			proc_b.terminate()
		finally:
			proc_a.join(timeout=5.0)
			proc_b.join(timeout=5.0)
