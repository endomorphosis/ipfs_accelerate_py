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
def test_base_cache_encrypts_remote_and_decrypts_on_readthrough(tmp_path):
	from ipfs_accelerate_py.common.base_cache import BaseAPICache
	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
	import ipfs_accelerate_py.p2p_tasks.client as p2p_client

	class DummyCache(BaseAPICache):
		def get_cache_namespace(self) -> str:
			return "dummy_api"

		def extract_validation_fields(self, operation: str, data):
			return None

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

		secret = "test-provider-secret"

		# Writer cache instance.
		writer = DummyCache(
			cache_dir=str(tmp_path / "local_writer"),
			enable_persistence=False,
			enable_p2p=True,
			p2p_shared_secret=secret,
			p2p_secret_salt=b"dummy-task-p2p-cache-real",
			enable_pubsub=False,
			default_ttl=60,
		)
		writer._task_p2p_remote = lambda: remote

		writer.put("op", {"value": 123}, "a", x=1, ttl=60)

		# Verify the remote stored ciphertext wrapper (not plaintext).
		cache_key = writer.make_cache_key("op", "a", x=1)
		remote_key = writer._task_p2p_key(cache_key)
		remote_resp = p2p_client.cache_get_sync(remote=remote, key=remote_key, timeout_s=10.0)
		assert remote_resp.get("ok") is True
		assert remote_resp.get("hit") is True
		wrapped = remote_resp.get("value")
		assert isinstance(wrapped, dict)
		assert wrapped.get("enc") == "fernet-v1"
		assert isinstance(wrapped.get("ct"), str)

		# Reader cache instance should read-through from remote and decrypt.
		reader = DummyCache(
			cache_dir=str(tmp_path / "local_reader"),
			enable_persistence=False,
			enable_p2p=True,
			p2p_shared_secret=secret,
			p2p_secret_salt=b"dummy-task-p2p-cache-real",
			enable_pubsub=False,
			default_ttl=60,
		)
		reader._task_p2p_remote = lambda: remote

		assert reader.get("op", "a", x=1) == {"value": 123}
		stats = reader.get_stats()
		assert stats.get("peer_hits", 0) >= 1

		# Wrong secret should not decrypt remote values (anti-injection).
		wrong = DummyCache(
			cache_dir=str(tmp_path / "local_wrong"),
			enable_persistence=False,
			enable_p2p=True,
			p2p_shared_secret="wrong-secret",
			p2p_secret_salt=b"dummy-task-p2p-cache-real",
			enable_pubsub=False,
			default_ttl=60,
		)
		wrong._task_p2p_remote = lambda: remote
		assert wrong.get("op", "a", x=1) is None
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
