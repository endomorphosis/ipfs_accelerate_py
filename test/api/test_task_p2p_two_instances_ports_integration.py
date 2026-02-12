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
	# Local-only deterministic behavior.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

	# Enable cache RPC on the service.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir

	# Enable tool calls.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

	# Ensure the worker will claim tool.call tasks.
	os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "tool.call"

	class _Accel:
		def call_tool(self, tool_name: str, args: dict):
			# Include identity so the caller can verify which instance executed.
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
def test_task_p2p_two_instances_different_ports_cache_and_toolcall(tmp_path):
	"""E2E: run two P2P instances on one machine using different ports.

	Validates:
	- both instances can bind concurrently
	- cache.get/set works against the *other* instance
	- tool.call tasks execute remotely (submit->wait)
	"""
	import anyio

	from ipfs_accelerate_py.p2p_tasks.client import (
		RemoteQueue,
		cache_get_sync,
		cache_set_sync,
		submit_task_sync,
		wait_task,
	)

	port_a = _free_port()
	port_b = _free_port()
	assert port_a != port_b

	queue_a = str(tmp_path / "queue_a.duckdb")
	queue_b = str(tmp_path / "queue_b.duckdb")
	announce_a = str(tmp_path / "announce_a.json")
	announce_b = str(tmp_path / "announce_b.json")
	cache_a = str(tmp_path / "cache_a")
	cache_b = str(tmp_path / "cache_b")

	ctx = mp.get_context("spawn")
	proc_a = ctx.Process(
		target=_run_worker_with_service,
		kwargs={
			"queue_path": queue_a,
			"listen_port": port_a,
			"announce_file": announce_a,
			"cache_dir": cache_a,
			"identity": "A",
		},
		daemon=True,
	)
	proc_b = ctx.Process(
		target=_run_worker_with_service,
		kwargs={
			"queue_path": queue_b,
			"listen_port": port_b,
			"announce_file": announce_b,
			"cache_dir": cache_b,
			"identity": "B",
		},
		daemon=True,
	)

	proc_a.start()
	proc_b.start()
	try:
		ann_a = _wait_for_announce(announce_a)
		ann_b = _wait_for_announce(announce_b)

		remote_a = RemoteQueue(peer_id=str(ann_a.get("peer_id") or ""), multiaddr=str(ann_a.get("multiaddr") or ""))
		remote_b = RemoteQueue(peer_id=str(ann_b.get("peer_id") or ""), multiaddr=str(ann_b.get("multiaddr") or ""))
		assert remote_a.peer_id and remote_b.peer_id
		assert "/p2p/" in remote_a.multiaddr
		assert "/p2p/" in remote_b.multiaddr

		# Cache write/read against B from the test process.
		set_resp = cache_set_sync(remote=remote_b, key="k", value={"from": "test"}, ttl_s=5.0, timeout_s=10.0)
		assert set_resp.get("ok") is True

		hit = cache_get_sync(remote=remote_b, key="k", timeout_s=10.0)
		assert hit.get("ok") is True
		assert hit.get("hit") is True
		assert hit.get("value") == {"from": "test"}

		# Submit a tool.call task to B; B's worker should execute it.
		task_id = submit_task_sync(
			remote=remote_b,
			task_type="tool.call",
			model_name="demo",
			payload={"tool": "unit_test.identity", "args": {"ping": True}},
		)
		assert isinstance(task_id, str) and task_id

		async def _wait() -> dict | None:
			return await wait_task(remote=remote_b, task_id=task_id, timeout_s=25.0)

		task = anyio.run(_wait, backend="trio")
		assert isinstance(task, dict)
		assert task.get("task_id") == task_id
		assert task.get("status") == "completed"
		result = task.get("result")
		assert isinstance(result, dict)

		inner = result.get("result")
		assert isinstance(inner, dict)
		assert inner.get("ok") is True
		assert inner.get("identity") == "B"
		assert inner.get("tool") == "unit_test.identity"
		assert inner.get("args") == {"ping": True}

		# Sanity: A is still reachable via status-like operations (cache miss ok).
		miss = cache_get_sync(remote=remote_a, key="nonexistent", timeout_s=10.0)
		assert miss.get("ok") is True
		assert miss.get("hit") is False
	finally:
		for p in (proc_a, proc_b):
			try:
				p.terminate()
			except Exception:
				pass
		for p in (proc_a, proc_b):
			try:
				p.join(timeout=5.0)
			except Exception:
				pass
