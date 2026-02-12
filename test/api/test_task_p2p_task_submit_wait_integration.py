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


def _run_worker_with_service(queue_path: str, listen_port: int, announce_file: str) -> None:
	# Deterministic local-only behavior.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

	# Ensure the worker will claim tool.call tasks.
	os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "tool.call"

	class _Accel:
		def call_tool(self, tool_name: str, args: dict):
			return {"ok": True, "tool": str(tool_name), "args": dict(args or {})}

	from ipfs_accelerate_py.p2p_tasks.worker import run_worker

	run_worker(
		queue_path=queue_path,
		worker_id="unit-test-worker",
		poll_interval_s=0.05,
		once=False,
		p2p_service=True,
		p2p_listen_port=int(listen_port),
		accelerate_instance=_Accel(),
		supported_task_types=["tool.call"],
	)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_submit_and_wait_roundtrip(tmp_path):
	"""E2E: submit a task over P2P, have worker execute, wait via P2P."""
	import anyio

	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_task_sync, wait_task

	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_worker_with_service, args=(queue_path, port, announce_file), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		remote = RemoteQueue(peer_id=str(ann.get("peer_id") or ""), multiaddr=str(ann.get("multiaddr") or ""))
		assert remote.peer_id
		assert "/p2p/" in remote.multiaddr

		task_id = submit_task_sync(
			remote=remote,
			task_type="tool.call",
			model_name="demo",
			payload={"tool": "unit_test.echo", "args": {"x": 1}},
		)
		assert isinstance(task_id, str) and task_id

		async def _wait() -> dict | None:
			return await wait_task(remote=remote, task_id=task_id, timeout_s=20.0)

		task = anyio.run(_wait, backend="trio")
		assert isinstance(task, dict)
		assert task.get("task_id") == task_id
		assert task.get("status") == "completed"
		result = task.get("result")
		assert isinstance(result, dict)
		# Worker handlers typically return a structured payload like:
		# {"tool": ..., "result": {"ok": True, ...}, "progress": {...}}
		inner = result.get("result")
		assert isinstance(inner, dict)
		assert inner.get("ok") is True
		assert inner.get("tool") == "unit_test.echo"
		assert inner.get("args") == {"x": 1}
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
