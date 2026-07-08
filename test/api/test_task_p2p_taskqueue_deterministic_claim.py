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


def _run_service(queue_path: str, listen_port: int, announce_file: str) -> None:
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
	# Ensure deterministic delegation enabled for this test.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DETERMINISTIC"] = "1"

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


def _expected_for_peer(tasks: list[dict], *, peer_id: str, peers: list[str], clock_hash: str) -> dict | None:
	from ipfs_accelerate_py.p2p_tasks.deterministic_scheduler import select_owner_peer, task_hash

	def _prio_key(t: dict) -> tuple[int, float]:
		payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
		try:
			pr = int(payload.get("priority") or 5)
		except Exception:
			pr = 5
		pr = max(1, min(10, pr))
		try:
			created = float(t.get("created_at") or 0.0)
		except Exception:
			created = 0.0
		return (10 - pr, created)

	candidates = sorted(tasks, key=_prio_key)
	for t in candidates:
		th = task_hash(task_id=str(t.get("task_id") or ""), task_type=str(t.get("task_type") or ""), model_name=str(t.get("model_name") or ""))
		owner = select_owner_peer(peer_ids=peers, clock_hash=clock_hash, task_hash_hex=th)
		if owner == peer_id:
			return t
	return None


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_deterministic_claim_and_visibility(tmp_path):
	from ipfs_accelerate_py.p2p_tasks.client import (
		RemoteQueue,
		heartbeat_sync,
		list_tasks_sync,
		submit_task_sync,
		claim_next_sync,
	)
	from ipfs_accelerate_py.p2p_tasks.deterministic_scheduler import MerkleClock

	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service, args=(queue_path, port, announce_file), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		remote = RemoteQueue(peer_id=str(ann.get("peer_id") or ""), multiaddr=str(ann.get("multiaddr") or ""))

		# Register peers (stable peer set, stable clock).
		peer_a = "peer-A"
		peer_b = "peer-B"
		heartbeat_sync(remote=remote, peer_id=peer_a)
		heartbeat_sync(remote=remote, peer_id=peer_b)

		# Submit tasks with distinct priorities so ordering is stable.
		for i, pr in enumerate([10, 9, 8, 7, 6, 5, 4, 3], start=1):
			submit_task_sync(
				remote=remote,
				task_type="unit-test",
				model_name="",
				payload={"i": i, "priority": pr},
			)

		queued = list_tasks_sync(remote=remote, status="queued", limit=100, task_types=["unit-test"]).get("tasks")
		assert isinstance(queued, list)
		assert len(queued) >= 8

		clock_hash = MerkleClock(node_id="taskqueue-service").get_hash()
		peers = sorted([peer_a, peer_b])
		exp_a = _expected_for_peer(queued, peer_id=peer_a, peers=peers, clock_hash=clock_hash)
		exp_b = _expected_for_peer(queued, peer_id=peer_b, peers=peers, clock_hash=clock_hash)

		got_a = claim_next_sync(remote=remote, worker_id="worker-a", peer_id=peer_a, supported_task_types=["unit-test"])
		got_b = claim_next_sync(remote=remote, worker_id="worker-b", peer_id=peer_b, supported_task_types=["unit-test"])

		if exp_a is None:
			assert got_a is None
		else:
			assert isinstance(got_a, dict)
			assert got_a.get("task_id") == exp_a.get("task_id")

		if exp_b is None:
			assert got_b is None
		else:
			assert isinstance(got_b, dict)
			assert got_b.get("task_id") == exp_b.get("task_id")

		if got_a is not None and got_b is not None:
			assert got_a.get("task_id") != got_b.get("task_id")

		# Visibility reflects claims (claimed tasks should no longer be queued).
		after = list_tasks_sync(remote=remote, status="queued", limit=100, task_types=["unit-test"]).get("tasks")
		assert isinstance(after, list)
		claimed_ids = {t.get("task_id") for t in [got_a, got_b] if isinstance(t, dict)}
		assert not claimed_ids.intersection({t.get("task_id") for t in after})
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
