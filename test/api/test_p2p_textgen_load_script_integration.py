import json
import os
import socket
import subprocess
import sys
import time
import multiprocessing as mp
from pathlib import Path

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


def _repo_root() -> Path:
	# .../ipfs_accelerate_py/test/api/<this file>
	return Path(__file__).resolve().parents[2]


def _run_worker_with_p2p_service(
	*,
	queue_path: str,
	listen_port: int,
	announce_file: str,
) -> None:
	os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

	# Ensure worker claims text-generation.
	os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "text-generation"

	class _Accel:
		def infer(self, model_name, data: dict, *, endpoint=None, endpoint_type=None):
			prompt = str((data or {}).get("prompt") or "")
			return {"generated_text": f"ok:{model_name}:{prompt[:16]}"}

	from ipfs_accelerate_py.p2p_tasks.worker import run_worker

	run_worker(
		queue_path=queue_path,
		worker_id="unit-test-worker:textgen",
		poll_interval_s=0.05,
		once=False,
		p2p_service=True,
		p2p_listen_port=int(listen_port),
		accelerate_instance=_Accel(),
		supported_task_types=["text-generation"],
	)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_queue_textgen_load_submits_and_waits(tmp_path):
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(
		target=_run_worker_with_p2p_service,
		kwargs={
			"queue_path": queue_path,
			"listen_port": port,
			"announce_file": announce_file,
		},
		daemon=True,
	)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		assert str(ann.get("peer_id") or "")

		script = str(_repo_root() / "scripts" / "queue_textgen_load.py")
		cmd = [
			sys.executable,
			script,
			"--announce-file",
			announce_file,
			"--count",
			"12",
			"--concurrency",
			"4",
			"--wait",
			"--timeout-s",
			"30",
			"--model",
			"gpt2",
			"--prompt",
			"unit test prompt",
		]
		p = subprocess.run(cmd, capture_output=True, text=True, check=False)
		assert p.returncode == 0, f"script failed rc={p.returncode}\nstdout={p.stdout}\nstderr={p.stderr}"
		out = (p.stdout or "").strip()
		assert out
		data = json.loads(out)
		assert data.get("ok") is True
		assert data.get("count") == 12
		assert data.get("wait") is True
		assert data.get("completed") == 12
		assert data.get("failed") == 0
		assert data.get("timed_out") == 0
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
