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


def _repo_root() -> Path:
	# .../ipfs_accelerate_py/test/api/<this file>
	return Path(__file__).resolve().parents[2]


def _p2p_rpc_script() -> str:
	return str(_repo_root() / "scripts" / "p2p_rpc.py")


def _run_service_cache(queue_path: str, listen_port: int, announce_file: str, cache_dir: str) -> None:
	# Deterministic local-only behavior.
	os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir

	from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
	import anyio

	async def _main() -> None:
		await serve_task_queue(queue_path=queue_path, listen_port=listen_port, accelerate_instance=None)

	anyio.run(_main, backend="trio")


def _run_service_with_mcp_tools(queue_path: str, listen_port: int, announce_file: str) -> None:
	# Deterministic local-only behavior.
	os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

	# Create MCP server instance (registers tools + sets global instance).
	from ipfs_accelerate_py.mcp.server import create_mcp_server

	_ = create_mcp_server(accelerate_instance=None)

	from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
	import anyio

	async def _main() -> None:
		await serve_task_queue(queue_path=queue_path, listen_port=listen_port, accelerate_instance=None)

	anyio.run(_main, backend="trio")


def _run_worker_with_service(queue_path: str, listen_port: int, announce_file: str) -> None:
	# Deterministic local-only behavior.
	os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
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


def _run_cli_json(args: list[str], *, env: dict | None = None) -> dict:
	proc = subprocess.run(
		[sys.executable, _p2p_rpc_script(), *args],
		env=env,
		capture_output=True,
		text=True,
		check=False,
	)
	assert proc.returncode == 0, f"cli failed (rc={proc.returncode})\nstdout={proc.stdout}\nstderr={proc.stderr}"
	out = (proc.stdout or "").strip()
	assert out, f"no stdout\nstderr={proc.stderr}"
	return json.loads(out)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_p2p_rpc_cli_status_and_cache_roundtrip(tmp_path):
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")
	cache_dir = str(tmp_path / "p2p_cache")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service_cache, args=(queue_path, port, announce_file, cache_dir), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		assert str(ann.get("peer_id") or "")

		status = _run_cli_json(["--announce-file", announce_file, "status"])
		assert status.get("ok") is True
		assert status.get("peer_id")

		set_resp = _run_cli_json(
			["--announce-file", announce_file, "cache-set", "--key", "demo", "--value", '{"a": 1}']
		)
		assert set_resp.get("ok") is True

		get_resp = _run_cli_json(["--announce-file", announce_file, "cache-get", "--key", "demo"])
		assert get_resp.get("ok") is True
		assert get_resp.get("hit") is True
		assert get_resp.get("value") == {"a": 1}

		del_resp = _run_cli_json(["--announce-file", announce_file, "cache-delete", "--key", "demo"])
		assert del_resp.get("ok") is True

		has_resp = _run_cli_json(["--announce-file", announce_file, "cache-has", "--key", "demo"])
		assert has_resp.get("ok") is True
		assert has_resp.get("hit") is False
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_p2p_rpc_cli_call_tool_get_server_status(tmp_path):
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service_with_mcp_tools, args=(queue_path, port, announce_file), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		assert str(ann.get("peer_id") or "")

		resp = _run_cli_json(
			[
				"--announce-file",
				announce_file,
				"call-tool",
				"--tool",
				"get_server_status",
				"--args",
				"{}",
			]
		)
		assert resp.get("ok") is True
		assert isinstance(resp.get("result"), dict)
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_p2p_rpc_cli_task_submit_and_wait_roundtrip(tmp_path):
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_worker_with_service, args=(queue_path, port, announce_file), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		assert str(ann.get("peer_id") or "")

		sub = _run_cli_json(
			[
				"--announce-file",
				announce_file,
				"task-submit",
				"--task-type",
				"tool.call",
				"--model-name",
				"demo",
				"--payload",
				'{"tool":"unit_test.echo","args":{"x":1}}',
			]
		)
		assert sub.get("ok") is True
		task_id = str(sub.get("task_id") or "")
		assert task_id

		wait = _run_cli_json(
			[
				"--announce-file",
				announce_file,
				"task-wait",
				"--task-id",
				task_id,
				"--timeout",
				"20",
			]
		)
		assert wait.get("ok") is True
		res = wait.get("result")
		assert isinstance(res, dict)
		assert res.get("task_id") == task_id
		assert res.get("status") == "completed"

		stored = res.get("result")
		assert isinstance(stored, dict)
		inner = stored.get("result")
		assert isinstance(inner, dict)
		assert inner.get("ok") is True
		assert inner.get("tool") == "unit_test.echo"
		assert inner.get("args") == {"x": 1}
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_p2p_rpc_cli_announce_file_zero_falls_back_to_env(tmp_path):
	"""Regression: `--announce-file 0` should not open a file named '0'."""
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")
	cache_dir = str(tmp_path / "p2p_cache")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service_cache, args=(queue_path, port, announce_file, cache_dir), daemon=True)
	proc.start()
	try:
		_ = _wait_for_announce(announce_file)

		env = dict(os.environ)
		env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
		# If the CLI incorrectly treats '0' as a path, this would fail.
		resp = _run_cli_json(["--announce-file", "0", "discover", "--timeout", "5"], env=env)
		assert resp.get("ok") is True
		attempts = resp.get("attempts")
		assert isinstance(attempts, list)
		assert any(a.get("method") in {"explicit", "announce-file"} and a.get("ok") is True for a in attempts)
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_p2p_rpc_cli_discover_peer_id_hint_uses_env_announce_file(tmp_path):
	"""E2E: `discover` works with `--peer-id` hint and env announce-file.

	This matches the README's suggested flow for "no pre-shared multiaddr" cases:
	call `discover` with a peer-id hint, relying on discovery mechanisms.
	In this deterministic test we rely on the announce-file (via env var).
	"""
	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	announce_file = str(tmp_path / "announce.json")
	cache_dir = str(tmp_path / "p2p_cache")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(target=_run_service_cache, args=(queue_path, port, announce_file, cache_dir), daemon=True)
	proc.start()
	try:
		ann = _wait_for_announce(announce_file)
		peer_id = str(ann.get("peer_id") or "").strip()
		assert peer_id

		env = dict(os.environ)
		env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file

		resp = _run_cli_json(
			[
				"--announce-file",
				"0",
				"--peer-id",
				peer_id,
				"discover",
				"--timeout",
				"10",
				"--detail",
			],
			env=env,
		)
		assert resp.get("ok") is True
		attempts = resp.get("attempts")
		assert isinstance(attempts, list) and attempts
		assert any(a.get("method") in {"announce-file", "explicit", "announce-file:env"} and a.get("ok") is True for a in attempts)
	finally:
		proc.terminate()
		proc.join(timeout=5.0)
