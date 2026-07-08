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


def _wait_for_json(path: str, timeout_s: float = 15.0) -> dict:
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		if os.path.exists(path) and os.path.getsize(path) > 0:
			try:
				with open(path, "r", encoding="utf-8") as handle:
					data = json.load(handle)
					if isinstance(data, dict):
						return data
			except Exception:
				pass
		time.sleep(0.1)
	raise TimeoutError(f"ready file not written: {path}")


def _run_dht_bootstrap_node(ready_file: str, listen_port: int) -> None:
	# Deterministic local-only behavior.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT"] = "0"

	from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible

	if not ensure_libp2p_compatible():
		raise RuntimeError("libp2p compatibility patches failed")

	import anyio
	import inspect

	from libp2p import new_host
	from libp2p.tools.async_service import background_trio_service
	from multiaddr import Multiaddr

	async def _main() -> None:
		host_obj = new_host()
		host = await host_obj if inspect.isawaitable(host_obj) else host_obj

		async with background_trio_service(host.get_network()):
			await host.get_network().listen(Multiaddr(f"/ip4/127.0.0.1/tcp/{int(listen_port)}"))

			# Start a KadDHT server so other nodes can bootstrap.
			dht = None
			candidates = [
				("libp2p.kad_dht.kad_dht", "KadDHT"),
				("libp2p.kad_dht", "KadDHT"),
			]
			for module_name, symbol in candidates:
				try:
					mod = __import__(module_name, fromlist=[symbol])
					cls = getattr(mod, symbol)
					try:
						from libp2p.kad_dht.kad_dht import DHTMode  # type: ignore

						dht = cls(host, DHTMode.SERVER)
					except Exception:
						dht = cls(host)

					start = getattr(dht, "start", None)
					if callable(start):
						maybe = start()
						if hasattr(maybe, "__await__"):
							await maybe
					bootstrap = getattr(dht, "bootstrap", None)
					if callable(bootstrap):
						maybe = bootstrap()
						if hasattr(maybe, "__await__"):
							await maybe
					break
				except Exception:
					continue

			if dht is None:
				raise RuntimeError("KadDHT unavailable; cannot run bootstrap node")

			try:
				import trio
				from libp2p.tools.async_service.trio_service import background_trio_service as bg_trio

				async def _run_dht() -> None:
					async with bg_trio(dht):
						await trio.sleep_forever()

				# Write ready file after DHT service is scheduled.
				peer_id = host.get_id().pretty()
				multiaddr = f"/ip4/127.0.0.1/tcp/{int(listen_port)}/p2p/{peer_id}"
				os.makedirs(os.path.dirname(ready_file) or ".", exist_ok=True)
				with open(ready_file, "w", encoding="utf-8") as handle:
					handle.write(json.dumps({"peer_id": peer_id, "multiaddr": multiaddr}, ensure_ascii=False))

				async with anyio.create_task_group() as tg:
					tg.start_soon(_run_dht)
					await anyio.sleep_forever()
			except Exception:
				raise

	anyio.run(_main, backend="trio")


def _run_taskqueue_service_with_dht(
	queue_path: str,
	listen_port: int,
	bootstrap_peer: str,
	cache_dir: str,
) -> None:
	# Force local-only network identity and deterministic discovery.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = "127.0.0.1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = str(bootstrap_peer)
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
	# Prevent test from relying on the announce-file hint.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT"] = "0"

	# Enable cache + tools to prove sharing works via discovery.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = cache_dir
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

	class _Accel:
		def call_tool(self, tool_name: str, args: dict):
			return {"ok": True, "tool": str(tool_name), "args": dict(args or {})}

	from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
	import anyio

	async def _main() -> None:
		await serve_task_queue(
			queue_path=queue_path,
			listen_port=int(listen_port),
			accelerate_instance=_Accel(),
		)

	anyio.run(_main, backend="trio")


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_dht_auto_discovery_allows_tools_cache_and_tasks(tmp_path, monkeypatch):
	"""E2E: discover TaskQueue peer via DHT (no multiaddr/announce), then use tools/cache/tasks."""
	from ipfs_accelerate_py.p2p_tasks.client import (
		RemoteQueue,
		discover_status_sync,
		cache_get_sync,
		cache_set_sync,
		submit_task_sync,
		list_tasks_sync,
		call_tool_sync,
	)

	bootstrap_port = _free_port()
	bootstrap_ready = str(tmp_path / "bootstrap_ready.json")

	service_port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	cache_dir = str(tmp_path / "p2p_cache")

	ctx = mp.get_context("spawn")
	boot_proc = ctx.Process(
		target=_run_dht_bootstrap_node,
		args=(bootstrap_ready, bootstrap_port),
		daemon=True,
	)
	boot_proc.start()

	boot = _wait_for_json(bootstrap_ready)
	bootstrap_ma = str(boot.get("multiaddr") or "").strip()
	assert "/p2p/" in bootstrap_ma

	service_proc = ctx.Process(
		target=_run_taskqueue_service_with_dht,
		args=(queue_path, service_port, bootstrap_ma, cache_dir),
		daemon=True,
	)
	service_proc.start()

	try:
		# Client discovery config: use the bootstrap peer, but no announce-file
		# and no peer multiaddr.
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS", bootstrap_ma)
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_DIAL", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST", "127.0.0.1")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_DHT", "1")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_MDNS", "0")

		remote = RemoteQueue(peer_id="", multiaddr="")

		# Give the service a moment to connect/bootstrap/provide.
		time.sleep(1.5)

		# Single discovery attempt: avoids creating multiple ephemeral peers that
		# can pollute the local DHT routing table.
		last = discover_status_sync(remote=remote, timeout_s=15.0, detail=True)

		assert isinstance(last, dict)
		assert last.get("ok") is True
		result = last.get("result")
		assert isinstance(result, dict)
		assert result.get("ok") is True

		# From here on, use the discovered multiaddr for stable RPCs.
		discovered_ma = ""
		for att in list(last.get("attempts") or []):
			if not isinstance(att, dict):
				continue
			if att.get("ok") is True and str(att.get("multiaddr") or "").strip():
				discovered_ma = str(att.get("multiaddr") or "").strip()
				break
		assert "/p2p/" in discovered_ma
		remote = RemoteQueue(peer_id=str(result.get("peer_id") or "").strip(), multiaddr=discovered_ma)
		assert remote.peer_id

		# Cache roundtrip after discovery (no preconfigured peer multiaddr).
		set_resp = cache_set_sync(remote=remote, key="k", value={"v": 1}, ttl_s=30.0, timeout_s=10.0)
		assert set_resp.get("ok") is True

		hit = cache_get_sync(remote=remote, key="k", timeout_s=10.0)
		assert hit.get("ok") is True
		assert hit.get("hit") is True
		assert hit.get("value") == {"v": 1}

		# Tool call after discovery (no preconfigured peer multiaddr).
		tool_resp = call_tool_sync(remote=remote, tool_name="unit_test.echo", args={"x": 1}, timeout_s=10.0)
		assert tool_resp.get("ok") is True
		assert tool_resp.get("tool") == "unit_test.echo"
		assert tool_resp.get("args") == {"x": 1}

		# Task submit/list after discovery (no preconfigured peer multiaddr).
		task_id = submit_task_sync(remote=remote, task_type="text-generation", model_name="demo", payload={"p": 1})
		assert isinstance(task_id, str) and task_id

		listed = list_tasks_sync(remote=remote, status="queued", limit=50)
		assert listed.get("ok") is True
		tasks = listed.get("tasks")
		assert isinstance(tasks, list)
		assert any(isinstance(t, dict) and t.get("task_id") == task_id for t in tasks)
	finally:
		service_proc.terminate()
		service_proc.join(timeout=5.0)
		boot_proc.terminate()
		boot_proc.join(timeout=5.0)
