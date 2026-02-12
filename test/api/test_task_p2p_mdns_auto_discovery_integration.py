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


def _truthy(text: str | None) -> bool:
	return str(text or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _free_port() -> int:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("127.0.0.1", 0))
	port = int(s.getsockname()[1])
	s.close()
	return port


def _run_taskqueue_service_with_mdns(queue_path: str, listen_port: int, cache_dir: str) -> None:
	# Keep discovery local and deterministic.
	os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "auto"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = "0.0.0.0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT"] = str(int(listen_port))
	os.environ["IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT"] = str(int(listen_port))

	# Force mDNS-only discovery signal.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "1"
	# Prevent any local announce-file hints from being used.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = "0"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT"] = "0"

	# Enable cache so the status call proves the service is fully initialized.
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE"] = "1"
	os.environ["IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR"] = str(cache_dir)

	from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
	import anyio

	async def _main() -> None:
		await serve_task_queue(queue_path=queue_path, listen_port=int(listen_port), accelerate_instance=None)

	anyio.run(_main, backend="trio")


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
@pytest.mark.integration
@pytest.mark.flaky
def test_task_p2p_mdns_auto_discovery_status(tmp_path, monkeypatch):
	"""E2E: discover a TaskQueue peer via mDNS (no multiaddr/announce/DHT/rendezvous).

	Notes:
	- mDNS requires multicast; some CI/container environments block it.
	- This test is opt-in to avoid flakiness.
	"""
	if not _truthy(os.environ.get("IPFS_ACCELERATE_PY_RUN_MDNS_TESTS")):
		pytest.skip("Set IPFS_ACCELERATE_PY_RUN_MDNS_TESTS=1 to run mDNS discovery integration test")

	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, discover_status_sync

	port = _free_port()
	queue_path = str(tmp_path / "task_queue.duckdb")
	cache_dir = str(tmp_path / "p2p_cache")

	ctx = mp.get_context("spawn")
	proc = ctx.Process(
		target=_run_taskqueue_service_with_mdns,
		args=(queue_path, port, cache_dir),
		daemon=True,
	)
	proc.start()

	try:
		# Client: force the discovery chain to end at mDNS.
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_DHT", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS", "0")
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_MDNS", "1")
		# Client and service must share the same mDNS UDP port.
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT", str(int(port)))
		monkeypatch.setenv("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", str(int(port)))
		monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST", "0.0.0.0")

		# Give mDNS announcer a moment to come up.
		time.sleep(1.5)

		remote = RemoteQueue(peer_id="", multiaddr="")
		trace = discover_status_sync(remote=remote, timeout_s=15.0, detail=True)
		assert isinstance(trace, dict)
		assert trace.get("ok") is True
		result = trace.get("result")
		assert isinstance(result, dict)
		assert result.get("ok") is True

		attempts = [a for a in (trace.get("attempts") or []) if isinstance(a, dict)]
		ok_attempts = [a for a in attempts if a.get("ok") is True]
		assert ok_attempts, f"no successful discovery attempts: {attempts}"
		assert ok_attempts[0].get("method") == "mdns"
		assert "/p2p/" in str(ok_attempts[0].get("multiaddr") or "")
	finally:
		proc.terminate()
		proc.join(timeout=5.0)


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
@pytest.mark.integration
@pytest.mark.flaky
def test_task_p2p_mdns_discovers_existing_peer_on_lan(monkeypatch):
	"""LAN E2E: discover a peer via mDNS without spinning up a local service.

	This is designed for the real two-box workflow:
	- Box A and Box B both run the systemd service (TaskQueue service + worker).
	- Run this test on Box B to validate that it can discover Box A via mDNS.

	To enable:
	- Set IPFS_ACCELERATE_PY_RUN_MDNS_LAN_TESTS=1
	- Ensure IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT matches the service port (systemd default: 9100)
	"""
	if not _truthy(os.environ.get("IPFS_ACCELERATE_PY_RUN_MDNS_LAN_TESTS")):
		pytest.skip("Set IPFS_ACCELERATE_PY_RUN_MDNS_LAN_TESTS=1 to run LAN mDNS discovery test")

	port = (
		os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT")
		or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT")
		or ""
	).strip()
	if not port:
		pytest.skip("Set IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT (e.g. 9100) for LAN mDNS test")

	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, discover_status_sync

	# Force the discovery chain to end at mDNS, and avoid dialing the local
	# announce-file endpoint first.
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE", "0")
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS", "0")
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_DHT", "0")
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS", "0")
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_MDNS", "1")
	monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT", str(port))
	monkeypatch.setenv("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", str(port))

	remote = RemoteQueue(peer_id="", multiaddr="")
	trace = discover_status_sync(remote=remote, timeout_s=15.0, detail=True)
	assert isinstance(trace, dict)
	assert trace.get("ok") is True
	result = trace.get("result")
	assert isinstance(result, dict)
	assert result.get("ok") is True

	attempts = [a for a in (trace.get("attempts") or []) if isinstance(a, dict)]
	ok_attempts = [a for a in attempts if a.get("ok") is True]
	assert ok_attempts, f"no successful discovery attempts: {attempts}"
	assert ok_attempts[0].get("method") == "mdns"
	assert "/p2p/" in str(ok_attempts[0].get("multiaddr") or "")
