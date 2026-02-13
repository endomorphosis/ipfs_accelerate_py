import json
import logging
import os
import re
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import pytest


logger = logging.getLogger(__name__)


# This integration test is primarily used for manual validation on real
# multi-machine LANs. Default to running with the required opt-ins enabled,
# while still allowing callers/CI to override by explicitly setting env vars.
os.environ.setdefault("IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST", "1")
os.environ.setdefault("IPFS_ACCELERATE_PY_TEST_FORCE_CUDA", "1")
os.environ.setdefault("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER", "1")
os.environ.setdefault("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_TIMEOUT_S", "20")
os.environ.setdefault("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_LIMIT", "50")


def _have_libp2p() -> bool:
	try:
		import libp2p  # noqa: F401
		return True
	except Exception:
		return False


def _truthy(text: str | None) -> bool:
	return str(text or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_targets_from_file(path: str) -> list[dict[str, str]]:
	with open(path, "r", encoding="utf-8") as handle:
		data = json.load(handle)

	# Common formats:
	# - {"targets": [{"peer_id":..., "multiaddr":...}, ...]}
	# - [{"peer_id":..., "multiaddr":...}, ...]
	if isinstance(data, dict) and isinstance(data.get("targets"), list):
		items = data.get("targets")
	elif isinstance(data, list):
		items = data
	else:
		raise ValueError(f"Unrecognized targets file format: {path}")

	out: list[dict[str, str]] = []
	for item in items:
		if not isinstance(item, dict):
			continue
		peer_id = str(item.get("peer_id") or "").strip()
		multiaddr = str(item.get("multiaddr") or "").strip()
		if not peer_id and multiaddr and "/p2p/" in multiaddr:
			peer_id = str(multiaddr.split("/p2p/")[-1]).strip()
		if peer_id or multiaddr:
			out.append({"peer_id": peer_id, "multiaddr": multiaddr})
	return out


def _load_targets_from_env() -> list[dict[str, str]]:
	file_path = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_TARGETS_FILE") or "").strip()
	if file_path:
		return _load_targets_from_file(file_path)

	# Comma-separated multiaddrs.
	raw = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_TARGETS") or "").strip()
	if raw:
		items: list[dict[str, str]] = []
		for ma in [x.strip() for x in raw.split(",") if x.strip()]:
			pid = ""
			if "/p2p/" in ma:
				pid = str(ma.split("/p2p/")[-1]).strip()
			items.append({"peer_id": pid, "multiaddr": ma})
		return items

	return []


def _validate_targets_unique(*, targets: list[dict[str, str]]) -> None:
	"""Fail fast on common target misconfiguration.

	If callers pass two multiaddrs with the same `/p2p/<peer_id>`, the test can
	appear to only hit one peer (all failures attributed to the same peer_id).
	"""
	peer_to_endpoints: dict[str, set[str]] = {}
	for t in list(targets or []):
		if not isinstance(t, dict):
			continue
		pid = str(t.get("peer_id") or "").strip()
		ma = str(t.get("multiaddr") or "").strip()
		if not pid or not ma:
			continue
		peer_to_endpoints.setdefault(pid, set()).add(_endpoint_key_from_multiaddr(ma))

	dupes = {pid: sorted(eps) for pid, eps in peer_to_endpoints.items() if len(eps) > 1}
	if dupes:
		pytest.skip(
			"Misconfigured targets: duplicate peer_id across multiple endpoints. "
			"This usually means you copied the wrong `/p2p/<peer_id>` into one of the target multiaddrs. "
			f"duplicates={dupes}"
		)


def _classify_incompatibility(text: str) -> str | None:
	msg = (text or "").lower()
	if not msg:
		return None
	patterns: list[tuple[str, str]] = [
		("nvidia container toolkit missing", r"nvidia-container-cli|nvidia-container-runtime"),
		("docker no gpu driver", r"could not select device driver|no matching devices found"),
		("docker permission", r"permission denied|got permission denied"),
		("docker daemon unreachable", r"cannot connect to the docker daemon|is the docker daemon running"),
		("image pull/auth", r"pull access denied|unauthorized|not found|manifest unknown"),
	]
	for label, pat in patterns:
		if re.search(pat, msg):
			return label
	return None


def _extract_gpu_uuids(stdout: str) -> set[str]:
	"""Extract GPU UUIDs from `nvidia-smi` output.

	This provides a practical signal of which physical machine executed a task
	without relying on worker_id naming conventions.
	"""
	text = str(stdout or "")
	if not text:
		return set()
	# Typical line: "GPU 0: ... (UUID: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
	uuids = set(re.findall(r"UUID:\s*(GPU-[A-Za-z0-9\-]+)", text))
	return {u.strip() for u in uuids if str(u).strip()}


def _default_report_path() -> str:
	"""Default report path under the repo's state folder.

	This integration test is opt-in; when it runs, leaving a report artifact in
	`state/` helps operators audit peer compatibility across runs.
	"""
	repo_root = Path(__file__).resolve().parents[2]
	state_dir = repo_root / "state" / "p2p"
	state_dir.mkdir(parents=True, exist_ok=True)
	ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
	return str(state_dir / f"docker_execute_nvidia_smi_network_{ts}.json")


def _detect_outbound_ipv4() -> str:
	"""Best-effort non-loopback IPv4 for deciding whether a target is local."""
	try:
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		try:
			sock.connect(("8.8.8.8", 80))
			ip = str(sock.getsockname()[0] or "").strip()
		finally:
			try:
				sock.close()
			except Exception:
				pass
		if ip and not ip.startswith("127."):
			return ip
	except Exception:
		pass

	try:
		ip = str(socket.gethostbyname(socket.gethostname()) or "").strip()
		if ip and not ip.startswith("127."):
			return ip
	except Exception:
		pass

	return "127.0.0.1"


def _endpoint_key_from_multiaddr(ma: str) -> str:
	"""Best-effort endpoint key (ip:port) from a /ip4/.../tcp/... multiaddr."""
	text = str(ma or "")
	m = re.search(r"/ip4/([^/]+)/tcp/(\d+)", text)
	if m:
		return f"{m.group(1)}:{m.group(2)}"
	return text


def _filter_targets_by_status(
	*,
	targets: list[dict[str, str]],
	timeout_s: float,
	require_docker_worker: bool = False,
) -> list[dict[str, str]]:
	"""Drop stale/unreachable targets by attempting a quick TaskQueue `status` RPC.

	Also de-duplicates by endpoint (ip:port) and keeps the first candidate per
	endpoint that successfully responds.
	"""
	try:
		from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, discover_status_sync
	except Exception:
		return list(targets or [])

	by_endpoint: dict[str, list[dict[str, str]]] = {}
	for t in list(targets or []):
		if not isinstance(t, dict):
			continue
		ma = str(t.get("multiaddr") or "").strip()
		pid = str(t.get("peer_id") or "").strip()
		if not ma:
			continue
		key = _endpoint_key_from_multiaddr(ma)
		by_endpoint.setdefault(key, []).append({"peer_id": pid, "multiaddr": ma})

	filtered: list[dict[str, str]] = []
	seen_peer_ids: set[str] = set()
	for _endpoint, candidates in by_endpoint.items():
		chosen: dict[str, str] | None = None
		for c in candidates:
			ma = str(c.get("multiaddr") or "")
			pid = str(c.get("peer_id") or "")
			try:
				remote = RemoteQueue(peer_id=pid, multiaddr=ma)
				trace = discover_status_sync(remote=remote, timeout_s=float(timeout_s), detail=True)
				if not (isinstance(trace, dict) and trace.get("ok") and isinstance(trace.get("result"), dict)):
					continue
				result = trace.get("result") or {}
				resolved_pid = str((result.get("peer_id") or pid)).strip()
				if require_docker_worker:
					scheduler = result.get("scheduler") if isinstance(result.get("scheduler"), dict) else {}
					counts = scheduler.get("counts") if isinstance(scheduler.get("counts"), dict) else {}
					try:
						docker_workers = int(counts.get("docker_workers") or 0)
					except Exception:
						docker_workers = 0

					local_ok = False
					local = result.get("local_worker")
					if isinstance(local, dict):
						if bool(local.get("enabled")) and bool(local.get("docker_enabled")):
							sup = local.get("supported_task_types")
							if isinstance(sup, str):
								sup_list = [p.strip() for p in sup.split(",") if p.strip()]
							elif isinstance(sup, (list, tuple, set)):
								sup_list = [str(x).strip() for x in sup if str(x).strip()]
							else:
								sup_list = []
							if any(t.startswith("docker.") for t in sup_list) or ("docker.execute" in sup_list):
								local_ok = True

					if docker_workers <= 0 and not local_ok:
						continue
				chosen = {"peer_id": resolved_pid, "multiaddr": ma}
				break
			except Exception:
				continue
		if chosen and chosen.get("peer_id") and chosen["peer_id"] not in seen_peer_ids:
			seen_peer_ids.add(chosen["peer_id"])
			filtered.append(chosen)

	return filtered


def _load_local_announce(*, expected_port: int) -> dict[str, str]:
	repo_root = Path(__file__).resolve().parents[2]
	paths: list[Path] = []

	# Prefer explicit env-configured announce file when present.
	env_path = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE") or "").strip()
	if env_path:
		try:
			paths.append(Path(env_path))
		except Exception:
			pass

	# Repo-local files (common in dev).
	paths.extend(
		[
			repo_root / "state" / "p2p" / "task_p2p_announce.json",
			repo_root / "state" / "p2p" / "task_p2p_announce_mcp.json",
			repo_root / "state" / "task_p2p_announce.json",
			repo_root / "state" / "task_p2p_announce_mcp.json",
		]
	)

	# systemd-managed locations (used by the provided unit files).
	paths.extend(
		[
			Path("/var/cache/ipfs-accelerate/task_p2p_announce.json"),
			Path("/var/cache/ipfs-accelerate/task_p2p_announce_mcp.json"),
		]
	)

	def _port_matches(ma: str) -> bool:
		try:
			return f"/tcp/{int(expected_port)}/" in str(ma or "")
		except Exception:
			return False

	for path in paths:
		try:
			if not path.exists():
				continue
			data = json.loads(path.read_text(encoding="utf-8"))
			if not isinstance(data, dict):
				continue
			peer_id = str(data.get("peer_id") or "").strip()
			multiaddr = str(data.get("multiaddr") or "").strip()
			if peer_id and multiaddr and _port_matches(multiaddr):
				return {"peer_id": peer_id, "multiaddr": multiaddr}
		except Exception:
			continue

	return {}


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
def test_task_p2p_docker_execute_nvidia_smi_50x_across_network():
	"""Opt-in network integration test.

	Submits 50 `docker.execute` tasks across N peers (currently 2).

	Enable and configure via env:
	- IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST=1
	- IPFS_ACCELERATE_PY_TEST_P2P_TARGETS_FILE=/path/to/targets.json
	or IPFS_ACCELERATE_PY_TEST_P2P_TARGETS="/ip4/.../tcp/.../p2p/<pid>,..."
	- If you do not set targets, the test can auto-discover peers via mDNS.

	Optional tuning:
	- IPFS_ACCELERATE_PY_TEST_DOCKER_P2P_MODE=task|mcp (default: task)
	- IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER=1 (default: 1 when no targets provided)
	- IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_TIMEOUT_S (default: 6)
	- IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_LIMIT (default: 25)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_TASKS (default: 50)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_CONCURRENCY (default: 10)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_TIMEOUT_S (default: 180)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_IMAGE (default: nvidia/cuda:12.4.0-base-ubuntu22.04)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_GPUS (default: all)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_SUBMIT_SINGLE_TARGET=1 (submit all tasks to targets[0])
	- IPFS_ACCELERATE_PY_TEST_DOCKER_REPORT_PATH=/path/to/report.json
	- IPFS_ACCELERATE_PY_TEST_EXPECT_MESH_DISTRIBUTION=1
	(asserts >1 executor observed via GPU UUIDs; falls back to worker_id)
	- IPFS_ACCELERATE_PY_TEST_EXPECT_ALL_PEERS=1
	(asserts >=N executors observed, where N=len(targets); best-effort and may be
	flaky if concurrency/tasks are too low)
	"""

	if not _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST")):
		pytest.skip("Set IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST=1 to run")

	p2p_mode = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_P2P_MODE") or "task").strip().lower() or "task"
	# mDNS uses a generic `_p2p._udp.local` service type, so we filter by port to
	# avoid latching onto unrelated libp2p services on the LAN.
	explicit_port = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_P2P_PORT") or "").strip()
	if explicit_port:
		try:
			expected_port = int(explicit_port)
		except Exception:
			expected_port = 9710
	else:
		if p2p_mode == "mcp":
			port_raw = os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_PORT") or "9100"
		else:
			port_raw = (
				os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT")
				or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT")
				or "9710"
			)
		try:
			expected_port = int(str(port_raw).strip())
		except Exception:
			expected_port = 9100 if p2p_mode == "mcp" else 9710

	targets = _load_targets_from_env()
	targets_from_env = list(targets or [])
	if len(targets) < 2:
		autodiscover_default = "1" if not targets else "0"
		autodiscover = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER") or autodiscover_default)
		if autodiscover:
			try:
				from ipfs_accelerate_py.p2p_tasks.client import (
					discover_peers_via_dht_sync,
					discover_peers_via_mdns_sync,
					discover_peers_via_rendezvous_sync,
				)
			except Exception as exc:
				pytest.skip(f"mDNS autodiscovery unavailable: {exc}")

			try:
				timeout_s = float(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_TIMEOUT_S") or "6")
				limit = int(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_AUTODISCOVER_LIMIT") or "25")
			except Exception:
				timeout_s, limit = 6.0, 25

			local = _load_local_announce(expected_port=int(expected_port))
			local_ipv4 = _detect_outbound_ipv4()

			# py-libp2p's mDNS discovery in this repo expects the discovery port to
			# match the peer's libp2p listen port.
			prior_mdns_port = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MDNS_PORT")
			os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS_PORT"] = str(int(expected_port))
			try:
				discovered = discover_peers_via_mdns_sync(timeout_s=float(timeout_s), limit=int(limit), exclude_self=True)
			finally:
				if prior_mdns_port is None:
					os.environ.pop("IPFS_ACCELERATE_PY_TASK_P2P_MDNS_PORT", None)
				else:
					os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS_PORT"] = prior_mdns_port

			# If multicast is blocked or peer advertisements are missing (including
			# cases where mDNS only finds local/self peers), fall back to DHT/provider
			# discovery and rendezvous (when available).
			fallback: list[object] = []
			mdns_list = list(discovered or [])
			mdns_only_local = False
			try:
				mdns_only_local = bool(mdns_list) and all(
					f"/ip4/{local_ipv4}/" in str(getattr(rq, "multiaddr", "") or "")
					or "/ip4/127." in str(getattr(rq, "multiaddr", "") or "")
					for rq in mdns_list
				)
			except Exception:
				mdns_only_local = False

			if len(mdns_list) < 2 or mdns_only_local:
				try:
					fallback.extend(
						discover_peers_via_dht_sync(
							timeout_s=float(timeout_s),
							limit=int(limit),
							exclude_self=True,
						)
						or []
					)
				except Exception:
					pass
				try:
					fallback.extend(
						discover_peers_via_rendezvous_sync(
							timeout_s=float(timeout_s),
							limit=int(limit),
							exclude_self=True,
						)
						or []
					)
				except Exception:
					pass
			if fallback:
				discovered = list(discovered or []) + list(fallback or [])

			merged: list[dict[str, str]] = []
			seen: set[str] = set()

			if local.get("peer_id") and local.get("multiaddr"):
				merged.append({"peer_id": str(local["peer_id"]), "multiaddr": str(local["multiaddr"])})
				seen.add(str(local["peer_id"]))

			for rq in list(discovered or []):
				pid = str(getattr(rq, "peer_id", "") or "").strip()
				ma = str(getattr(rq, "multiaddr", "") or "").strip()
				if not pid or not ma:
					continue
				if pid in seen:
					continue
				# Filter to the expected service port for this test mode.
				if f"/tcp/{int(expected_port)}/" not in ma:
					continue
				seen.add(pid)
				merged.append({"peer_id": pid, "multiaddr": ma})

			targets = merged
		else:
			pytest.skip("Provide at least 2 targets via IPFS_ACCELERATE_PY_TEST_P2P_TARGETS[_FILE] or enable mDNS autodiscovery")

	# Keep a snapshot of the candidates prior to status filtering so the JSON
	# report can explain why the effective peer set is smaller.
	targets_before_status_filter = list(targets or [])

	# Drop stale/unreachable targets (common when services restart and peer IDs rotate).
	# Keep this quick so it doesn't dominate the runtime.
	try:
		status_timeout_s = float(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_STATUS_TIMEOUT_S") or "3")
	except Exception:
		status_timeout_s = 3.0
	require_docker_worker = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_P2P_REQUIRE_DOCKER_WORKER") or "1")
	targets = _filter_targets_by_status(
		targets=targets,
		timeout_s=float(status_timeout_s),
		require_docker_worker=bool(require_docker_worker),
	)
	_validate_targets_unique(targets=targets)

	expect_mesh = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_EXPECT_MESH_DISTRIBUTION"))
	expect_all = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_EXPECT_ALL_PEERS"))
	if expect_mesh and len(targets) < 2:
		pytest.skip(
			"Need at least 2 reachable TaskQueue peers to assert mesh distribution. "
			f"mode={p2p_mode} expected_port={expected_port} targets_found={len(targets)}. "
			"Ensure both machines are running the TaskQueue p2p service on the same port (default 9710)."
		)
	if expect_all and len(targets) < 2:
		pytest.skip("Need at least 2 reachable peers to assert all-peers distribution")

	if len(targets) < 1:
		pretty = ", ".join([str(t.get("multiaddr") or "") for t in (targets or []) if t])
		pytest.skip(
			"Need at least 1 target (post-discovery) to run. "
			f"mode={p2p_mode} expected_port={expected_port} targets_found={len(targets)} "
			f"targets=[{pretty}]"
		)

	try:
		tasks = int(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_TASKS") or "50")
		concurrency = int(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_CONCURRENCY") or "10")
		timeout_s = float(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_TIMEOUT_S") or "180")
	except Exception:
		tasks, concurrency, timeout_s = 50, 10, 180.0

	image = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_IMAGE") or "nvidia/cuda:12.4.0-base-ubuntu22.04").strip()
	gpus = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_GPUS") or "all").strip()
	submit_single = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_SUBMIT_SINGLE_TARGET"))
	report_path = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_REPORT_PATH") or "").strip() or _default_report_path()

	import anyio

	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_docker_hub_task, wait_task

	remotes = [RemoteQueue(peer_id=str(t.get("peer_id") or ""), multiaddr=str(t.get("multiaddr") or "")) for t in targets]

	results: list[dict[str, Any]] = [None for _ in range(tasks)]  # type: ignore[list-item]

	async def _run_one(i: int) -> None:
		remote = remotes[0] if submit_single else remotes[i % len(remotes)]
		try:
			task_id = await submit_docker_hub_task(
				remote=remote,
				image=image,
				command=["nvidia-smi", "-L"],
				gpus=gpus,
				stream_output=False,
			)
			task = await wait_task(remote=remote, task_id=str(task_id), timeout_s=float(timeout_s))
			results[i] = {
				"i": i,
				"peer_id": remote.peer_id,
				"multiaddr": remote.multiaddr,
				"task_id": task_id,
				"task": task,
			}
		except Exception as exc:  # noqa: BLE001
			results[i] = {
				"i": i,
				"peer_id": remote.peer_id,
				"multiaddr": remote.multiaddr,
				"error": str(exc),
				"error_type": type(exc).__name__,
				"error_detail": traceback.format_exc()[-16000:],
			}

	async def _run_all() -> None:
		sem = anyio.Semaphore(max(1, int(concurrency)))
		async with anyio.create_task_group() as tg:
			for i in range(tasks):
				await sem.acquire()
				tg.start_soon(_runner, tg, sem, i)

	async def _runner(tg: anyio.abc.TaskGroup, sem: anyio.Semaphore, i: int) -> None:
		try:
			await _run_one(i)
		finally:
			sem.release()

	anyio.run(_run_all, backend="trio")

	# Analyze + log.
	failures = 0
	worker_ids: set[str] = set()
	gpu_uuids: set[str] = set()
	failures_by_peer: dict[str, int] = {}
	outputs: list[dict[str, Any]] = []

	for r in results:
		if not isinstance(r, dict):
			failures += 1
			outputs.append(
				{
					"status": "missing_result",
					"error": "missing result dict",
					"classification": None,
					"stdout_tail": "",
					"stderr_tail": "",
					"gpu_uuids": [],
				}
			)
			continue
		pid = str(r.get("peer_id") or "")
		if r.get("error"):
			failures += 1
			failures_by_peer[pid] = failures_by_peer.get(pid, 0) + 1
			logger.error("docker.execute submit/wait failed peer_id=%s multiaddr=%s error=%s", pid, r.get("multiaddr"), r.get("error"))
			outputs.append(
				{
					"i": r.get("i"),
					"task_id": r.get("task_id"),
					"submitted_peer_id": pid,
					"submitted_multiaddr": r.get("multiaddr"),
					"status": "submit_or_wait_error",
					"error": r.get("error"),
					"error_type": r.get("error_type"),
					"error_detail": r.get("error_detail"),
					"classification": _classify_incompatibility(str(r.get("error") or "")),
					"gpu_uuids": [],
					"stdout_tail": "",
					"stderr_tail": "",
				}
			)
			continue

		task = r.get("task")
		if not isinstance(task, dict):
			failures += 1
			failures_by_peer[pid] = failures_by_peer.get(pid, 0) + 1
			logger.error("docker.execute invalid task response peer_id=%s multiaddr=%s task=%r", pid, r.get("multiaddr"), task)
			outputs.append(
				{
					"i": r.get("i"),
					"task_id": r.get("task_id"),
					"submitted_peer_id": pid,
					"submitted_multiaddr": r.get("multiaddr"),
					"status": "invalid_task_response",
					"error": "invalid task response",
					"classification": None,
					"gpu_uuids": [],
					"stdout_tail": "",
					"stderr_tail": "",
				}
			)
			continue

		status = str(task.get("status") or "")
		result = task.get("result")
		error = task.get("error")
		stdout = ""
		stderr = ""
		if isinstance(result, dict):
			progress = result.get("progress")
			if isinstance(progress, dict) and progress.get("worker_id"):
				worker_ids.add(str(progress.get("worker_id")))
			stdout = str(result.get("stdout") or "")
			stderr = str(result.get("stderr") or "")
			gpu_uuids |= _extract_gpu_uuids(stdout)
			exit_code = result.get("exit_code")
			success = result.get("success")
		else:
			exit_code = None
			success = None

		classification = None
		if status != "completed":
			classification = _classify_incompatibility("\n".join([str(error or ""), stderr]))

		outputs.append(
			{
				"i": r.get("i"),
				"task_id": r.get("task_id"),
				"submitted_peer_id": pid,
				"submitted_multiaddr": r.get("multiaddr"),
				"status": status,
				"error": error,
				"classification": classification,
				"success": success,
				"exit_code": exit_code,
				"gpu_uuids": sorted(_extract_gpu_uuids(stdout)),
				"stdout_tail": stdout[-2000:],
				"stderr_tail": stderr[-4000:],
			}
		)

		ok = status == "completed" and isinstance(result, dict) and bool(result.get("success")) and int(result.get("exit_code", -1)) == 0
		if not ok:
			failures += 1
			failures_by_peer[pid] = failures_by_peer.get(pid, 0) + 1
			stderr = ""
			stdout = ""
			exit_code = None
			success = None
			if isinstance(result, dict):
				stderr = str(result.get("stderr") or "")
				stdout = str(result.get("stdout") or "")
				exit_code = result.get("exit_code")
				success = result.get("success")

			classification = _classify_incompatibility("\n".join([str(error or ""), stderr]))
			logger.error(
				"docker.execute failed i=%s peer_id=%s worker_ids=%s gpu_uuids=%s status=%s success=%r exit_code=%r class=%s error=%r stderr_tail=%r stdout_tail=%r",
				r.get("i"),
				pid,
				sorted(worker_ids)[:5],
				sorted(gpu_uuids)[:5],
				status,
				success,
				exit_code,
				classification,
				error,
				stderr[-4000:],
				stdout[-2000:],
			)

	report = {
		"ok": failures == 0,
		"task_type": "docker.execute",
		"workload": {
			"image": image,
			"command": ["nvidia-smi", "-L"],
			"gpus": gpus,
		},
		"p2p": {
			"mode": p2p_mode,
			"expected_port": int(expected_port),
			"status_timeout_s": float(status_timeout_s),
			"require_docker_worker": bool(require_docker_worker),
		},
		"tasks": tasks,
		"concurrency": concurrency,
		"timeout_s": timeout_s,
		"submit_single": submit_single,
		"targets_from_env": targets_from_env,
		"targets_before_status_filter": targets_before_status_filter,
		"targets": targets,
		"failures": failures,
		"failures_by_peer": failures_by_peer,
		"worker_ids": sorted(worker_ids),
		"gpu_uuids": sorted(gpu_uuids),
		"outputs": outputs,
	}

	try:
		Path(report_path).parent.mkdir(parents=True, exist_ok=True)
		with open(report_path, "w", encoding="utf-8") as handle:
			json.dump(report, handle, indent=2, sort_keys=True)
		logger.info("docker.execute wrote report path=%s", report_path)
	except Exception as exc:  # noqa: BLE001
		logger.error("docker.execute failed to write report path=%s error=%s", report_path, exc)

	logger.info(
		"docker.execute summary tasks=%s peers=%s submit_single=%s failures=%s failures_by_peer=%s worker_ids=%s gpu_uuids=%s",
		tasks,
		len(remotes),
		submit_single,
		failures,
		failures_by_peer,
		sorted(worker_ids),
		sorted(gpu_uuids),
	)

	if expect_mesh:
		# Prefer GPU UUIDs as the executor identity signal. If stdout is missing,
		# fall back to worker_id diversity.
		if gpu_uuids:
			assert len(gpu_uuids) >= 2, f"Expected distribution (>=2 GPU UUIDs), saw {sorted(gpu_uuids)}"
		else:
			assert len(worker_ids) >= 2, f"Expected distribution (>=2 worker_ids), saw {sorted(worker_ids)}"

	if expect_all:
		# Stronger version of the above: require evidence that all targets executed
		# at least one task. This relies on either unique GPU UUIDs per host or
		# unique worker_id naming across peers.
		want = int(len(targets))
		if gpu_uuids:
			assert len(gpu_uuids) >= want, f"Expected >= {want} GPU UUIDs, saw {sorted(gpu_uuids)}"
		else:
			assert len(worker_ids) >= want, f"Expected >= {want} worker_ids, saw {sorted(worker_ids)}"

	assert failures == 0, f"docker.execute had {failures} failures (see logs for per-peer diagnostics)"
