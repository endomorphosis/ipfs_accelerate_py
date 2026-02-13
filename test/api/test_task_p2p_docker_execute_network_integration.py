import json
import logging
import os
import re
from typing import Any

import pytest


logger = logging.getLogger(__name__)


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

	Optional tuning:
	- IPFS_ACCELERATE_PY_TEST_DOCKER_TASKS (default: 50)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_CONCURRENCY (default: 10)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_TIMEOUT_S (default: 180)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_IMAGE (default: nvidia/cuda:12.4.0-base)
	- IPFS_ACCELERATE_PY_TEST_DOCKER_GPUS (default: all)
	- IPFS_ACCELERATE_PY_TEST_EXPECT_MESH_DISTRIBUTION=1 (asserts >1 worker_id observed)
	"""

	if not _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST")):
		pytest.skip("Set IPFS_ACCELERATE_PY_TEST_ENABLE_DOCKER_NETWORK_TEST=1 to run")

	targets = _load_targets_from_env()
	if len(targets) < 2:
		pytest.skip("Provide at least 2 targets via IPFS_ACCELERATE_PY_TEST_P2P_TARGETS[_FILE]")

	try:
		tasks = int(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_TASKS") or "50")
		concurrency = int(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_CONCURRENCY") or "10")
		timeout_s = float(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_TIMEOUT_S") or "180")
	except Exception:
		tasks, concurrency, timeout_s = 50, 10, 180.0

	image = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_IMAGE") or "nvidia/cuda:12.4.0-base").strip()
	gpus = str(os.environ.get("IPFS_ACCELERATE_PY_TEST_DOCKER_GPUS") or "all").strip()
	expect_mesh = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TEST_EXPECT_MESH_DISTRIBUTION"))

	import anyio

	from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_docker_hub_task, wait_task

	remotes = [RemoteQueue(peer_id=str(t.get("peer_id") or ""), multiaddr=str(t.get("multiaddr") or "")) for t in targets]

	results: list[dict[str, Any]] = [None for _ in range(tasks)]  # type: ignore[list-item]

	async def _run_one(i: int) -> None:
		remote = remotes[i % len(remotes)]
		try:
			task_id = await submit_docker_hub_task(
				remote=remote,
				image=image,
				command=["nvidia-smi"],
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
	failures_by_peer: dict[str, int] = {}

	for r in results:
		if not isinstance(r, dict):
			failures += 1
			continue
		pid = str(r.get("peer_id") or "")
		if r.get("error"):
			failures += 1
			failures_by_peer[pid] = failures_by_peer.get(pid, 0) + 1
			logger.error("docker.execute submit/wait failed peer_id=%s multiaddr=%s error=%s", pid, r.get("multiaddr"), r.get("error"))
			continue

		task = r.get("task")
		if not isinstance(task, dict):
			failures += 1
			failures_by_peer[pid] = failures_by_peer.get(pid, 0) + 1
			logger.error("docker.execute invalid task response peer_id=%s multiaddr=%s task=%r", pid, r.get("multiaddr"), task)
			continue

		status = str(task.get("status") or "")
		result = task.get("result")
		error = task.get("error")
		if isinstance(result, dict):
			progress = result.get("progress")
			if isinstance(progress, dict) and progress.get("worker_id"):
				worker_ids.add(str(progress.get("worker_id")))

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
				"docker.execute failed i=%s peer_id=%s worker_ids=%s status=%s success=%r exit_code=%r class=%s error=%r stderr_tail=%r stdout_tail=%r",
				r.get("i"),
				pid,
				sorted(worker_ids)[:5],
				status,
				success,
				exit_code,
				classification,
				error,
				stderr[-4000:],
				stdout[-2000:],
			)

	logger.info(
		"docker.execute summary tasks=%s peers=%s failures=%s failures_by_peer=%s worker_ids=%s",
		tasks,
		len(remotes),
		failures,
		failures_by_peer,
		sorted(worker_ids),
	)

	if expect_mesh:
		assert len(worker_ids) >= 2, f"Expected mesh distribution (>=2 worker_ids), saw {sorted(worker_ids)}"

	assert failures == 0, f"docker.execute had {failures} failures (see logs for per-peer diagnostics)"
