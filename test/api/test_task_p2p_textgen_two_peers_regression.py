import json
import os
import socket
import sys
import time
import multiprocessing as mp
from pathlib import Path
import runpy

import pytest


def _have_libp2p() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("libp2p") is not None
    except Exception:
        return False


def _have_textgen_deps() -> bool:
    try:
        import importlib.util

        return (importlib.util.find_spec("transformers") is not None) and (importlib.util.find_spec("torch") is not None)
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_for_announce(path: str, timeout_s: float = 25.0) -> dict:
    deadline = time.time() + float(timeout_s)
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


def _run_textgen_worker_with_service(*, queue_path: str, listen_port: int, announce_file: str, worker_id: str) -> None:
    # Deterministic local-only behavior.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RELAY"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_HOLEPUNCH"] = "0"

    # Keep worker behavior stable and minimal.
    os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
    os.environ["IPFS_KIT_DISABLE"] = "1"
    os.environ["STORAGE_FORCE_LOCAL"] = "1"
    os.environ["TRANSFORMERS_PATCH_DISABLE"] = "1"

    os.environ["IPFS_ACCELERATE_PY_LLM_PROVIDER"] = "hf"
    os.environ["IPFS_ACCELERATE_PY_LLM_MODEL"] = "gpt2"

    # Ensure the worker will claim text-generation tasks.
    os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "text-generation"

    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    run_worker(
        queue_path=queue_path,
        worker_id=str(worker_id),
        poll_interval_s=0.05,
        once=False,
        p2p_service=True,
        p2p_listen_port=int(listen_port),
        accelerate_instance=None,
        supported_task_types=["text-generation"],
    )


@pytest.mark.skipif(not _have_libp2p(), reason="libp2p not installed")
@pytest.mark.skipif(not _have_textgen_deps(), reason="transformers/torch not installed")
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.text
def test_task_p2p_two_peers_textgen_regression_50(tmp_path: Path):
    """Regression: 2 peers, 50 GPT-2 generations, collect outputs.

    This test specifically guards against:
    - DuckDB attach/handle flakiness under concurrent P2P RPC access
    - non-durable reporting from the load driver (must write --output JSON)
    """

    port_a = _free_port()
    port_b = _free_port()
    assert port_a != port_b

    state_dir = tmp_path / "p2p_textgen_regression"
    state_dir.mkdir(parents=True, exist_ok=True)

    queue_a = str(state_dir / "peer1_queue.duckdb")
    queue_b = str(state_dir / "peer2_queue.duckdb")
    announce_a = str(state_dir / "peer1_announce.json")
    announce_b = str(state_dir / "peer2_announce.json")

    ctx = mp.get_context("spawn")
    proc_a = ctx.Process(
        target=_run_textgen_worker_with_service,
        kwargs={
            "queue_path": queue_a,
            "listen_port": port_a,
            "announce_file": announce_a,
            "worker_id": "peer1",
        },
        daemon=True,
    )
    proc_b = ctx.Process(
        target=_run_textgen_worker_with_service,
        kwargs={
            "queue_path": queue_b,
            "listen_port": port_b,
            "announce_file": announce_b,
            "worker_id": "peer2",
        },
        daemon=True,
    )

    proc_a.start()
    proc_b.start()

    try:
        ann_a = _wait_for_announce(announce_a)
        ann_b = _wait_for_announce(announce_b)

        report_path = str(state_dir / "load_report.json")

        script = Path(__file__).resolve().parents[2] / "scripts" / "queue_textgen_load.py"
        assert script.exists(), f"missing load driver: {script}"

        os.environ["IPFS_KIT_DISABLE"] = "1"
        os.environ["STORAGE_FORCE_LOCAL"] = "1"
        os.environ["TRANSFORMERS_PATCH_DISABLE"] = "1"
        os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"

        # Run the load driver in-process to avoid import shadowing issues
        # (the repo contains similarly-named modules under test/).
        mod = runpy.run_path(str(script))
        rc = int(mod["main"](
            [
                "--announce-file",
                announce_a,
                "--announce-file",
                announce_b,
                "--count",
                "50",
                "--concurrency",
                "10",
                "--wait",
                "--timeout-s",
                "300",
                "--collect-results",
                "--suffix-index",
                "--max-new-tokens",
                "16",
                "--temperature",
                "0.2",
                "--prompt",
                "The quick brown fox",
                "--submit-retries",
                "2",
                "--submit-retry-sleep-s",
                "0.25",
                "--output",
                report_path,
            ]
        ))
        assert rc == 0

        assert os.path.exists(report_path) and os.path.getsize(report_path) > 0
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)

        assert report.get("ok") is True
        assert int(report.get("count") or 0) == 50
        assert int(report.get("submit_ok_count") or 0) == 50
        assert int(report.get("submit_failed_count") or 0) == 0
        assert report.get("wait") is True
        assert int(report.get("completed") or 0) == 50
        assert int(report.get("failed") or 0) == 0
        assert int(report.get("timed_out") or 0) == 0

        outputs = report.get("outputs")
        assert isinstance(outputs, list)
        assert len(outputs) == 50

        # Validate distribution: the load driver round-robins by index, so we
        # expect an exact 25/25 split across the two targets.
        peer_counts: dict[str, int] = {}
        for item in outputs:
            assert isinstance(item, dict)
            assert item.get("status") == "completed"
            txt = item.get("text")
            assert isinstance(txt, str) and len(txt) > 0
            peer_id = str(item.get("peer_id") or "")
            peer_counts[peer_id] = peer_counts.get(peer_id, 0) + 1

        expected_peer_a = str(ann_a.get("peer_id") or "")
        expected_peer_b = str(ann_b.get("peer_id") or "")
        assert expected_peer_a and expected_peer_b
        assert peer_counts.get(expected_peer_a, 0) == 25
        assert peer_counts.get(expected_peer_b, 0) == 25
    finally:
        for p in (proc_a, proc_b):
            try:
                p.terminate()
            except Exception:
                pass
        for p in (proc_a, proc_b):
            try:
                p.join(timeout=10.0)
            except Exception:
                pass
