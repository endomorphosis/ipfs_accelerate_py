#!/usr/bin/env python3
"""Two-machine LAN demo: start a TaskQueue peer and run textgen load.

Run this same command on TWO machines on the same LAN.

What it does:
1) Starts a local TaskQueue worker+service (libp2p) listening on 0.0.0.0.
2) Waits for its announce file.
3) Discovers the other peer via mDNS.
4) Runs a bounded-concurrency text-generation load split across BOTH peers.

Defaults are chosen so that if you run it on both machines at once you get
~50 total tasks across the LAN (25 from each machine).

Example (run on both machines):
  ./.venv/bin/python scripts/lan_mdns_two_machine_textgen.py --count 25

Notes:
- Requires multicast/mDNS to be allowed on the LAN.
- Uses explicit multiaddrs after discovery for stability.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import socket
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Tuple


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "peer"


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    return data if isinstance(data, dict) else {}


def _wait_for_file(path: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + max(0.1, float(timeout_s))
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for file: {path}")


def _tcp_port_from_multiaddr_text(multiaddr: str) -> int:
    text = str(multiaddr or "")
    if "/tcp/" not in text:
        return 0
    try:
        tail = text.split("/tcp/", 1)[1]
        port_text = tail.split("/", 1)[0]
        return int(port_text)
    except Exception:
        return 0


def _discover_other_peer_multiaddr(
    *,
    self_peer_id: str,
    expected_tcp_port: int,
    expected_session: str,
    timeout_s: float,
    poll_s: float = 0.25,
) -> Tuple[str, str]:
    """Return (peer_id, multiaddr) for a non-self peer discovered via mDNS.

    Strategy:
    - Prefer peers advertising the same TCP port as our local listen port.
    - Verify the peer responds to a `status` RPC before returning.
    """

    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, discover_peers_via_mdns_sync, request_status_sync

    deadline = time.time() + max(0.1, float(timeout_s))
    self_peer_id = str(self_peer_id or "").strip()
    if not self_peer_id:
        raise ValueError("self_peer_id is required")

    last_seen: list[Tuple[str, str]] = []
    while time.time() < deadline:
        peers = discover_peers_via_mdns_sync(timeout_s=1.0, limit=25, exclude_self=True)
        candidates: list[Tuple[str, str]] = []
        for p in list(peers or []):
            pid = str(getattr(p, "peer_id", "") or "").strip()
            ma = str(getattr(p, "multiaddr", "") or "").strip()
            if not pid or not ma:
                continue
            if pid == self_peer_id:
                continue
            candidates.append((pid, ma))

        # Prefer same port to avoid accidentally selecting unrelated LAN peers.
        preferred = [c for c in candidates if _tcp_port_from_multiaddr_text(c[1]) == int(expected_tcp_port)]
        ordered = preferred + [c for c in candidates if c not in preferred]
        last_seen = ordered

        for pid, ma in ordered:
            if int(expected_tcp_port) and _tcp_port_from_multiaddr_text(ma) != int(expected_tcp_port):
                continue
            try:
                resp = request_status_sync(remote=RemoteQueue(peer_id=pid, multiaddr=ma), timeout_s=5.0, detail=False)
                if not (isinstance(resp, dict) and resp.get("ok")):
                    continue
                if expected_session:
                    remote_session = str(resp.get("session") or "").strip()
                    if remote_session != expected_session:
                        continue
                    return pid, ma
            except Exception:
                continue

        time.sleep(max(0.05, float(poll_s)))

    if last_seen:
        # Give a useful hint when we saw peers but none matched/answered.
        seen = ", ".join([f"{pid} ({ma})" for pid, ma in last_seen[:5]])
        raise RuntimeError(
            "mDNS discovery timed out: saw peers but none responded on the expected port. "
            f"expected_tcp_port={int(expected_tcp_port)}; seen={seen}"
        )
    raise RuntimeError("mDNS discovery timed out: no non-self peers found")


def _run_worker_process(
    *, queue_path: str, listen_host: str, listen_port: int, announce_file: str, worker_id: str, public_ip: str
) -> None:
    # Keep this process minimal and LAN-reachable.
    os.environ["IPFS_KIT_DISABLE"] = "1"
    os.environ["STORAGE_FORCE_LOCAL"] = "1"
    os.environ["TRANSFORMERS_PATCH_DISABLE"] = "1"
    os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"

    # Networking: listen on LAN, announce detected LAN IP.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = str(listen_host or "0.0.0.0")
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT"] = str(int(listen_port))
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = str(public_ip or "auto")
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = str(announce_file)

    # Discovery: mDNS enabled; disable cross-subnet mechanisms by default.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "1"

    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    run_worker(
        queue_path=str(queue_path),
        worker_id=str(worker_id),
        poll_interval_s=0.05,
        p2p_service=True,
        p2p_listen_port=int(listen_port),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a LAN peer and run mDNS-discovered textgen load")
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="IP/interface to bind the local TaskQueue peer (default: 0.0.0.0).",
    )
    parser.add_argument("--listen-port", type=int, default=9710, help="TCP listen port for the local TaskQueue peer")
    parser.add_argument(
        "--public-ip",
        default="auto",
        help="IP to advertise in the service multiaddr (default: auto). Use this if auto picks loopback.",
    )
    parser.add_argument("--state-dir", default="state/lan_mdns_two_machine", help="Directory for queue/announce/report files")
    parser.add_argument("--worker-id", default="", help="Worker id (default: hostname)")
    parser.add_argument(
        "--session",
        default="lan-mdns-two-machine",
        help="Session tag used to filter mDNS-discovered peers (default: lan-mdns-two-machine).",
    )

    parser.add_argument("--count", type=int, default=25, help="Tasks to submit from THIS machine")
    parser.add_argument("--concurrency", type=int, default=10, help="Submit/wait concurrency")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Per-task wait timeout")

    parser.add_argument("--prompt", default="The quick brown fox", help="Prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--mdns-timeout-s", type=float, default=30.0, help="How long to wait for mDNS peer discovery")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Keep the parent process minimal too (prevents noisy optional subsystem
    # initialization when importing ipfs_accelerate_py modules).
    os.environ.setdefault("IPFS_KIT_DISABLE", "1")
    os.environ.setdefault("STORAGE_FORCE_LOCAL", "1")
    os.environ.setdefault("TRANSFORMERS_PATCH_DISABLE", "1")
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")

    state_dir = Path(str(args.state_dir))
    state_dir.mkdir(parents=True, exist_ok=True)

    worker_id = str(args.worker_id or "").strip() or _hostname()
    queue_path = str(state_dir / "queue.duckdb")
    announce_file = str(state_dir / "announce.json")
    report_path = str(state_dir / f"load_report_{worker_id}.json")

    # Session tag: used to ensure mDNS discovery finds the *other* peer running
    # this script, not any random libp2p node on the LAN.
    session = str(args.session or "").strip()
    if session:
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = session

    ctx = get_context("spawn")
    proc = ctx.Process(
        target=_run_worker_process,
        kwargs={
            "queue_path": queue_path,
            "listen_host": str(args.listen_host),
            "listen_port": int(args.listen_port),
            "announce_file": announce_file,
            "worker_id": worker_id,
            "public_ip": str(args.public_ip),
        },
        daemon=True,
    )
    proc.start()

    try:
        _wait_for_file(announce_file, timeout_s=30.0)
        ann = _read_json(announce_file)
        self_peer_id = str(ann.get("peer_id") or "").strip()
        self_ma = str(ann.get("multiaddr") or "").strip()
        if not self_peer_id or not self_ma:
            raise RuntimeError(f"invalid announce file: {announce_file}")

        # Client env: keep discovery focused on mDNS.
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
        # Align mDNS port with the service listen port.
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT"] = str(int(args.listen_port))
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "1"
        if session:
            os.environ["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = session

        # Discover the other peer via mDNS and verify it answers a `status` RPC.
        expected_port = int(args.listen_port)
        other_peer_id, other_multiaddr = _discover_other_peer_multiaddr(
            self_peer_id=self_peer_id,
            expected_tcp_port=expected_port,
            expected_session=session,
            timeout_s=float(args.mdns_timeout_s),
        )

        # Run load across BOTH peers using explicit multiaddrs for stability.
        script = Path(__file__).resolve().parents[1] / "scripts" / "queue_textgen_load.py"
        if not script.exists():
            # When executed as a module or from repo root.
            script = Path(__file__).resolve().parents[0] / "queue_textgen_load.py"
        if not script.exists():
            raise RuntimeError("cannot locate scripts/queue_textgen_load.py")

        mod = runpy.run_path(str(script))
        rc = int(
            mod["main"](
                [
                    "--multiaddr",
                    self_ma,
                    "--multiaddr",
                    str(other_multiaddr),
                    "--count",
                    str(int(args.count)),
                    "--concurrency",
                    str(int(args.concurrency)),
                    "--wait",
                    "--timeout-s",
                    str(float(args.timeout_s)),
                    "--collect-results",
                    "--suffix-index",
                    "--max-new-tokens",
                    str(int(args.max_new_tokens)),
                    "--temperature",
                    str(float(args.temperature)),
                    "--prompt",
                    str(args.prompt),
                    "--submit-retries",
                    "2",
                    "--submit-retry-sleep-s",
                    "0.25",
                    "--output",
                    report_path,
                ]
            )
        )
        return rc
    except Exception as exc:
        # Always write a report file so operators can find outputs even when
        # discovery fails.
        tmp = f"{report_path}.tmp"
        payload = {
            "ok": False,
            "error": str(exc),
            "worker_id": worker_id,
            "state_dir": str(state_dir),
            "listen_host": str(args.listen_host),
            "listen_port": int(args.listen_port),
            "public_ip": str(args.public_ip),
            "session": session,
        }
        try:
            Path(tmp).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, report_path)
        except Exception:
            pass
        raise
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=10.0)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
