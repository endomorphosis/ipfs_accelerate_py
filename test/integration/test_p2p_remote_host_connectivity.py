#!/usr/bin/env python3
"""Remote host libp2p connectivity probe.

Goal
----
When you run P2P on two different laptops, it can be hard to *prove* you're
connected to a distinct remote host vs. reusing the same host object from
multiple threads.

This module provides:
- A small probe server (run on laptop A) that listens on a TCP port via libp2p
  and responds on a dedicated probe protocol with its `peer_id`, `hostname`,
  and `pid`.
- A pytest test (run on laptop B) that dials the server's multiaddr, opens a
  stream, and asserts the remote `peer_id` matches the multiaddr and is
  different from the local peer id.

Usage (two laptops)
-------------------
Laptop A (server):
    ./.venv/bin/python -m test.integration.test_p2p_remote_host_connectivity \
    --serve --port 9100 --advertise-ip <LAN_IP>

It prints a line like:
  P2P_PROBE_MULTIADDR=/ip4/<LAN_IP>/tcp/9100/p2p/<PEER_ID>

Laptop B (client test):
  P2P_REMOTE_MULTIADDR="/ip4/<LAN_IP>/tcp/9100/p2p/<PEER_ID>" \
        ./.venv/bin/python -m pytest -c test/pytest.ini -vv \
    test/integration/test_p2p_remote_host_connectivity.py::test_remote_host_probe

Notes
-----
- Ensure the server port is reachable (firewall/router).
- This test is skipped unless `P2P_REMOTE_MULTIADDR` is set.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


PROBE_PROTOCOL = "/ipfs-accelerate/probe/1.0.0"
DEFAULT_PORT = 9100


def _peer_id_pretty(peer_id: Any) -> str:
    if peer_id is None:
        return ""
    pretty = getattr(peer_id, "pretty", None)
    if callable(pretty):
        try:
            return str(pretty())
        except Exception:
            return str(peer_id)
    return str(peer_id)


def _guess_advertise_ip() -> str:
    # Best-effort LAN IP discovery (not an external "public" IP resolver).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class ProbeResponse:
    peer_id: str
    hostname: str
    pid: int
    ts: float


async def run_probe_server(
    *,
    port: int,
    advertise_ip: Optional[str],
    exit_after_one: bool,
    relay_hop: bool,
) -> None:
    import trio

    try:
        from libp2p import new_host
        from multiaddr import Multiaddr
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"libp2p deps missing: {e}")

    listen_ma = Multiaddr(f"/ip4/0.0.0.0/tcp/{port}")
    host = await _maybe_await(new_host())

    local_peer_id = _peer_id_pretty(host.get_id())
    advertise_ip = advertise_ip or os.environ.get("P2P_ADVERTISE_IP") or _guess_advertise_ip()

    done = trio.Event()

    async def _handler(stream) -> None:
        request_bytes = b""
        try:
            request_bytes = await stream.read(4096)
        except Exception:
            request_bytes = b""

        try:
            req = json.loads(request_bytes.decode("utf-8")) if request_bytes else {}
        except Exception:
            req = {"_raw": request_bytes[:200].decode("utf-8", errors="replace")}

        resp: Dict[str, Any] = {
            "peer_id": local_peer_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "ts": time.time(),
            "request": req,
        }

        await stream.write(json.dumps(resp, sort_keys=True).encode("utf-8"))
        await stream.close()

        if exit_after_one:
            done.set()

    host.set_stream_handler(PROBE_PROTOCOL, _handler)

    async with host.run([listen_ma]):
        # Optionally run this host as a circuit relay v2 HOP.
        # This is the key ingredient for NATed peers to be reachable without port-forwarding.
        if relay_hop:
            try:
                from libp2p.relay.circuit_v2 import CircuitV2Protocol

                relay_service = CircuitV2Protocol(host, allow_hop=True)

                async def _run_relay() -> None:
                    await relay_service.run()

                # Run relay service in the background.
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(_run_relay)
                    # Wait until the relay service has registered handlers.
                    with trio.fail_after(10):
                        await relay_service.event_started.wait()
                    # Now continue with normal probe lifecycle.
                    await _serve_loop(host, advertise_ip, port, local_peer_id, done, exit_after_one)
                    nursery.cancel_scope.cancel()
            except Exception as e:
                print(f"P2P_RELAY_HOP_ERROR={e}", flush=True)
                await _serve_loop(host, advertise_ip, port, local_peer_id, done, exit_after_one)
        else:
            await _serve_loop(host, advertise_ip, port, local_peer_id, done, exit_after_one)


async def _serve_loop(host, advertise_ip: str, port: int, local_peer_id: str, done, exit_after_one: bool) -> None:
    import trio

    # Compute the actual bound port from the host (in case of remapping).
    addrs = [str(a) for a in host.get_addrs()]
    bound_tcp_port = None
    for addr in host.get_addrs():
        bound_tcp_port = addr.value_for_protocol("tcp")
        if bound_tcp_port:
            break

    tcp_port = int(bound_tcp_port) if bound_tcp_port else int(port)
    advertised = f"/ip4/{advertise_ip}/tcp/{tcp_port}/p2p/{local_peer_id}"

    # Print an easy-to-copy env var assignment.
    print(f"P2P_PROBE_MULTIADDR={advertised}", flush=True)
    # Optional extra context for debugging.
    if addrs:
        print(f"P2P_PROBE_LISTENING={';'.join(addrs)}", flush=True)

    if exit_after_one:
        await done.wait()
    else:
        await trio.sleep_forever()

    try:
        await host.close()
    except Exception:
        pass


async def probe_remote(remote_multiaddr: str, timeout_s: float = 10.0) -> Tuple[str, str, ProbeResponse]:
    import trio

    try:
        from libp2p import new_host
        from libp2p.peer.peerinfo import info_from_p2p_addr
        from multiaddr import Multiaddr
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"libp2p deps missing: {e}")

    host = await _maybe_await(new_host())
    local_peer_id = _peer_id_pretty(host.get_id())

    ma = Multiaddr(remote_multiaddr)
    expected_remote_peer_id = ma.value_for_protocol("p2p")
    if not expected_remote_peer_id:
        raise ValueError("Remote multiaddr must include /p2p/<peer_id>")

    peer_info = info_from_p2p_addr(ma)

    # The host must be "running" (background swarm service) for dialing to work.
    async with host.run([Multiaddr("/ip4/0.0.0.0/tcp/0")]):
        with trio.fail_after(timeout_s):
            await host.connect(peer_info)

        with trio.fail_after(timeout_s):
            stream = await host.new_stream(peer_info.peer_id, [PROBE_PROTOCOL])

        req = {
            "from_peer_id": local_peer_id,
            "from_hostname": socket.gethostname(),
            "from_pid": os.getpid(),
            "ts": time.time(),
        }

        with trio.fail_after(timeout_s):
            await stream.write(json.dumps(req, sort_keys=True).encode("utf-8"))
        with trio.fail_after(timeout_s):
            raw = await stream.read(4096)
        await stream.close()

    try:
        await host.close()
    except Exception:
        pass

    payload = json.loads(raw.decode("utf-8")) if raw else {}
    resp = ProbeResponse(
        peer_id=str(payload.get("peer_id", "")),
        hostname=str(payload.get("hostname", "")),
        pid=int(payload.get("pid", -1)),
        ts=float(payload.get("ts", 0.0) or 0.0),
    )

    return local_peer_id, str(expected_remote_peer_id), resp


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Remote-host libp2p connectivity probe")
    parser.add_argument("--serve", action="store_true", help="Run probe server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--advertise-ip", default=None)
    parser.add_argument("--exit-after-one", action="store_true")
    parser.add_argument(
        "--relay-hop",
        action="store_true",
        help="Also run as a Circuit Relay v2 HOP (useful for a public relay node)",
    )

    args = parser.parse_args(argv)

    if args.serve:
        import trio

        async def _serve() -> None:
            await run_probe_server(
                port=args.port,
                advertise_ip=args.advertise_ip,
                exit_after_one=args.exit_after_one,
                relay_hop=args.relay_hop,
            )

        trio.run(_serve)
        return 0

    # Client mode: read from env.
    remote = os.environ.get("P2P_REMOTE_MULTIADDR", "").strip()
    if not remote:
        raise SystemExit("Set P2P_REMOTE_MULTIADDR or use --serve")

    import trio

    local_peer_id, expected_remote_peer_id, resp = trio.run(probe_remote, remote)
    print(f"local_peer_id={local_peer_id}")
    print(f"expected_remote_peer_id={expected_remote_peer_id}")
    print(f"remote_peer_id={resp.peer_id}")
    print(f"remote_hostname={resp.hostname}")
    print(f"remote_pid={resp.pid}")
    return 0


def test_remote_host_probe():
    """Ensure we can connect to a *remote* libp2p host.

    Skips unless P2P_REMOTE_MULTIADDR is set.

    This proves you're connected to a distinct host because:
    - the remote responds with its own peer id;
    - the peer id matches the /p2p/<peer_id> in the multiaddr;
    - the remote peer id is different from the local peer id.
    """

    import pytest

    remote = os.environ.get("P2P_REMOTE_MULTIADDR", "").strip()
    if not remote:
        pytest.skip("Set P2P_REMOTE_MULTIADDR from the server output")

    try:
        import trio

        local_peer_id, expected_remote_peer_id, resp = trio.run(probe_remote, remote)
    except Exception as e:
        pytest.fail(f"Remote probe failed: {e}")
        return

    assert resp.peer_id, "Remote did not return a peer_id"
    assert resp.peer_id == expected_remote_peer_id, (
        f"Remote peer_id mismatch: expected {expected_remote_peer_id}, got {resp.peer_id}"
    )
    assert resp.peer_id != local_peer_id, "Remote peer_id equals local peer_id (not a distinct host)"


if __name__ == "__main__":
    raise SystemExit(_main())
