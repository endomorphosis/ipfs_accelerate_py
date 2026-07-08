#!/usr/bin/env python3
"""Optional networked MCP+p2p transport integration tests."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import anyio

from ipfs_accelerate_py.p2p_tasks.mcp_p2p import PROTOCOL_MCP_P2P_V1
from ipfs_accelerate_py.p2p_tasks.mcp_p2p_client import (
    MCPP2PClient,
    open_libp2p_stream_by_multiaddr,
    trio_libp2p_host_listen,
)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@unittest.skipUnless(importlib.util.find_spec("libp2p") is not None, "libp2p not installed")
class TestMCPTransportTrioMCPP2PNetworked(unittest.TestCase):
    """Networked MCP+p2p checks for live initialize handshake semantics."""

    def test_initialize_advertises_effective_limits_over_network(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mcp_transport_mcp_p2p_networked_") as td:
            queue_path = os.path.join(td, "queue.json")
            announce_file = os.path.join(td, "task_p2p_announce.json")
            port = _pick_free_port()

            env = dict(os.environ)
            env.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
            env["PYTHONUNBUFFERED"] = "1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
            env["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
            env["IPFS_ACCELERATE_PY_MCP_P2P_MAX_FRAME_BYTES"] = "4096"
            env["IPFS_ACCELERATE_PY_MCP_P2P_MAX_FRAMES"] = "55"
            env["IPFS_ACCELERATE_PY_MCP_P2P_RATE_CAPACITY"] = "9"
            env["IPFS_ACCELERATE_PY_MCP_P2P_RATE_REFILL_PER_SEC"] = "3.25"

            code = (
                "import os, anyio, functools; "
                "from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue; "
                "fn = functools.partial(serve_task_queue, queue_path=os.environ['Q'], listen_port=int(os.environ['P'])); "
                "anyio.run(fn, backend='trio')"
            )

            env["Q"] = queue_path
            env["P"] = str(port)

            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                deadline = time.time() + 30.0
                while time.time() < deadline and not os.path.exists(announce_file):
                    if proc.poll() is not None:
                        out = proc.stdout.read() if proc.stdout else ""
                        self.fail(f"service exited early: {proc.returncode}\n{out}")
                    time.sleep(0.05)

                self.assertTrue(os.path.exists(announce_file), "announce file not created")
                info = json.loads(open(announce_file, "r", encoding="utf-8").read())
                multiaddr = str(info.get("multiaddr") or "")
                self.assertIn("/p2p/", multiaddr)

                async def _run_client() -> dict:
                    async with trio_libp2p_host_listen(listen_multiaddr="/ip4/127.0.0.1/tcp/0") as host:
                        stream = await open_libp2p_stream_by_multiaddr(
                            host,
                            peer_multiaddr=multiaddr,
                            protocols=[PROTOCOL_MCP_P2P_V1],
                        )
                        client = MCPP2PClient(stream=stream, max_frame_bytes=1024 * 1024)
                        response = await client.initialize({})
                        await client.aclose()
                        return response

                init_response = anyio.run(_run_client, backend="trio")
                result = init_response.get("result", {})
                self.assertTrue(result.get("ok"))
                self.assertEqual(result.get("transport"), PROTOCOL_MCP_P2P_V1)

                limits = result.get("limits", {})
                self.assertEqual(limits.get("max_frame_bytes"), 4096)
                self.assertEqual(limits.get("max_frames"), 55)
                self.assertEqual(limits.get("rate_capacity"), 9)
                self.assertEqual(limits.get("rate_refill_per_sec"), 3.25)

                negotiation = result.get("profile_negotiation", {})
                self.assertTrue(negotiation.get("supports_profile_negotiation"))
                self.assertEqual(negotiation.get("mode"), "optional_additive")
                profiles = negotiation.get("profiles", [])
                self.assertIsInstance(profiles, list)
                self.assertIn("mcp++/profile-e-mcp-p2p", profiles)
                self.assertIn(result.get("active_profile"), profiles)

            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def test_mixed_version_protocol_list_and_profile_negotiation_over_network(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mcp_transport_mcp_p2p_networked_mixed_") as td:
            queue_path = os.path.join(td, "queue.json")
            announce_file = os.path.join(td, "task_p2p_announce.json")
            port = _pick_free_port()

            env = dict(os.environ)
            env.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
            env["PYTHONUNBUFFERED"] = "1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
            env["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

            code = (
                "import os, anyio, functools; "
                "from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue; "
                "fn = functools.partial(serve_task_queue, queue_path=os.environ['Q'], listen_port=int(os.environ['P'])); "
                "anyio.run(fn, backend='trio')"
            )

            env["Q"] = queue_path
            env["P"] = str(port)

            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                deadline = time.time() + 30.0
                while time.time() < deadline and not os.path.exists(announce_file):
                    if proc.poll() is not None:
                        out = proc.stdout.read() if proc.stdout else ""
                        self.fail(f"service exited early: {proc.returncode}\n{out}")
                    time.sleep(0.05)

                self.assertTrue(os.path.exists(announce_file), "announce file not created")
                info = json.loads(open(announce_file, "r", encoding="utf-8").read())
                multiaddr = str(info.get("multiaddr") or "")
                self.assertIn("/p2p/", multiaddr)

                async def _run_client() -> dict:
                    async with trio_libp2p_host_listen(listen_multiaddr="/ip4/127.0.0.1/tcp/0") as host:
                        stream = await open_libp2p_stream_by_multiaddr(
                            host,
                            peer_multiaddr=multiaddr,
                            protocols=["/mcp+p2p/2.0.0", PROTOCOL_MCP_P2P_V1],
                        )
                        client = MCPP2PClient(stream=stream, max_frame_bytes=1024 * 1024)
                        response = await client.initialize(
                            {
                                "profiles": [
                                    "mcp++/profile-z-next",
                                    "mcp++/profile-e-mcp-p2p",
                                ]
                            }
                        )
                        await client.aclose()
                        return response

                init_response = anyio.run(_run_client, backend="trio")
                result = init_response.get("result", {})
                self.assertTrue(result.get("ok"))
                self.assertEqual(result.get("transport"), PROTOCOL_MCP_P2P_V1)
                self.assertEqual(result.get("active_profile"), "mcp++/profile-e-mcp-p2p")

                negotiation = result.get("profile_negotiation", {})
                self.assertTrue(negotiation.get("supports_profile_negotiation"))
                self.assertEqual(negotiation.get("mode"), "optional_additive")
                profiles = negotiation.get("profiles", [])
                self.assertIsInstance(profiles, list)
                self.assertIn("mcp++/profile-e-mcp-p2p", profiles)
                self.assertEqual(result.get("active_profile"), "mcp++/profile-e-mcp-p2p")

            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def test_mixed_version_unknown_profile_and_alias_tools_list_remain_deterministic_over_network(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mcp_transport_mcp_p2p_networked_alias_") as td:
            queue_path = os.path.join(td, "queue.json")
            announce_file = os.path.join(td, "task_p2p_announce.json")
            port = _pick_free_port()

            env = dict(os.environ)
            env.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
            env["PYTHONUNBUFFERED"] = "1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
            env["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
            env["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
            env["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

            code = (
                "import os, anyio, functools; "
                "from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue; "
                "fn = functools.partial(serve_task_queue, queue_path=os.environ['Q'], listen_port=int(os.environ['P'])); "
                "anyio.run(fn, backend='trio')"
            )

            env["Q"] = queue_path
            env["P"] = str(port)

            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                deadline = time.time() + 30.0
                while time.time() < deadline and not os.path.exists(announce_file):
                    if proc.poll() is not None:
                        out = proc.stdout.read() if proc.stdout else ""
                        self.fail(f"service exited early: {proc.returncode}\n{out}")
                    time.sleep(0.05)

                self.assertTrue(os.path.exists(announce_file), "announce file not created")
                info = json.loads(open(announce_file, "r", encoding="utf-8").read())
                multiaddr = str(info.get("multiaddr") or "")
                self.assertIn("/p2p/", multiaddr)

                async def _run_client() -> tuple[dict, dict, dict]:
                    async with trio_libp2p_host_listen(listen_multiaddr="/ip4/127.0.0.1/tcp/0") as host:
                        stream = await open_libp2p_stream_by_multiaddr(
                            host,
                            peer_multiaddr=multiaddr,
                            protocols=["/mcp+p2p/3.0.0", "/mcp+p2p/2.0.0", PROTOCOL_MCP_P2P_V1],
                        )
                        client = MCPP2PClient(stream=stream, max_frame_bytes=1024 * 1024)
                        init_response = await client.initialize(
                            {
                                "protocol_version": "/mcp+p2p/0.8.0",
                                "client_version": "0.8.7",
                                "profile": "mcp++/profile-z-unknown",
                                "profiles": [
                                    "mcp++/profile-z-unknown",
                                    "mcp++/profile-a-idl",
                                ],
                            }
                        )
                        canonical_list = await client.request_raw(
                            {
                                "jsonrpc": "2.0",
                                "id": 2,
                                "method": "tools/list",
                                "params": {},
                            }
                        )
                        alias_list = await client.request_raw(
                            {
                                "jsonrpc": "2.0",
                                "id": 3,
                                "method": "tools.list",
                                "params": {},
                            }
                        )
                        await client.aclose()
                        return init_response, canonical_list, alias_list

                init_response, canonical_list, alias_list = anyio.run(_run_client, backend="trio")
                result = init_response.get("result", {})
                self.assertTrue(result.get("ok"))
                self.assertEqual(result.get("transport"), PROTOCOL_MCP_P2P_V1)
                self.assertEqual(result.get("active_profile"), "mcp++/profile-a-idl")

                negotiation = result.get("profile_negotiation", {})
                self.assertTrue(negotiation.get("supports_profile_negotiation"))
                self.assertEqual(negotiation.get("mode"), "optional_additive")

                canonical_tools = (canonical_list.get("result") or {}).get("tools")
                alias_tools = (alias_list.get("result") or {}).get("tools")
                self.assertIsInstance(canonical_tools, list)
                self.assertIsInstance(alias_tools, list)
                self.assertEqual(
                    [tool.get("name") for tool in canonical_tools if isinstance(tool, dict)],
                    [tool.get("name") for tool in alias_tools if isinstance(tool, dict)],
                )
                self.assertEqual(canonical_tools, alias_tools)

            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
