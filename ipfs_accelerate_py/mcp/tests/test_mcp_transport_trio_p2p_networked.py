#!/usr/bin/env python3
"""Optional networked trio-p2p transport integration tests."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class _DummyServer:
    def __init__(self):
        self.tools = {}
        self.mcp = None

    def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema,
            "execution_context": execution_context,
            "tags": tags,
        }


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@unittest.skipUnless(importlib.util.find_spec("libp2p") is not None, "libp2p not installed")
class TestMCPTransportTrioP2PNetworked(unittest.TestCase):
    """Networked trio-p2p integration checks (optional by dependency)."""

    def test_unified_p2p_status_dispatch_against_real_service(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mcp_trio_p2p_networked_") as td:
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

                with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", return_value=_DummyServer()):
                    with patch.dict(
                        os.environ,
                        {
                            "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                            "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                            "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
                        },
                        clear=False,
                    ):
                        server = create_mcp_server(name="trio-p2p-networked")

                async def _call_status() -> dict:
                    dispatch = server.tools["tools_dispatch"]["function"]
                    response = await dispatch(
                        "p2p",
                        "p2p_taskqueue_status",
                        {"remote_multiaddr": multiaddr, "timeout_s": 10.0, "detail": False},
                    )
                    self.assertIsInstance(response, dict)
                    return response

                result = anyio.run(_call_status)
                self.assertTrue(result.get("ok"), msg=f"unexpected response: {result}")
                self.assertIn("peer_id", result)

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
