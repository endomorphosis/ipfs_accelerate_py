import os
import socket
import subprocess
import sys
import tempfile
import time


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_mcp_p2p_taskqueue_status_against_real_service() -> None:
    """End-to-end: start libp2p TaskQueue service, call MCP tool wrapper."""

    with tempfile.TemporaryDirectory(prefix="mcp_p2p_taskqueue_test_") as td:
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
            deadline = time.time() + 20.0
            while time.time() < deadline and not os.path.exists(announce_file):
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    raise AssertionError(f"service exited early: {proc.returncode}\n{out}")
                time.sleep(0.05)

            assert os.path.exists(announce_file), "announce file not created"
            info = __import__("json").loads(open(announce_file, "r", encoding="utf-8").read())
            multiaddr = str(info.get("multiaddr") or "")
            assert "/p2p/" in multiaddr

            # Create a minimal MCP instance and register the new tool.
            from ipfs_accelerate_py.mcp.server import StandaloneMCP
            from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import register_tools

            mcp = StandaloneMCP(name="test")
            register_tools(mcp)

            tool_fn = mcp.tools["p2p_taskqueue_status"]["function"]
            resp = tool_fn(remote_multiaddr=multiaddr, timeout_s=10.0, detail=False)
            assert isinstance(resp, dict)
            assert resp.get("ok") is True
            assert resp.get("peer_id")
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
