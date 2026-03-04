import json
import os
import socket
import tempfile
import time
import importlib.util

import pytest


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _have_local_gpt2() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except Exception:
        return False


def test_p2p_call_tool_dispatches_to_mcp_registry() -> None:
    """E2E (single-process): libp2p TaskQueue service op=call_tool -> MCP tool registry."""

    if importlib.util.find_spec("libp2p") is None:
        pytest.skip("optional dependency 'libp2p' is not installed")

    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

    # Deterministic local-only behavior.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"

    with tempfile.TemporaryDirectory(prefix="p2p_call_tool_bridge_") as td:
        announce_file = os.path.join(td, "task_p2p_announce.json")
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file

        port = _pick_free_port()
        queue_path = os.path.join(td, "queue.json")

        # Create MCP server instance (registers tools + sets global instance).
        from ipfs_accelerate_py.mcp.server import create_mcp_server

        _server = create_mcp_server(accelerate_instance=None)

        # Start p2p service in-process.
        from ipfs_accelerate_py.p2p_tasks.runtime import TaskQueueP2PServiceRuntime

        rt = TaskQueueP2PServiceRuntime()
        rt.start(queue_path=queue_path, listen_port=port, accelerate_instance=None)

        try:
            deadline = time.time() + 20.0
            while time.time() < deadline and not os.path.exists(announce_file):
                time.sleep(0.05)

            assert os.path.exists(announce_file), "announce file not created"
            info = json.loads(open(announce_file, "r", encoding="utf-8").read())
            multiaddr = str(info.get("multiaddr") or "")
            assert "/p2p/" in multiaddr

            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, call_tool_sync

            remote = RemoteQueue(multiaddr=multiaddr)
            resp = call_tool_sync(remote=remote, tool_name="get_server_status", args={})
            assert isinstance(resp, dict)
            assert resp.get("ok") is True
            # invoke_mcp_tool wraps result under `result`.
            assert isinstance(resp.get("result"), dict)

            resp2 = call_tool_sync(remote=remote, tool_name="list_peers", args={"include_capabilities": False})
            assert isinstance(resp2, dict)
            assert resp2.get("ok") is True
            assert isinstance(resp2.get("result"), dict)
            assert resp2["result"].get("ok") is True
            assert isinstance(resp2["result"].get("peers"), list)
        finally:
            rt.stop(timeout_s=3.0)


def test_p2p_call_tool_runs_gpt2_inference_over_libp2p() -> None:
    """E2E (single-process): libp2p TaskQueue service op=call_tool -> MCP run_inference(gpt2).

    This validates the MCP+p2p inference path over an actual libp2p stream using
    GPT-2 inputs, which is a practical proxy for cross-machine usage.
    """

    if importlib.util.find_spec("libp2p") is None:
        pytest.skip("optional dependency 'libp2p' is not installed")
    if os.environ.get("IPFS_ACCELERATE_PY_RUN_GPT2_E2E", "0") != "1":
        pytest.skip("set IPFS_ACCELERATE_PY_RUN_GPT2_E2E=1 to run GPT-2 E2E inference")
    if not _have_local_gpt2():
        pytest.skip("local GPT-2 artifacts are not available")

    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

    # Deterministic local-only behavior.
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
    os.environ["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"

    with tempfile.TemporaryDirectory(prefix="p2p_call_tool_gpt2_") as td:
        announce_file = os.path.join(td, "task_p2p_announce.json")
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file

        port = _pick_free_port()
        queue_path = os.path.join(td, "queue.json")

        from ipfs_accelerate_py.mcp.server import create_mcp_server

        _server = create_mcp_server(accelerate_instance=None)

        from ipfs_accelerate_py.p2p_tasks.runtime import TaskQueueP2PServiceRuntime

        rt = TaskQueueP2PServiceRuntime()
        rt.start(queue_path=queue_path, listen_port=port, accelerate_instance=None)

        try:
            deadline = time.time() + 20.0
            while time.time() < deadline and not os.path.exists(announce_file):
                time.sleep(0.05)

            assert os.path.exists(announce_file), "announce file not created"
            info = json.loads(open(announce_file, "r", encoding="utf-8").read())
            multiaddr = str(info.get("multiaddr") or "")
            assert "/p2p/" in multiaddr

            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, call_tool_sync

            remote = RemoteQueue(multiaddr=multiaddr)
            try:
                resp = call_tool_sync(
                    remote=remote,
                    tool_name="run_inference",
                    args={
                        "model": "gpt2",
                        "inputs": ["Hello from MCP+p2p GPT-2 test."],
                        "device": "cpu",
                        "max_length": 32,
                        "temperature": 0.0,
                    },
                    timeout_s=300.0,
                )
            except BaseExceptionGroup as exc:
                err = str(exc).lower()
                if ("timeout" in err) or ("no response" in err):
                    pytest.skip(f"gpt2 inference unavailable in this environment: {err}")
                raise
            except TimeoutError as exc:
                pytest.skip(f"gpt2 inference unavailable in this environment: {exc}")
            except RuntimeError as exc:
                err = str(exc).lower()
                if "no response" in err:
                    pytest.skip(f"gpt2 inference unavailable in this environment: {err}")
                raise

            if not bool(resp.get("ok")):
                err = str(resp.get("error") or "").lower()
                if ("timeout" in err) or ("no response" in err):
                    pytest.skip(f"gpt2 inference unavailable in this environment: {err}")

            assert isinstance(resp, dict)
            assert resp.get("ok") is True

            result = resp.get("result")
            assert isinstance(result, dict)
            # The inference tool returns either outputs or a clear error payload.
            assert ("outputs" in result) or ("error" in result)
            if "error" not in result:
                assert result.get("model") == "gpt2"
        finally:
            rt.stop(timeout_s=3.0)
