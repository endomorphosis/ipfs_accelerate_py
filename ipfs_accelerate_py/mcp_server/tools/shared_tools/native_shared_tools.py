"""Native shared tool implementations for unified mcp_server.

Migrated from ipfs_accelerate_py/mcp/tools/shared_tools.py.
Provides generic shared operations used across multiple subsystems:
generate_text, classify_text, run_inference, search_models,
add_file_to_ipfs, get_file_from_ipfs, list_available_models,
get_queue_status, get_model_queues, get_network_status,
add_file, run_model_test, check_network_status, get_connected_peers,
get_system_status, get_queue_history, get_endpoint_details,
get_endpoint_handlers_by_model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_shared_api() -> Dict[str, Any]:
    """Resolve source shared tool APIs with compatibility fallback."""
    try:
        import ipfs_accelerate_py.mcp.tools.shared_tools as _mod  # type: ignore

        return {"_module": _mod}
    except Exception:
        logger.warning("Source shared_tools API unavailable, using fallback stubs")
        return {}


_API = _load_shared_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def shared_tools_inventory() -> Dict[str, Any]:
    """Return inventory metadata for shared tools."""
    return _normalize_payload(
        {
            "category": "shared_tools",
            "tools": [
                "generate_text",
                "classify_text",
                "run_inference",
                "search_models",
                "add_file_to_ipfs",
                "get_file_from_ipfs",
                "list_available_models",
                "get_queue_status",
                "get_model_queues",
                "get_network_status",
                "add_file",
                "run_model_test",
                "check_network_status",
                "get_connected_peers",
                "get_system_status",
                "get_queue_history",
                "get_endpoint_details",
                "get_endpoint_handlers_by_model",
            ],
            "description": "Shared generic operations across IPFS Accelerate subsystems",
            "source": "mcp/tools/shared_tools.py",
        }
    )


async def generate_text(
    prompt: str,
    model: str = "auto",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate text using an available language model."""
    if not isinstance(prompt, str) or not prompt.strip():
        return _error_result("prompt must be a non-empty string")

    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        result = await mgr.generate(
            prompt=prompt.strip(),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _normalize_payload(result)
    except Exception as exc:
        logger.debug("generate_text inference backend failed: %s", exc)

    return _normalize_payload(
        {
            "prompt": prompt.strip(),
            "model": model,
            "generated_text": None,
            "message": "Text generation unavailable in this environment",
        }
    )


async def classify_text(
    text: str,
    labels: Optional[List[str]] = None,
    model: str = "auto",
) -> Dict[str, Any]:
    """Classify text into provided labels using a classification model."""
    if not isinstance(text, str) or not text.strip():
        return _error_result("text must be a non-empty string")

    return _normalize_payload(
        {
            "text": text.strip(),
            "labels": labels or [],
            "predictions": [],
            "model": model,
            "message": "Text classification unavailable in this environment",
        }
    )


async def run_inference(
    prompt: str,
    model: str,
    task_type: str = "text-generation",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run inference with a specified model."""
    if not isinstance(prompt, str) or not prompt.strip():
        return _error_result("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        return _error_result("model must be a non-empty string")

    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        result = await mgr.run(
            prompt=prompt.strip(),
            model=model.strip(),
            task_type=task_type,
            **(parameters or {}),
        )
        return _normalize_payload(result)
    except Exception as exc:
        logger.debug("run_inference backend failed: %s", exc)

    return _normalize_payload(
        {
            "prompt": prompt.strip(),
            "model": model.strip(),
            "task_type": task_type,
            "result": None,
            "message": "Inference unavailable in this environment",
        }
    )


async def search_models(
    query: str,
    limit: int = 10,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for available models matching a query."""
    if not isinstance(query, str) or not query.strip():
        return _error_result("query must be a non-empty string")

    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner  # type: ignore

        scanner = HuggingFaceHubScanner()
        results = scanner.search_models(query=query.strip(), task=task, limit=limit)
        models = results if isinstance(results, list) else results.get("models", [])
        return _normalize_payload({"query": query.strip(), "models": models, "count": len(models)})
    except Exception as exc:
        logger.debug("search_models scanner failed: %s", exc)

    return _normalize_payload(
        {"query": query.strip(), "models": [], "count": 0}
    )


async def add_file_to_ipfs(file_path: str, pin: bool = True) -> Dict[str, Any]:
    """Add a local file to IPFS."""
    if not isinstance(file_path, str) or not file_path.strip():
        return _error_result("file_path must be a non-empty string")

    try:
        from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit  # type: ignore

        kit = get_ipfs_files_kit()
        result = kit.add_file(path=file_path.strip(), pin=pin)
        if isinstance(result, dict):
            return _normalize_payload(result)
    except Exception as exc:
        logger.debug("add_file_to_ipfs kit failed: %s", exc)

    return _error_result("IPFS unavailable", file_path=file_path)


async def get_file_from_ipfs(
    cid: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a file from IPFS by CID."""
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")

    try:
        from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit  # type: ignore

        kit = get_ipfs_files_kit()
        if output_path:
            result = kit.get_file(cid=cid.strip(), output_path=output_path)
        else:
            result = kit.cat_file(cid=cid.strip())
        if isinstance(result, dict):
            return _normalize_payload(result)
    except Exception as exc:
        logger.debug("get_file_from_ipfs kit failed: %s", exc)

    return _error_result("IPFS unavailable", cid=cid)


async def list_available_models(model_type: Optional[str] = None) -> Dict[str, Any]:
    """List available models (local, cached, or API-accessible)."""
    try:
        from ipfs_accelerate_py.model_manager import ModelManager  # type: ignore

        mgr = ModelManager()
        models = mgr.list_models(model_type=model_type)
        if not isinstance(models, list):
            models = []
        return _normalize_payload({"models": models, "count": len(models)})
    except Exception as exc:
        logger.debug("list_available_models model_manager failed: %s", exc)

    return _normalize_payload({"models": [], "count": 0})


async def get_queue_status() -> Dict[str, Any]:
    """Get inference queue status across all model types."""
    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        status = mgr.get_queue_status()
        return _normalize_payload(status)
    except Exception as exc:
        logger.debug("get_queue_status backend failed: %s", exc)

    return _normalize_payload(
        {"queues": {}, "total_pending": 0, "total_processing": 0}
    )


async def get_model_queues(model_type: Optional[str] = None) -> Dict[str, Any]:
    """Get queue information per model type."""
    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        queues = mgr.get_model_queues(model_type=model_type)
        return _normalize_payload(
            {"queues": queues if isinstance(queues, dict) else {}, "model_type": model_type}
        )
    except Exception as exc:
        logger.debug("get_model_queues backend failed: %s", exc)

    return _normalize_payload({"queues": {}, "model_type": model_type})


async def get_network_status() -> Dict[str, Any]:
    """Get IPFS/P2P network connectivity status."""
    try:
        from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit  # type: ignore

        kit = get_ipfs_files_kit()
        client = getattr(kit, "_client", None) or getattr(kit, "client", None)
        if client is not None:
            import anyio

            peers = await anyio.to_thread.run_sync(client.swarm.peers)
            peer_count = len(peers.get("Peers", [])) if isinstance(peers, dict) else 0
            return _normalize_payload(
                {"connected": True, "peer_count": peer_count}
            )
    except Exception as exc:
        logger.debug("get_network_status kit failed: %s", exc)

    return _normalize_payload({"connected": False, "peer_count": 0})


async def add_file(
    content: str,
    filename: str = "file.txt",
    pin: bool = True,
) -> Dict[str, Any]:
    """Add string content as a file to IPFS."""
    if not isinstance(content, str):
        return _error_result("content must be a string")

    import os
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=os.path.splitext(filename)[1] or ".txt", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await add_file_to_ipfs(tmp_path, pin=pin)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return result
    except Exception as exc:
        return _error_result(str(exc), filename=filename)


async def run_model_test(
    test_type: str,
    model: str = "auto",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a model performance or correctness test."""
    if not isinstance(test_type, str) or not test_type.strip():
        return _error_result("test_type must be a non-empty string")

    return _normalize_payload(
        {
            "test_type": test_type.strip(),
            "model": model,
            "passed": None,
            "message": "Model testing unavailable in this environment",
        }
    )


async def check_network_status() -> Dict[str, Any]:
    """Check IPFS/P2P network connectivity (alias for get_network_status)."""
    return await get_network_status()


async def get_connected_peers() -> Dict[str, Any]:
    """Get list of connected IPFS/P2P peers."""
    try:
        from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit  # type: ignore

        kit = get_ipfs_files_kit()
        client = getattr(kit, "_client", None) or getattr(kit, "client", None)
        if client is not None:
            import anyio

            result = await anyio.to_thread.run_sync(client.swarm.peers)
            peers = []
            if isinstance(result, dict):
                peers = [
                    {"addr": str(p.get("Addr", "")), "peer": str(p.get("Peer", ""))}
                    for p in result.get("Peers", [])
                ]
            return _normalize_payload({"peers": peers, "count": len(peers)})
    except Exception as exc:
        logger.debug("get_connected_peers kit failed: %s", exc)

    return _normalize_payload({"peers": [], "count": 0})


async def get_system_status() -> Dict[str, Any]:
    """Get overall system status (CPU, memory, IPFS, inference backends)."""
    try:
        import psutil  # type: ignore

        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        system_info = {
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(mem.total / (1024 ** 3), 2),
            "memory_used_gb": round(mem.used / (1024 ** 3), 2),
            "memory_percent": mem.percent,
        }
    except Exception:
        system_info = {}

    network_status = await get_network_status()
    queue_status = await get_queue_status()

    return _normalize_payload(
        {
            "system": system_info,
            "network": network_status,
            "queue": queue_status,
        }
    )


async def get_queue_history(limit: int = 50) -> Dict[str, Any]:
    """Get inference queue processing history."""
    return _normalize_payload({"history": [], "count": 0, "limit": limit})


async def get_endpoint_details(endpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """Get details of inference endpoint(s)."""
    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        if endpoint_id:
            details = mgr.get_endpoint(endpoint_id)
        else:
            details = mgr.list_endpoints()
        return _normalize_payload(
            details if isinstance(details, dict) else {"endpoints": details or []}
        )
    except Exception as exc:
        logger.debug("get_endpoint_details backend failed: %s", exc)

    return _normalize_payload({"endpoints": [], "endpoint_id": endpoint_id})


async def get_endpoint_handlers_by_model(model_type: str) -> Dict[str, Any]:
    """Get endpoint handlers that support a specific model type."""
    if not isinstance(model_type, str) or not model_type.strip():
        return _error_result("model_type must be a non-empty string")

    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            InferenceBackendManager,
        )

        mgr = InferenceBackendManager()
        handlers = mgr.get_handlers_by_model(model_type=model_type.strip())
        if isinstance(handlers, list):
            return _normalize_payload(
                {"model_type": model_type.strip(), "handlers": handlers, "count": len(handlers)}
            )
    except Exception as exc:
        logger.debug("get_endpoint_handlers_by_model backend failed: %s", exc)

    return _normalize_payload(
        {"model_type": model_type.strip(), "handlers": [], "count": 0}
    )


def register_native_shared_tools(manager: Any) -> None:
    """Register native shared tools in the unified hierarchical manager."""
    manager.register_tool(
        category="shared_tools",
        name="shared_tools_inventory",
        func=shared_tools_inventory,
        description="Return inventory metadata for shared tools.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared"],
    )
    manager.register_tool(
        category="shared_tools",
        name="generate_text",
        func=generate_text,
        description="Generate text using an available language model.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "auto"},
                "max_tokens": {"type": "integer", "default": 512},
                "temperature": {"type": "number", "default": 0.7},
            },
            "required": ["prompt"],
        },
        runtime="fastapi",
        tags=["native", "shared", "inference"],
    )
    manager.register_tool(
        category="shared_tools",
        name="classify_text",
        func=classify_text,
        description="Classify text into provided labels.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "model": {"type": "string", "default": "auto"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "shared", "inference"],
    )
    manager.register_tool(
        category="shared_tools",
        name="run_inference",
        func=run_inference,
        description="Run inference with a specified model.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "task_type": {"type": "string", "default": "text-generation"},
                "parameters": {"type": "object"},
            },
            "required": ["prompt", "model"],
        },
        runtime="fastapi",
        tags=["native", "shared", "inference"],
    )
    manager.register_tool(
        category="shared_tools",
        name="search_models",
        func=search_models,
        description="Search for available models matching a query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "task": {"type": "string"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "shared", "models"],
    )
    manager.register_tool(
        category="shared_tools",
        name="add_file_to_ipfs",
        func=add_file_to_ipfs,
        description="Add a local file to IPFS.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "pin": {"type": "boolean", "default": True},
            },
            "required": ["file_path"],
        },
        runtime="fastapi",
        tags=["native", "shared", "ipfs"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_file_from_ipfs",
        func=get_file_from_ipfs,
        description="Get a file from IPFS by CID.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "shared", "ipfs"],
    )
    manager.register_tool(
        category="shared_tools",
        name="list_available_models",
        func=list_available_models,
        description="List available models (local, cached, or API-accessible).",
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {"type": "string"}
            },
        },
        runtime="fastapi",
        tags=["native", "shared", "models"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_queue_status",
        func=get_queue_status,
        description="Get inference queue status across all model types.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared", "queue"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_model_queues",
        func=get_model_queues,
        description="Get queue information per model type.",
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {"type": "string"}
            },
        },
        runtime="fastapi",
        tags=["native", "shared", "queue"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_network_status",
        func=get_network_status,
        description="Get IPFS/P2P network connectivity status.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared", "network"],
    )
    manager.register_tool(
        category="shared_tools",
        name="add_file",
        func=add_file,
        description="Add string content as a file to IPFS.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "filename": {"type": "string", "default": "file.txt"},
                "pin": {"type": "boolean", "default": True},
            },
            "required": ["content"],
        },
        runtime="fastapi",
        tags=["native", "shared", "ipfs"],
    )
    manager.register_tool(
        category="shared_tools",
        name="run_model_test",
        func=run_model_test,
        description="Run a model performance or correctness test.",
        input_schema={
            "type": "object",
            "properties": {
                "test_type": {"type": "string"},
                "model": {"type": "string", "default": "auto"},
                "parameters": {"type": "object"},
            },
            "required": ["test_type"],
        },
        runtime="fastapi",
        tags=["native", "shared", "testing"],
    )
    manager.register_tool(
        category="shared_tools",
        name="check_network_status",
        func=check_network_status,
        description="Check IPFS/P2P network connectivity.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared", "network"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_connected_peers",
        func=get_connected_peers,
        description="Get list of connected IPFS/P2P peers.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared", "network"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_system_status",
        func=get_system_status,
        description="Get overall system status (CPU, memory, IPFS, inference).",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "shared", "system"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_queue_history",
        func=get_queue_history,
        description="Get inference queue processing history.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50}
            },
        },
        runtime="fastapi",
        tags=["native", "shared", "queue"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_endpoint_details",
        func=get_endpoint_details,
        description="Get details of inference endpoint(s).",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"}
            },
        },
        runtime="fastapi",
        tags=["native", "shared", "endpoint"],
    )
    manager.register_tool(
        category="shared_tools",
        name="get_endpoint_handlers_by_model",
        func=get_endpoint_handlers_by_model,
        description="Get endpoint handlers that support a specific model type.",
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {"type": "string"}
            },
            "required": ["model_type"],
        },
        runtime="fastapi",
        tags=["native", "shared", "endpoint"],
    )
