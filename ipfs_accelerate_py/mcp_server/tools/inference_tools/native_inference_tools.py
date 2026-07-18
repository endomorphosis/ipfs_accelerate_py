"""Native inference-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_inference_tools_api() -> Dict[str, Any]:
    """Resolve source inference-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.inference import (  # type: ignore
            register_tools as _register_tools_inference,
        )
        # Import standalone functions that exist outside register_tools
        inference_module = __import__(
            "ipfs_accelerate_py.mcp.tools.inference",
            fromlist=["_attach_inference_persistence_metadata"],
        )
        return {"_inference_module": inference_module}
    except Exception:
        pass

    try:
        from ipfs_accelerate_py.mcp.tools.enhanced_inference import (  # type: ignore
            register_tools as _register_tools_enhanced,
        )
    except Exception:
        pass

    return {}


def _load_inference_backends() -> Dict[str, Any]:
    """Load inference backend instances with fallback."""
    try:
        from ipfs_accelerate_py.inference_backend_manager import (  # type: ignore
            get_backend_manager,
        )

        return {"get_backend_manager": get_backend_manager}
    except Exception:
        logger.warning("Inference backend manager unavailable, using fallback stubs")
        return {}


_INFERENCE_API = _load_inference_tools_api()
_BACKEND_API = _load_inference_backends()


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
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def inference_run(
    model: str,
    input_data: Any,
    task: str = "text-generation",
    backend: str = "auto",
    max_length: int = 512,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run inference using a specified model and backend."""
    try:
        manager_factory = _BACKEND_API.get("get_backend_manager")
        if manager_factory is not None:
            mgr = manager_factory()
            result = await mgr.run_inference(
                model=model,
                inputs=input_data,
                task=task,
                backend_preference=backend if backend != "auto" else None,
            )
            return _normalize_payload(result if isinstance(result, dict) else {"output": result})
        return _normalize_payload(
            {
                "model": model,
                "input": input_data,
                "task": task,
                "output": None,
                "backend_available": False,
            }
        )
    except Exception as exc:
        return _error_result(str(exc), model=model, task=task)


async def inference_get_model_list(
    task_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """List models available for inference."""
    try:
        manager_factory = _BACKEND_API.get("get_backend_manager")
        if manager_factory is not None:
            mgr = manager_factory()
            models = mgr.list_models(task=task_filter)
            return _normalize_payload({"models": models, "count": len(models) if isinstance(models, list) else 0})
        return _normalize_payload({"models": [], "count": 0, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc))


async def inference_download_model(
    model_name: str, force: bool = False
) -> Dict[str, Any]:
    """Download a model for local inference."""
    try:
        try:
            from ipfs_accelerate_py.model_manager import get_model_manager  # type: ignore

            mgr = get_model_manager()
            result = mgr.download_model(model_name=model_name, force=force)
            return _normalize_payload(result if isinstance(result, dict) else {"model_name": model_name, "downloaded": result})
        except Exception:
            pass
        return _normalize_payload(
            {
                "model_name": model_name,
                "downloaded": False,
                "backend_available": False,
            }
        )
    except Exception as exc:
        return _error_result(str(exc), model_name=model_name)


async def inference_multiplex(
    model: str,
    inputs: List[Any],
    endpoint_ids: Optional[List[str]] = None,
    task: str = "text-generation",
) -> Dict[str, Any]:
    """Run inference across multiple endpoints and aggregate results."""
    try:
        manager_factory = _BACKEND_API.get("get_backend_manager")
        if manager_factory is not None:
            mgr = manager_factory()
            results = []
            for inp in inputs:
                r = await mgr.run_inference(model=model, inputs=inp, task=task)
                results.append(r if isinstance(r, dict) else {"output": r})
            return _normalize_payload({"model": model, "results": results, "count": len(results)})
        return _normalize_payload(
            {"model": model, "results": [], "count": 0, "backend_available": False}
        )
    except Exception as exc:
        return _error_result(str(exc), model=model)


async def inference_get_endpoint_status(
    endpoint_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get status of inference endpoints."""
    try:
        manager_factory = _BACKEND_API.get("get_backend_manager")
        if manager_factory is not None:
            mgr = manager_factory()
            if endpoint_id:
                status = mgr.get_endpoint_status(endpoint_id=endpoint_id)
            else:
                status = mgr.get_all_endpoint_status()
            return _normalize_payload(status if isinstance(status, dict) else {"status_data": status})
        return _normalize_payload({"endpoints": {}, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


async def inference_configure_api_provider(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Configure an external API inference provider.

    Instantiates the appropriate backend class and registers it with the
    InferenceBackendManager so subsequent inference_run calls can route to it.

    Supported providers (and aliases):
      - ``xai`` / ``grok`` / ``xai_grok``  — xAI Grok
      - ``meta_ai`` / ``meta`` / ``spark`` / ``meta_spark`` / ``meta_llama`` — Meta AI (Llama/Spark)
      - ``openai`` / ``openai_api``
      - ``claude`` / ``anthropic``
      - ``gemini``
      - ``groq``
      - ``hf_tei``, ``hf_tgi``, ``ollama``, ``vllm``

    When *api_key* is omitted the relevant environment variable is checked
    automatically (e.g. ``XAI_API_KEY``, ``META_AI_API_KEY``, ``OPENAI_API_KEY``).
    """
    try:
        manager_factory = _BACKEND_API.get("get_backend_manager")
        if manager_factory is not None:
            mgr = manager_factory()
            result = mgr.configure_provider(
                provider=provider, api_key=api_key, base_url=base_url, **kwargs
            )
            return _normalize_payload(result if isinstance(result, dict) else {"provider": provider, "configured": result})
        return _normalize_payload(
            {"provider": provider, "configured": False, "backend_available": False}
        )
    except Exception as exc:
        return _error_result(str(exc), provider=provider)


def register_native_inference_tools(manager: Any) -> None:
    """Register native inference-tools category tools in unified manager."""
    manager.register_tool(
        category="inference_tools",
        name="inference_run",
        func=inference_run,
        description="Run inference using a specified model and backend.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model identifier."},
                "input_data": {"description": "Input data for inference (string, list, or dict)."},
                "task": {
                    "type": "string",
                    "description": "Inference task type (e.g., text-generation).",
                    "default": "text-generation",
                },
                "backend": {
                    "type": "string",
                    "description": "Backend to use: 'auto', 'cpu', 'cuda', 'mps', etc.",
                    "default": "auto",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum output length.",
                    "default": 512,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature.",
                    "default": 1.0,
                },
            },
            "required": ["model", "input_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
    manager.register_tool(
        category="inference_tools",
        name="inference_get_model_list",
        func=inference_get_model_list,
        description="List models available for inference.",
        input_schema={
            "type": "object",
            "properties": {
                "task_filter": {
                    "type": "string",
                    "description": "Optional task type filter.",
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
    manager.register_tool(
        category="inference_tools",
        name="inference_download_model",
        func=inference_download_model,
        description="Download a model for local inference.",
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model name or HuggingFace ID."},
                "force": {
                    "type": "boolean",
                    "description": "Force re-download even if cached.",
                    "default": False,
                },
            },
            "required": ["model_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
    manager.register_tool(
        category="inference_tools",
        name="inference_multiplex",
        func=inference_multiplex,
        description="Run inference across multiple endpoints and aggregate results.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model identifier."},
                "inputs": {
                    "type": "array",
                    "description": "List of inputs to run in parallel.",
                    "items": {},
                },
                "endpoint_ids": {
                    "type": "array",
                    "description": "Optional list of endpoint IDs to use.",
                    "items": {"type": "string"},
                },
                "task": {
                    "type": "string",
                    "description": "Inference task type.",
                    "default": "text-generation",
                },
            },
            "required": ["model", "inputs"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
    manager.register_tool(
        category="inference_tools",
        name="inference_get_endpoint_status",
        func=inference_get_endpoint_status,
        description="Get status of registered inference endpoints.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {
                    "type": "string",
                    "description": "Optional specific endpoint ID to query.",
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
    manager.register_tool(
        category="inference_tools",
        name="inference_configure_api_provider",
        func=inference_configure_api_provider,
        description=(
            "Configure an external API inference provider. "
            "Supported: xai/grok, meta_ai/spark/meta_llama, openai, claude/anthropic, "
            "gemini, groq, hf_tei, hf_tgi, ollama, vllm. "
            "When api_key is omitted the relevant environment variable is used automatically."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "Provider name or alias. Examples: 'xai', 'grok', 'meta_ai', "
                        "'spark', 'meta_llama', 'openai', 'claude', 'gemini', 'groq'."
                    ),
                },
                "api_key": {
                    "type": "string",
                    "description": (
                        "API key for the provider. "
                        "If omitted, the environment variable is checked automatically "
                        "(e.g. XAI_API_KEY, META_AI_API_KEY, OPENAI_API_KEY)."
                    ),
                },
                "base_url": {
                    "type": "string",
                    "description": "Optional custom base URL override.",
                },
            },
            "required": ["provider"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "inference-tools"],
    )
