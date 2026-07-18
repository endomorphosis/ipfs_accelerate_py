"""Native model-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_model_tools_api() -> Dict[str, Any]:
    """Resolve source model-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.models import (  # type: ignore
            search_models_tool as _search_models,
            recommend_models_tool as _recommend_models,
            get_model_details_tool as _get_model_details,
            get_model_stats_tool as _get_model_stats,
            list_hf_inference_models_tool as _list_hf_inference_models,
            get_hf_inference_model_metadata_tool as _get_hf_inference_model_metadata,
            build_hf_inference_ipld_document_tool as _build_hf_inference_ipld_document,
            get_hf_inference_ipld_cid_tool as _get_hf_inference_ipld_cid,
            publish_hf_inference_ipld_to_ipfs_tool as _publish_hf_inference_ipld_to_ipfs,
            load_hf_inference_ipld_from_ipfs_tool as _load_hf_inference_ipld_from_ipfs,
        )

        return {
            "search_models": _search_models,
            "recommend_models": _recommend_models,
            "get_model_details": _get_model_details,
            "get_model_stats": _get_model_stats,
            "list_hf_inference_models": _list_hf_inference_models,
            "get_hf_inference_model_metadata": _get_hf_inference_model_metadata,
            "build_hf_inference_ipld_document": _build_hf_inference_ipld_document,
            "get_hf_inference_ipld_cid": _get_hf_inference_ipld_cid,
            "publish_hf_inference_ipld_to_ipfs": _publish_hf_inference_ipld_to_ipfs,
            "load_hf_inference_ipld_from_ipfs": _load_hf_inference_ipld_from_ipfs,
        }
    except Exception:
        logger.warning("Source model_tools import unavailable, using fallback model functions")

        def _search_fallback(
            query: str,
            task_filter: Optional[str] = None,
            limit: int = 10,
        ) -> Dict[str, Any]:
            return {"status": "success", "models": [], "query": query, "count": 0}

        def _recommend_fallback(
            task_type: str,
            hardware: str = "cpu",
            max_size_gb: Optional[float] = None,
        ) -> Dict[str, Any]:
            return {"status": "success", "recommendations": [], "task_type": task_type}

        def _details_fallback(model_id: str) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "details": {}}

        def _stats_fallback() -> Dict[str, Any]:
            return {"status": "success", "stats": {}}

        def _list_hf_fallback(model_kind: Optional[str] = None) -> Dict[str, Any]:
            return {"status": "success", "models": [], "count": 0}

        def _hf_metadata_fallback(model_id: str) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "metadata": {}}

        def _build_ipld_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "document": {}}

        def _get_cid_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "cid": None}

        def _publish_ipld_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "published": False}

        def _load_ipld_fallback(cid: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "cid": cid, "document": {}}

        return {
            "search_models": _search_fallback,
            "recommend_models": _recommend_fallback,
            "get_model_details": _details_fallback,
            "get_model_stats": _stats_fallback,
            "list_hf_inference_models": _list_hf_fallback,
            "get_hf_inference_model_metadata": _hf_metadata_fallback,
            "build_hf_inference_ipld_document": _build_ipld_fallback,
            "get_hf_inference_ipld_cid": _get_cid_fallback,
            "publish_hf_inference_ipld_to_ipfs": _publish_ipld_fallback,
            "load_hf_inference_ipld_from_ipfs": _load_ipld_fallback,
        }


_API = _load_model_tools_api()


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


async def model_search(
    query: str,
    task_filter: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search for models matching a query."""
    try:
        result = _API["search_models"](query=query, task_filter=task_filter, limit=limit)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), query=query)


async def model_recommend(
    task_type: str,
    hardware: str = "cpu",
    max_size_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Recommend models for a task and hardware configuration."""
    try:
        result = _API["recommend_models"](
            task_type=task_type, hardware=hardware, max_size_gb=max_size_gb
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), task_type=task_type, hardware=hardware)


async def model_get_details(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        result = _API["get_model_details"](model_id=model_id)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_get_stats() -> Dict[str, Any]:
    """Get aggregate statistics about available models."""
    try:
        result = _API["get_model_stats"]()
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def model_list_hf_inference(
    model_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """List HuggingFace inference models."""
    try:
        result = _API["list_hf_inference_models"](model_kind=model_kind)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def model_get_hf_metadata(model_id: str) -> Dict[str, Any]:
    """Get metadata for a HuggingFace inference model."""
    try:
        result = _API["get_hf_inference_model_metadata"](model_id=model_id)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_build_hf_ipld_document(
    model_id: str,
    include_config: bool = True,
    include_tokenizer: bool = True,
) -> Dict[str, Any]:
    """Build an IPLD document for a HuggingFace inference model."""
    try:
        result = _API["build_hf_inference_ipld_document"](
            model_id=model_id,
            include_config=include_config,
            include_tokenizer=include_tokenizer,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_get_hf_ipld_cid(
    model_id: str,
    include_config: bool = True,
    include_tokenizer: bool = True,
) -> Dict[str, Any]:
    """Get the IPLD CID for a HuggingFace inference model document."""
    try:
        result = _API["get_hf_inference_ipld_cid"](
            model_id=model_id,
            include_config=include_config,
            include_tokenizer=include_tokenizer,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_publish_hf_ipld_to_ipfs(
    model_id: str,
    pin: bool = True,
) -> Dict[str, Any]:
    """Publish a HuggingFace model IPLD document to IPFS."""
    try:
        result = _API["publish_hf_inference_ipld_to_ipfs"](model_id=model_id, pin=pin)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_load_hf_ipld_from_ipfs(cid: str) -> Dict[str, Any]:
    """Load a HuggingFace model IPLD document from IPFS."""
    try:
        result = _API["load_hf_inference_ipld_from_ipfs"](cid=cid)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), cid=cid)


def register_native_model_tools(manager: Any) -> None:
    """Register native model-tools category tools in unified manager."""
    manager.register_tool(
        category="model_tools",
        name="model_search",
        func=model_search,
        description="Search for models matching a query string and optional task filter.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "task_filter": {
                    "type": "string",
                    "description": "Optional task type filter (e.g., 'text-generation').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_recommend",
        func=model_recommend,
        description="Recommend models for a task type and hardware configuration.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Task type (e.g., 'text-generation')."},
                "hardware": {
                    "type": "string",
                    "description": "Target hardware (e.g., 'cpu', 'cuda').",
                    "default": "cpu",
                },
                "max_size_gb": {
                    "type": "number",
                    "description": "Optional maximum model size in GB.",
                },
            },
            "required": ["task_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_details",
        func=model_get_details,
        description="Get detailed information about a specific model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model identifier or HuggingFace ID."}
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_stats",
        func=model_get_stats,
        description="Get aggregate statistics about available models.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_list_hf_inference",
        func=model_list_hf_inference,
        description="List HuggingFace inference API compatible models.",
        input_schema={
            "type": "object",
            "properties": {
                "model_kind": {
                    "type": "string",
                    "description": "Optional model kind filter.",
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_hf_metadata",
        func=model_get_hf_metadata,
        description="Get metadata for a HuggingFace inference model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."}
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_build_hf_ipld_document",
        func=model_build_hf_ipld_document,
        description="Build an IPLD document for a HuggingFace model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "include_config": {
                    "type": "boolean",
                    "description": "Include model configuration.",
                    "default": True,
                },
                "include_tokenizer": {
                    "type": "boolean",
                    "description": "Include tokenizer configuration.",
                    "default": True,
                },
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_hf_ipld_cid",
        func=model_get_hf_ipld_cid,
        description="Get the IPLD CID for a HuggingFace model document.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "include_config": {"type": "boolean", "default": True},
                "include_tokenizer": {"type": "boolean", "default": True},
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_publish_hf_ipld_to_ipfs",
        func=model_publish_hf_ipld_to_ipfs,
        description="Publish a HuggingFace model IPLD document to IPFS.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "pin": {
                    "type": "boolean",
                    "description": "Pin the published document.",
                    "default": True,
                },
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_load_hf_ipld_from_ipfs",
        func=model_load_hf_ipld_from_ipfs,
        description="Load a HuggingFace model IPLD document from IPFS by CID.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "IPFS content identifier."}
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
