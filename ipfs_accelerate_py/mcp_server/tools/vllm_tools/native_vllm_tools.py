"""Native vLLM inference tool implementations for unified mcp_server.

vLLM (https://github.com/vllm-project/vllm) is a high-throughput LLM
inference and serving library.  These tools expose its OpenAI-compatible
HTTP API through the MCP++ tool dispatch surface so that any MCP client
can run text and chat completions against a running vLLM server.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_vllm_backend() -> Any:
    """Import and return the vllm backend class, or None if unavailable."""
    try:
        from ipfs_accelerate_py.api_backends.vllm import vllm as _vllm

        return _vllm
    except Exception as exc:
        logger.warning("vllm backend unavailable: %s", exc)
        return None


_VLLM_CLASS = _get_vllm_backend()


def _make_instance(
    endpoint_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Instantiate the vllm backend with caller-supplied connection details."""
    if _VLLM_CLASS is None:
        return None
    metadata: Dict[str, str] = {}
    if endpoint_url:
        metadata["vllm_api_url"] = endpoint_url
    if api_key:
        metadata["vllm_api_key"] = api_key
    try:
        return _VLLM_CLASS(metadata=metadata)
    except Exception as exc:
        logger.error("Failed to instantiate vllm backend: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def vllm_generate_text(
    prompt: str,
    model: str = "default",
    endpoint_url: str = "",
    api_key: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """Generate text from a prompt using a vLLM server.

    Args:
        prompt: Input text prompt.
        model: Model name served by the vLLM server (e.g. "meta-llama/Llama-2-7b").
        endpoint_url: Base URL of the vLLM server.  Falls back to VLLM_API_URL
            env var or http://localhost:8000.
        api_key: Optional bearer token for the vLLM server.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0–2).
        top_p: Nucleus sampling probability mass.

    Returns:
        {"status": "success", "text": "<generated>", "model": "<model>", ...}
    """
    backend = _make_instance(endpoint_url or None, api_key or None)
    if backend is None:
        return {
            "status": "error",
            "success": False,
            "error": "vllm_backend_unavailable",
        }

    try:
        text = backend.generate_text(
            prompt=prompt,
            model=model or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if text is None:
            return {
                "status": "error",
                "success": False,
                "error": "no_response_from_vllm",
            }
        return {
            "status": "success",
            "success": True,
            "text": text,
            "model": model,
            "prompt_length": len(prompt),
        }
    except Exception as exc:
        return {
            "status": "error",
            "success": False,
            "error": str(exc),
        }


async def vllm_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "default",
    endpoint_url: str = "",
    api_key: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """Chat completion via a vLLM server (/v1/chat/completions).

    Args:
        messages: List of messages with 'role' and 'content' keys.
        model: Model name served by the vLLM server.
        endpoint_url: Base URL of the vLLM server.
        api_key: Optional bearer token.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.

    Returns:
        {"status": "success", "reply": "<assistant text>", ...}
    """
    backend = _make_instance(endpoint_url or None, api_key or None)
    if backend is None:
        return {
            "status": "error",
            "success": False,
            "error": "vllm_backend_unavailable",
        }

    if not isinstance(messages, list) or not messages:
        return {
            "status": "error",
            "success": False,
            "error": "messages_must_be_non_empty_list",
        }

    try:
        reply = backend.chat_completion(
            messages=messages,
            model=model or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if reply is None:
            return {
                "status": "error",
                "success": False,
                "error": "no_response_from_vllm",
            }
        return {
            "status": "success",
            "success": True,
            "reply": reply,
            "model": model,
            "message_count": len(messages),
        }
    except Exception as exc:
        return {
            "status": "error",
            "success": False,
            "error": str(exc),
        }


async def vllm_list_models(
    endpoint_url: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    """List models available on a vLLM server (/v1/models).

    Args:
        endpoint_url: Base URL of the vLLM server.
        api_key: Optional bearer token.

    Returns:
        {"status": "success", "models": [...], "count": N}
    """
    backend = _make_instance(endpoint_url or None, api_key or None)
    if backend is None:
        return {
            "status": "error",
            "success": False,
            "error": "vllm_backend_unavailable",
        }

    try:
        models = backend.list_models()
        return {
            "status": "success",
            "success": True,
            "models": models,
            "count": len(models),
        }
    except Exception as exc:
        return {
            "status": "error",
            "success": False,
            "error": str(exc),
        }


async def vllm_server_status(
    endpoint_url: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    """Check whether the vLLM server is reachable and return its model list.

    Returns:
        {"status": "success", "reachable": true, "models": [...]}
    """
    result = await vllm_list_models(endpoint_url=endpoint_url, api_key=api_key)
    reachable = result.get("success", False)
    return {
        "status": "success" if reachable else "error",
        "success": reachable,
        "reachable": reachable,
        "models": result.get("models", []),
        "model_count": result.get("count", 0),
        "error": result.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Category registration
# ---------------------------------------------------------------------------

_TOOL_SPECS = {
    "vllm_generate_text": {
        "function": vllm_generate_text,
        "description": (
            "Generate text from a prompt using a vLLM inference server "
            "(/v1/completions). Shares the same handler code path as all "
            "other API backends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Input text prompt."},
                "model": {"type": "string", "description": "Model name on the vLLM server.", "default": "default"},
                "endpoint_url": {"type": "string", "description": "vLLM server base URL.", "default": ""},
                "api_key": {"type": "string", "description": "Optional bearer token.", "default": ""},
                "max_tokens": {"type": "integer", "description": "Max tokens to generate.", "default": 256},
                "temperature": {"type": "number", "description": "Sampling temperature.", "default": 0.7},
                "top_p": {"type": "number", "description": "Nucleus sampling probability.", "default": 0.95},
            },
            "required": ["prompt"],
        },
        "tags": ["vllm", "inference", "completion", "llm"],
    },
    "vllm_chat_completion": {
        "function": vllm_chat_completion,
        "description": (
            "Chat completions via vLLM (/v1/chat/completions).  "
            "Accepts an OpenAI-style messages list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "List of {role, content} message dicts.",
                    "items": {"type": "object"},
                },
                "model": {"type": "string", "default": "default"},
                "endpoint_url": {"type": "string", "default": ""},
                "api_key": {"type": "string", "default": ""},
                "max_tokens": {"type": "integer", "default": 256},
                "temperature": {"type": "number", "default": 0.7},
                "top_p": {"type": "number", "default": 0.95},
            },
            "required": ["messages"],
        },
        "tags": ["vllm", "inference", "chat", "llm"],
    },
    "vllm_list_models": {
        "function": vllm_list_models,
        "description": "List models served by a vLLM server (/v1/models).",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint_url": {"type": "string", "default": ""},
                "api_key": {"type": "string", "default": ""},
            },
            "required": [],
        },
        "tags": ["vllm", "models", "discovery"],
    },
    "vllm_server_status": {
        "function": vllm_server_status,
        "description": "Check reachability of a vLLM server and list its models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint_url": {"type": "string", "default": ""},
                "api_key": {"type": "string", "default": ""},
            },
            "required": [],
        },
        "tags": ["vllm", "health", "status"],
    },
}


def register_native_vllm_tools(manager: Any) -> None:
    """Register vLLM inference tools with a HierarchicalToolManager."""
    for tool_name, spec in _TOOL_SPECS.items():
        try:
            manager.register_tool(
                category="vllm_tools",
                name=tool_name,
                function=spec["function"],
                description=spec["description"],
                input_schema=spec.get("input_schema", {}),
                tags=spec.get("tags", []),
            )
        except Exception as exc:
            logger.warning("Failed to register vllm tool '%s': %s", tool_name, exc)


__all__ = [
    "register_native_vllm_tools",
    "vllm_generate_text",
    "vllm_chat_completion",
    "vllm_list_models",
    "vllm_server_status",
]
