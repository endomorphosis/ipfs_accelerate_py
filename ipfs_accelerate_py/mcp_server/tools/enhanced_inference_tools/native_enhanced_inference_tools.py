"""Native enhanced inference tool implementations for unified mcp_server.

Migrated from ipfs_accelerate_py/mcp/tools/enhanced_inference.py.
Operations: multiplex_inference, register_endpoint, get_endpoint_status,
configure_api_provider, search_huggingface_models, get_queue_status,
get_queue_history, register_cli_endpoint, list_cli_endpoints, cli_inference,
get_cli_providers, get_cli_config, get_cli_install, validate_cli_config,
check_cli_version.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_enhanced_inference_api() -> Dict[str, Any]:
    """Resolve source enhanced inference APIs with compatibility fallback."""
    try:
        import ipfs_accelerate_py.mcp.tools.enhanced_inference as _mod  # type: ignore

        return {
            "_module": _mod,
            "API_PROVIDERS": getattr(_mod, "API_PROVIDERS", {}),
            "CLI_PROVIDERS": getattr(_mod, "CLI_PROVIDERS", {}),
            "ENDPOINT_REGISTRY": getattr(_mod, "ENDPOINT_REGISTRY", {}),
            "QUEUE_MONITOR": getattr(_mod, "QUEUE_MONITOR", {}),
        }
    except Exception:
        logger.warning("Source enhanced inference API unavailable, using fallback stubs")
        return {}


_API = _load_enhanced_inference_api()


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


async def enhanced_inference_inventory() -> Dict[str, Any]:
    """Return inventory metadata for enhanced inference tools."""
    api_providers = list(_API.get("API_PROVIDERS", {}).keys())
    cli_providers = list(_API.get("CLI_PROVIDERS", {}).keys())
    return _normalize_payload(
        {
            "category": "enhanced_inference_tools",
            "tools": [
                "multiplex_inference",
                "register_endpoint",
                "get_endpoint_status",
                "configure_api_provider",
                "search_huggingface_models",
                "get_queue_status",
                "get_queue_history",
                "register_cli_endpoint",
                "list_cli_endpoints",
                "cli_inference",
                "get_cli_providers",
                "get_cli_config",
                "get_cli_install",
                "validate_cli_config",
                "check_cli_version",
            ],
            "api_providers": api_providers,
            "cli_providers": cli_providers,
            "description": "Enhanced inference: multiplexing, API providers, CLI adapters",
            "source": "mcp/tools/enhanced_inference.py",
        }
    )


async def multiplex_inference(
    prompt: str,
    model: str = "auto",
    task_type: str = "text-generation",
    provider: str = "auto",
    endpoint_id: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Run inference via the multiplexed endpoint registry.

    Routes the request to the best available provider (local, API, or CLI).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return _error_result("prompt must be a non-empty string")

    mod = _API.get("_module")
    if mod is not None:
        try:
            # Delegate to source module's internal inference helper
            from ipfs_accelerate_py.mcp.tools.enhanced_inference import (  # type: ignore
                _run_local_inference,
                _run_api_inference,
            )

            # Try local first, then API
            if provider in ("auto", "local"):
                result = _run_local_inference(
                    model=model,
                    prompt=prompt,
                    task_type=task_type,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if isinstance(result, dict) and not result.get("error"):
                    return _normalize_payload(result)

            if provider in ("auto", "openai", "anthropic", "huggingface"):
                prov = "openai" if provider == "auto" else provider
                result = _run_api_inference(
                    provider=prov,
                    model=model,
                    prompt=prompt,
                    task_type=task_type,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return _normalize_payload(result)
        except Exception as exc:
            logger.warning("multiplex_inference delegate failed: %s", exc)

    return _normalize_payload(
        {
            "prompt": prompt,
            "model": model,
            "task_type": task_type,
            "provider": provider,
            "result": None,
            "message": "Enhanced inference unavailable in this environment",
        }
    )


async def register_endpoint(
    endpoint_id: str,
    endpoint_type: str,
    model: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register an inference endpoint in the endpoint registry."""
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        return _error_result("endpoint_id must be a non-empty string")
    if not isinstance(endpoint_type, str) or not endpoint_type.strip():
        return _error_result("endpoint_type must be a non-empty string")

    registry = _API.get("ENDPOINT_REGISTRY")
    if isinstance(registry, dict):
        registry[endpoint_id.strip()] = {
            "endpoint_type": endpoint_type.strip(),
            "model": model,
            "config": config or {},
            "status": "registered",
        }
        return _normalize_payload(
            {
                "endpoint_id": endpoint_id.strip(),
                "registered": True,
                "endpoint_type": endpoint_type.strip(),
                "model": model,
            }
        )
    return _error_result("Endpoint registry unavailable")


async def get_endpoint_status(endpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """Get status of one or all registered inference endpoints."""
    registry = _API.get("ENDPOINT_REGISTRY", {})
    if endpoint_id:
        entry = registry.get(endpoint_id.strip())
        if entry is None:
            return _error_result(f"Endpoint '{endpoint_id}' not found", endpoint_id=endpoint_id)
        return _normalize_payload({"endpoint_id": endpoint_id.strip(), **entry})
    return _normalize_payload(
        {"endpoints": dict(registry), "count": len(registry)}
    )


async def configure_api_provider(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    enabled: bool = True,
) -> Dict[str, Any]:
    """Configure an API inference provider (openai, anthropic, huggingface)."""
    if not isinstance(provider, str) or not provider.strip():
        return _error_result("provider must be a non-empty string")

    api_providers = _API.get("API_PROVIDERS", {})
    known = list(api_providers.keys()) if api_providers else ["openai", "anthropic", "huggingface"]
    if provider.strip() not in known:
        return _error_result(
            f"Unknown provider '{provider}'. Known: {known}",
            provider=provider,
        )
    return _normalize_payload(
        {
            "provider": provider.strip(),
            "enabled": enabled,
            "base_url_configured": bool(base_url),
            "api_key_configured": bool(api_key),
        }
    )


async def search_huggingface_models(
    query: str,
    task: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search HuggingFace model hub for models matching a query."""
    if not isinstance(query, str) or not query.strip():
        return _error_result("query must be a non-empty string")

    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner

        scanner = HuggingFaceHubScanner()
        results = scanner.search_models(query=query.strip(), task=task, limit=limit)
        if isinstance(results, (list, dict)):
            models = results if isinstance(results, list) else results.get("models", [])
            return _normalize_payload(
                {"query": query.strip(), "models": models, "count": len(models)}
            )
    except Exception as exc:
        logger.warning("HuggingFace search failed: %s", exc)

    return _normalize_payload(
        {
            "query": query.strip(),
            "models": [],
            "count": 0,
            "message": "HuggingFace search unavailable in this environment",
        }
    )


async def get_queue_status() -> Dict[str, Any]:
    """Get current inference queue status and statistics."""
    monitor = _API.get("QUEUE_MONITOR", {})
    if monitor:
        return _normalize_payload(
            {
                "stats": monitor.get("stats", {}),
                "queue_length": len(monitor.get("global_queue", [])),
                "endpoint_count": len(monitor.get("endpoint_queues", {})),
            }
        )
    return _normalize_payload(
        {
            "stats": {
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "pending_tasks": 0,
            },
            "queue_length": 0,
            "endpoint_count": 0,
        }
    )


async def get_queue_history(limit: int = 50) -> Dict[str, Any]:
    """Get inference queue history."""
    monitor = _API.get("QUEUE_MONITOR", {})
    history = list(monitor.get("global_queue", []))[-limit:] if monitor else []
    return _normalize_payload({"history": history, "count": len(history), "limit": limit})


async def register_cli_endpoint(
    cli_type: str,
    endpoint_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a CLI inference endpoint (claude_cli, openai_cli, gemini_cli, vscode_cli)."""
    if not isinstance(cli_type, str) or not cli_type.strip():
        return _error_result("cli_type must be a non-empty string")
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        return _error_result("endpoint_id must be a non-empty string")

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            register_cli_endpoint as _register,
        )

        result = _register(
            cli_type=cli_type.strip(),
            endpoint_id=endpoint_id.strip(),
            config=config or {},
        )
        return _normalize_payload(result)
    except Exception as exc:
        logger.warning("register_cli_endpoint delegate failed: %s", exc)

    return _normalize_payload(
        {
            "cli_type": cli_type.strip(),
            "endpoint_id": endpoint_id.strip(),
            "registered": False,
            "message": "CLI adapter registry unavailable",
        }
    )


async def list_cli_endpoints() -> Dict[str, Any]:
    """List registered CLI inference endpoints."""
    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            list_cli_endpoints as _list,
            CLI_ADAPTER_REGISTRY,
        )

        endpoints = _list() if callable(_list) else list(CLI_ADAPTER_REGISTRY.keys())
        return _normalize_payload({"endpoints": endpoints, "count": len(endpoints)})
    except Exception as exc:
        logger.warning("list_cli_endpoints delegate failed: %s", exc)

    return _normalize_payload({"endpoints": [], "count": 0})


async def cli_inference(
    endpoint_id: str,
    prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference via a registered CLI endpoint."""
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        return _error_result("endpoint_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error_result("prompt must be a non-empty string")

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            execute_cli_inference,
        )

        result = await execute_cli_inference(
            endpoint_id=endpoint_id.strip(),
            prompt=prompt.strip(),
            model=model,
        )
        return _normalize_payload(result)
    except Exception as exc:
        logger.warning("cli_inference delegate failed: %s", exc)

    return _error_result("CLI inference unavailable", endpoint_id=endpoint_id)


async def get_cli_providers() -> Dict[str, Any]:
    """Get available CLI inference provider configurations."""
    cli_providers = _API.get("CLI_PROVIDERS", {})
    return _normalize_payload(
        {"providers": dict(cli_providers), "count": len(cli_providers)}
    )


async def get_cli_config(cli_type: str) -> Dict[str, Any]:
    """Get configuration details for a CLI inference provider."""
    if not isinstance(cli_type, str) or not cli_type.strip():
        return _error_result("cli_type must be a non-empty string")

    cli_providers = _API.get("CLI_PROVIDERS", {})
    entry = cli_providers.get(cli_type.strip())
    if entry is None:
        return _error_result(f"CLI provider '{cli_type}' not found", cli_type=cli_type)
    return _normalize_payload({"cli_type": cli_type.strip(), **entry})


async def get_cli_install(cli_type: str) -> Dict[str, Any]:
    """Get installation instructions for a CLI inference provider."""
    if not isinstance(cli_type, str) or not cli_type.strip():
        return _error_result("cli_type must be a non-empty string")

    install_hints: Dict[str, str] = {
        "claude_cli": "Install via: npm install -g @anthropic-ai/claude-code",
        "openai_cli": "Install via: pip install openai",
        "gemini_cli": "Install via: npm install -g @google/generative-ai",
        "vscode_cli": "Install VS Code and the GitHub Copilot extension",
    }
    hint = install_hints.get(cli_type.strip())
    if hint is None:
        return _error_result(
            f"No install instructions for '{cli_type}'", cli_type=cli_type
        )
    return _normalize_payload({"cli_type": cli_type.strip(), "install_instructions": hint})


async def validate_cli_config(endpoint_id: str) -> Dict[str, Any]:
    """Validate configuration for a registered CLI endpoint."""
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        return _error_result("endpoint_id must be a non-empty string")

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            CLI_ADAPTER_REGISTRY,
        )

        adapter = CLI_ADAPTER_REGISTRY.get(endpoint_id.strip())
        if adapter is None:
            return _error_result(
                f"Endpoint '{endpoint_id}' not found", endpoint_id=endpoint_id
            )
        return _normalize_payload(
            {"endpoint_id": endpoint_id.strip(), "valid": True, "config": str(adapter)}
        )
    except Exception as exc:
        logger.warning("validate_cli_config delegate failed: %s", exc)

    return _error_result("CLI adapter registry unavailable", endpoint_id=endpoint_id)


async def check_cli_version(endpoint_id: str) -> Dict[str, Any]:
    """Check the installed version of a CLI inference tool."""
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        return _error_result("endpoint_id must be a non-empty string")

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            CLI_ADAPTER_REGISTRY,
        )

        adapter = CLI_ADAPTER_REGISTRY.get(endpoint_id.strip())
        if adapter is not None and hasattr(adapter, "get_version"):
            version = adapter.get_version()
            return _normalize_payload(
                {"endpoint_id": endpoint_id.strip(), "version": version}
            )
    except Exception as exc:
        logger.warning("check_cli_version delegate failed: %s", exc)

    return _normalize_payload(
        {
            "endpoint_id": endpoint_id.strip(),
            "version": "unknown",
            "message": "Version check unavailable",
        }
    )


def register_native_enhanced_inference_tools(manager: Any) -> None:
    """Register native enhanced inference tools in the unified hierarchical manager."""
    manager.register_tool(
        category="enhanced_inference_tools",
        name="enhanced_inference_inventory",
        func=enhanced_inference_inventory,
        description="Return inventory metadata for enhanced inference tools.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="multiplex_inference",
        func=multiplex_inference,
        description="Run inference via the multiplexed endpoint registry (local, API, or CLI).",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Input prompt."},
                "model": {"type": "string", "default": "auto"},
                "task_type": {"type": "string", "default": "text-generation"},
                "provider": {
                    "type": "string",
                    "default": "auto",
                    "description": "Provider: auto, local, openai, anthropic, huggingface.",
                },
                "endpoint_id": {"type": "string", "description": "Specific endpoint ID."},
                "max_tokens": {"type": "integer", "default": 512},
                "temperature": {"type": "number", "default": 0.7},
            },
            "required": ["prompt"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="register_endpoint",
        func=register_endpoint,
        description="Register an inference endpoint in the endpoint registry.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"},
                "endpoint_type": {"type": "string"},
                "model": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["endpoint_id", "endpoint_type", "model"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_endpoint_status",
        func=get_endpoint_status,
        description="Get status of one or all registered inference endpoints.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {
                    "type": "string",
                    "description": "Endpoint ID; omit to list all.",
                }
            },
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="configure_api_provider",
        func=configure_api_provider,
        description="Configure an API inference provider (openai, anthropic, huggingface).",
        input_schema={
            "type": "object",
            "properties": {
                "provider": {"type": "string"},
                "api_key": {"type": "string", "default": ""},
                "base_url": {"type": "string", "default": ""},
                "enabled": {"type": "boolean", "default": True},
            },
            "required": ["provider"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="search_huggingface_models",
        func=search_huggingface_models,
        description="Search HuggingFace model hub for models matching a query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "task": {"type": "string", "description": "Filter by HF task."},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "huggingface"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_queue_status",
        func=get_queue_status,
        description="Get current inference queue status and statistics.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_queue_history",
        func=get_queue_history,
        description="Get inference queue history.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50}
            },
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="register_cli_endpoint",
        func=register_cli_endpoint,
        description="Register a CLI inference endpoint (claude_cli, openai_cli, gemini_cli, vscode_cli).",
        input_schema={
            "type": "object",
            "properties": {
                "cli_type": {"type": "string"},
                "endpoint_id": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["cli_type", "endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="list_cli_endpoints",
        func=list_cli_endpoints,
        description="List registered CLI inference endpoints.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="cli_inference",
        func=cli_inference,
        description="Run inference via a registered CLI endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"},
                "prompt": {"type": "string"},
                "model": {"type": "string"},
            },
            "required": ["endpoint_id", "prompt"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_cli_providers",
        func=get_cli_providers,
        description="Get available CLI inference provider configurations.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_cli_config",
        func=get_cli_config,
        description="Get configuration details for a CLI inference provider.",
        input_schema={
            "type": "object",
            "properties": {
                "cli_type": {"type": "string"}
            },
            "required": ["cli_type"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="get_cli_install",
        func=get_cli_install,
        description="Get installation instructions for a CLI inference provider.",
        input_schema={
            "type": "object",
            "properties": {
                "cli_type": {"type": "string"}
            },
            "required": ["cli_type"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="validate_cli_config",
        func=validate_cli_config,
        description="Validate configuration for a registered CLI endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"}
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
    manager.register_tool(
        category="enhanced_inference_tools",
        name="check_cli_version",
        func=check_cli_version,
        description="Check the installed version of a CLI inference tool.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"}
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "inference", "enhanced", "cli"],
    )
