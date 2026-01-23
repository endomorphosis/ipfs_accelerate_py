"""
IPFS Accelerate MCP Resources

This package provides resources for the IPFS Accelerate MCP server.
"""

import os
import logging
from typing import Any

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.resources")

def register_all_resources(mcp: Any) -> None:
    """
    Register all resources with the MCP server
    
    This function registers all resources with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering all resources with MCP server")
    
    try:
        # Register model information resources
        from .model_info import register_model_info_resources
        register_model_info_resources(mcp)

        # Register configuration/system resources
        try:
            from .config import register_config_resources
            register_config_resources(mcp)
        except Exception as e:
            logger.warning(f"Config resources not registered: {e}")

        # Lightweight inline resources needed by tools and prompts
        try:
            # Server config used by status tools
            mcp.register_resource(
                uri="server_config",
                function=lambda: {
                    "host": os.getenv("MCP_HOST", "localhost"),
                    "port": int(os.getenv("MCP_PORT", "8000")),
                },
                description="Server host/port configuration",
            )

            # Endpoints config used by endpoints tools
            mcp.register_resource(
                uri="endpoints_config",
                function=lambda: {
                    "max_endpoints": int(os.getenv("MCP_MAX_ENDPOINTS", "25"))
                },
                description="Endpoint manager configuration",
            )

            # Basic models config used by inference tools
            def _models_cfg():
                from .model_info import get_default_supported_models
                sm = get_default_supported_models()
                categories = sm.get("categories", {})
                # Build a simple name→type map for quick lookup
                models = { }
                for mtype, cat in categories.items():
                    for m in cat.get("models", []):
                        models.setdefault(mtype, []).append(m.get("name"))
                return models

            mcp.register_resource(
                uri="models_config",
                function=_models_cfg,
                description="Simplified models configuration by category",
            )

            # Inference config used by download_model
            mcp.register_resource(
                uri="inference_config",
                function=lambda: {
                    "model_cache_dir": os.path.expanduser(os.getenv("IPFS_ACCEL_MODEL_CACHE", "~/.ipfs_accelerate/models"))
                },
                description="Inference-related configuration",
            )

            # Provide resolver for model info lookups used by tools
            def _get_model_info(model_name: str):
                from .model_info import get_default_supported_models
                data = get_default_supported_models()
                for mtype, cat in data.get("categories", {}).items():
                    for m in cat.get("models", []):
                        if m.get("name") == model_name:
                            info = dict(m)
                            info["type"] = "embedding" if "embed" in model_name.lower() or "clip" in model_name.lower() else (
                                "generation" if "llama" in model_name.lower() else "unknown"
                            )
                            return info
                return None

            # StandaloneMCP exposes access_resource(name, **kwargs), so register callable under key used by tools
            mcp.register_resource(
                uri="get_model_info",
                function=_get_model_info,
                description="Lookup for a single model's info",
            )
        except Exception as e:
            logger.warning(f"Inline resources setup encountered an issue: {e}")

        logger.debug("All resources registered with MCP server")

    except Exception as e:
        logger.error(f"Error registering resources with MCP server: {e}")
        raise
