"""vLLM inference tools category for unified MCP server."""

from .native_vllm_tools import register_native_vllm_tools

__all__ = ["register_native_vllm_tools"]
