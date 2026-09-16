"""MCP tools for llm_router allocation, CLI sessions, and API key slots."""

from .native_llm_allocation_tools import (
    llm_bind_session_api_key,
    llm_choose_cli_route,
    llm_cli_tools_status,
    llm_ensure_cli_tool,
    llm_get_allocation_session,
    llm_list_api_key_slots,
    llm_migrate_cli_session,
    llm_register_api_key,
    llm_resolve_session,
    llm_router_generate_text,
    register_llm_allocation_tools,
    register_native_llm_allocation_tools,
)

__all__ = [
    "llm_bind_session_api_key",
    "llm_choose_cli_route",
    "llm_cli_tools_status",
    "llm_ensure_cli_tool",
    "llm_get_allocation_session",
    "llm_list_api_key_slots",
    "llm_migrate_cli_session",
    "llm_register_api_key",
    "llm_resolve_session",
    "llm_router_generate_text",
    "register_llm_allocation_tools",
    "register_native_llm_allocation_tools",
]
