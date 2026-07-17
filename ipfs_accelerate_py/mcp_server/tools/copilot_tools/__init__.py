"""Copilot-tools category for unified mcp_server."""

from .native_copilot_tools import (
    copilot_explain_command,
    copilot_sdk_create_session,
    copilot_sdk_destroy_session,
    copilot_sdk_get_tools,
    copilot_sdk_list_sessions,
    copilot_sdk_send_message,
    copilot_sdk_stream_message,
    copilot_suggest_command,
    copilot_suggest_git_command,
    register_native_copilot_tools,
)

__all__ = [
    "copilot_suggest_command",
    "copilot_explain_command",
    "copilot_suggest_git_command",
    "copilot_sdk_create_session",
    "copilot_sdk_send_message",
    "copilot_sdk_stream_message",
    "copilot_sdk_destroy_session",
    "copilot_sdk_list_sessions",
    "copilot_sdk_get_tools",
    "register_native_copilot_tools",
]
