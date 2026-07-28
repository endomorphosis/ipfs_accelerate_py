"""Goose ACP (Agent Client Protocol) stdio transport package.

Provides a managed client for persistent ``goose acp`` sessions with
bounded restarts, request correlation, and endpoint-local session state.

Importing this package does not start processes or open network listeners.
``goose serve`` and dangerously unauthenticated network modes are not exposed.
"""

from __future__ import annotations

from .goose_client import (
    ACP_PROTOCOL_VERSION,
    ACPBounds,
    ACPCapacityError,
    ACPClientState,
    ACPError,
    ACPNotReadyError,
    ACPProtocolError,
    ACPRestartExhaustedError,
    ACPRestartPolicy,
    ACPSessionRecord,
    ACPUncertainSideEffectError,
    CLIENT_NAME,
    CLIENT_VERSION,
    DEFAULT_MAX_IDLE_SECONDS,
    DEFAULT_MAX_PENDING_REQUESTS,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_MAX_SESSIONS,
    FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
    STATUS_UNCERTAIN_SIDE_EFFECT,
    GooseACPClient,
    build_text_prompt_blocks,
    create_goose_acp_client,
    encode_acp_message,
    parse_acp_line,
    split_ndjson_buffer,
)

__all__ = [
    "ACP_PROTOCOL_VERSION",
    "ACPBounds",
    "ACPCapacityError",
    "ACPClientState",
    "ACPError",
    "ACPNotReadyError",
    "ACPProtocolError",
    "ACPRestartExhaustedError",
    "ACPRestartPolicy",
    "ACPSessionRecord",
    "ACPUncertainSideEffectError",
    "CLIENT_NAME",
    "CLIENT_VERSION",
    "DEFAULT_MAX_IDLE_SECONDS",
    "DEFAULT_MAX_PENDING_REQUESTS",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_MAX_SESSIONS",
    "FAILURE_KIND_UNCERTAIN_SIDE_EFFECT",
    "STATUS_UNCERTAIN_SIDE_EFFECT",
    "GooseACPClient",
    "build_text_prompt_blocks",
    "create_goose_acp_client",
    "encode_acp_message",
    "parse_acp_line",
    "split_ndjson_buffer",
]
