"""Shared constants for the MCP-over-libp2p (mcp+p2p) transport.

This module intentionally contains *no* imports from other ipfs_accelerate_py
subsystems so it can be safely imported from both the TaskQueue transport layer
and the unified MCP server facade without creating circular imports.
"""

from __future__ import annotations

# Draft MCP++ transport binding protocol id.
PROTOCOL_MCP_P2P_V1 = "/mcp+p2p/1.0.0"
