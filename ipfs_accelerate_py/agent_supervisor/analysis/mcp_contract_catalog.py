"""Minimal MCP claim-family catalog required by origin/main analysis imports.

This file is absent from ``origin/main`` (``mcp_contract_analysis`` imports it
but the module was not shipped). The enum is the closed family set referenced
by ``PARITY_CLAIM_FAMILIES``.
"""

from __future__ import annotations

from enum import Enum


class McpClaimFamily(str, Enum):
    DESCRIPTOR_SCHEMA_MATCHES = "descriptor_schema_matches"
    ARGUMENTS_PRESERVED = "arguments_preserved"
    RESULT_ENVELOPE_PRESERVED = "result_envelope_preserved"
    POLICY_BEFORE_EFFECT = "policy_before_effect"
    NO_COMPATIBILITY_BYPASS = "no_compatibility_bypass"
    TRANSPORT_PARITY = "transport_parity"
    DISCOVERY_EXECUTION_PARITY = "discovery_execution_parity"
    FAILURE_PARITY = "failure_parity"
