"""Configuration surface for unified MCP server bootstrap and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List


def env_enabled(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return int(default)


def env_text(name: str, default: str = "") -> str:
    """Parse a text environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return str(default)
    return str(value).strip()


def parse_preload_categories(value: str | None, allowed: List[str]) -> List[str]:
    """Parse preload categories from env/config string.

    Accepts comma-separated category names or the special value `all`.
    Unknown category names are ignored.
    """
    if not value:
        return []

    raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw:
        return []

    if "all" in raw:
        return list(allowed)

    allowed_set = set(allowed)
    return [name for name in raw if name in allowed_set]


@dataclass
class UnifiedMCPServerConfig:
    """Configuration for unified MCP bootstrap behavior."""

    enable_unified_bridge: bool = False
    enable_unified_bootstrap: bool = False
    enable_cid_artifact_emission: bool = False
    enable_ucan_validation: bool = False
    enable_policy_evaluation: bool = False
    enable_policy_audit: bool = False
    enable_monitoring: bool = False
    enable_otel_tracing: bool = False
    enable_prometheus_exporter: bool = False
    enable_prometheus_http_server: bool = False
    enable_secrets_vault: bool = False
    enable_secrets_env_autoload: bool = False
    enable_secrets_env_overwrite: bool = False
    enable_risk_scoring: bool = False
    enable_risk_frontier_execution: bool = False
    otel_service_name: str = "ipfs-mcp-server"
    otel_exporter_endpoint: str = ""
    otel_export_protocol: str = "grpc"
    prometheus_port: int = 9090
    prometheus_namespace: str = "mcp"
    preload_categories: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, *, allowed_preload_categories: List[str]) -> "UnifiedMCPServerConfig":
        """Create config from environment variables.

        Environment variables:
        - `IPFS_MCP_ENABLE_UNIFIED_BRIDGE`
        - `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP`
        - `IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS`
        - `IPFS_MCP_SERVER_ENABLE_UCAN_VALIDATION`
        - `IPFS_MCP_SERVER_ENABLE_POLICY_EVALUATION`
        - `IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT`
        - `IPFS_MCP_SERVER_ENABLE_MONITORING`
        - `IPFS_MCP_SERVER_ENABLE_OTEL_TRACING`
        - `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER`
        - `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_HTTP_SERVER`
        - `IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT`
        - `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD`
        - `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_OVERWRITE`
        - `IPFS_MCP_SERVER_ENABLE_RISK_SCORING`
        - `IPFS_MCP_SERVER_ENABLE_RISK_FRONTIER_EXECUTION`
        - `IPFS_MCP_SERVER_OTEL_SERVICE_NAME`
        - `IPFS_MCP_SERVER_OTEL_EXPORTER_ENDPOINT`
        - `IPFS_MCP_SERVER_OTEL_EXPORT_PROTOCOL`
        - `IPFS_MCP_SERVER_PROMETHEUS_PORT`
        - `IPFS_MCP_SERVER_PROMETHEUS_NAMESPACE`
        - `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES`
        """
        return cls(
            enable_unified_bridge=env_enabled("IPFS_MCP_ENABLE_UNIFIED_BRIDGE", default=False),
            enable_unified_bootstrap=env_enabled("IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP", default=False),
            enable_cid_artifact_emission=env_enabled("IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS", default=False),
            enable_ucan_validation=env_enabled("IPFS_MCP_SERVER_ENABLE_UCAN_VALIDATION", default=False),
            enable_policy_evaluation=env_enabled("IPFS_MCP_SERVER_ENABLE_POLICY_EVALUATION", default=False),
            enable_policy_audit=env_enabled("IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT", default=False),
            enable_monitoring=env_enabled("IPFS_MCP_SERVER_ENABLE_MONITORING", default=False),
            enable_otel_tracing=env_enabled("IPFS_MCP_SERVER_ENABLE_OTEL_TRACING", default=False),
            enable_prometheus_exporter=env_enabled("IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER", default=False),
            enable_prometheus_http_server=env_enabled(
                "IPFS_MCP_SERVER_ENABLE_PROMETHEUS_HTTP_SERVER",
                default=False,
            ),
            enable_secrets_vault=env_enabled("IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT", default=False),
            enable_secrets_env_autoload=env_enabled("IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD", default=False),
            enable_secrets_env_overwrite=env_enabled("IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_OVERWRITE", default=False),
            enable_risk_scoring=env_enabled("IPFS_MCP_SERVER_ENABLE_RISK_SCORING", default=False),
            enable_risk_frontier_execution=env_enabled(
                "IPFS_MCP_SERVER_ENABLE_RISK_FRONTIER_EXECUTION",
                default=False,
            ),
            otel_service_name=env_text("IPFS_MCP_SERVER_OTEL_SERVICE_NAME", "ipfs-mcp-server"),
            otel_exporter_endpoint=env_text("IPFS_MCP_SERVER_OTEL_EXPORTER_ENDPOINT", ""),
            otel_export_protocol=env_text("IPFS_MCP_SERVER_OTEL_EXPORT_PROTOCOL", "grpc") or "grpc",
            prometheus_port=env_int("IPFS_MCP_SERVER_PROMETHEUS_PORT", 9090),
            prometheus_namespace=env_text("IPFS_MCP_SERVER_PROMETHEUS_NAMESPACE", "mcp") or "mcp",
            preload_categories=parse_preload_categories(
                os.environ.get("IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES", ""),
                allowed_preload_categories,
            ),
        )


__all__ = [
    "UnifiedMCPServerConfig",
    "env_enabled",
    "env_int",
    "env_text",
    "parse_preload_categories",
]
