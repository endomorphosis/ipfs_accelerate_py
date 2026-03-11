"""Registration adapters for incremental MCP migration.

This module captures compatibility registrations into
`ipfs_accelerate_py.mcp_server.HierarchicalToolManager` so migration can proceed
without a big-bang rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .hierarchical_tool_manager import HierarchicalToolManager

logger = logging.getLogger(__name__)


@dataclass
class LegacyToolRecord:
    """Captured tool registration metadata from legacy mcp tool registrars."""

    name: str
    function: Callable[..., Any]
    description: str
    input_schema: Dict[str, Any]
    category: Optional[str] = None
    execution_context: Optional[str] = None
    tags: Optional[List[str]] = None


class LegacyCollectorMCP:
    """Collector that emulates enough MCP API for legacy register_* functions."""

    def __init__(self) -> None:
        self.tools: Dict[str, LegacyToolRecord] = {}

    def register_tool(
        self,
        name: str | None = None,
        function: Callable[..., Any] | None = None,
        description: str = "",
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        tags: List[str] | None = None,
        *,
        category: str | None = None,
        func: Callable[..., Any] | None = None,
        runtime: str | None = None,
    ) -> None:
        """Collect explicit register_tool calls from legacy or hierarchical callers."""
        resolved_name = str(name or "").strip()
        resolved_function = function or func
        resolved_schema = input_schema or {"type": "object", "properties": {}, "required": []}
        resolved_execution_context = execution_context
        if resolved_execution_context is None:
            if runtime == "trio":
                resolved_execution_context = "worker"
            elif runtime == "fastapi":
                resolved_execution_context = "server"

        if not resolved_name:
            raise ValueError("register_tool requires a non-empty name")
        if not callable(resolved_function):
            raise ValueError(f"register_tool requires a callable function for '{resolved_name}'")

        self.tools[resolved_name] = LegacyToolRecord(
            name=resolved_name,
            function=resolved_function,
            description=description,
            input_schema=resolved_schema,
            category=category,
            execution_context=resolved_execution_context,
            tags=tags,
        )

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        tags: List[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator-compatible collector for `@mcp.tool()` style registrations."""

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = str(name or func.__name__)
            tool_description = description or (func.__doc__ or "")
            schema = input_schema or {"type": "object", "properties": {}, "required": []}
            self.register_tool(
                name=tool_name,
                function=func,
                description=tool_description,
                input_schema=schema,
                execution_context=execution_context,
                tags=tags,
            )
            return func

        return _decorator


def collect_legacy_mcp_tools(include_p2p_taskqueue_tools: bool = True) -> Dict[str, LegacyToolRecord]:
    """Collect compatibility tools from canonical native registrars."""

    collector = LegacyCollectorMCP()
    for registrar in _resolve_compatibility_registrars(
        include_p2p_taskqueue_tools=include_p2p_taskqueue_tools
    ):
        try:
            registrar(collector)
        except Exception as exc:
            logger.warning("Compatibility registrar %r failed: %s", registrar, exc)

    return collector.tools


def _resolve_compatibility_registrars(
    *, include_p2p_taskqueue_tools: bool
) -> List[Callable[[Any], None]]:
    """Resolve canonical registrars used by the compatibility adapter."""
    from .tools.admin_tools import register_native_admin_tools
    from .tools.alert_tools import register_native_alert_tools
    from .tools.analysis_tools import register_native_analysis_tools
    from .tools.auth_tools import register_native_auth_tools
    from .tools.background_task_tools import register_native_background_task_tools
    from .tools.bespoke_tools import register_native_bespoke_tools
    from .tools.cache_tools import register_native_cache_tools
    from .tools.cli import register_native_cli_tools
    from .tools.dashboard_tools import register_native_dashboard_tools
    from .tools.data_processing_tools import register_native_data_processing_tools
    from .tools.dataset_tools import register_native_dataset_tools
    from .tools.development_tools import register_native_development_tools
    from .tools.discord_tools import register_native_discord_tools
    from .tools.embedding_tools import register_native_embedding_tools
    from .tools.email_tools import register_native_email_tools
    from .tools.file_converter_tools import register_native_file_converter_tools
    from .tools.file_detection_tools import register_native_file_detection_tools
    from .tools.finance_data_tools import register_native_finance_data_tools
    from .tools.functions import register_native_function_tools
    from .tools.geospatial_tools import register_native_geospatial_tools
    from .tools.graph_tools import register_native_graph_tools
    from .tools.index_management_tools import register_native_index_management_tools
    from .tools.investigation_tools import register_native_investigation_tools
    from .tools.ipfs import register_native_ipfs_tools
    from .tools.ipfs_cluster_tools import register_native_ipfs_cluster_tools
    from .tools.legal_dataset_tools import register_native_legal_dataset_tools
    from .tools.legacy_mcp_tools import register_native_legacy_mcp_tools
    from .tools.lizardperson_argparse_programs import register_native_lizardperson_argparse_programs
    from .tools.lizardpersons_function_tools import register_native_lizardpersons_function_tools
    from .tools.logic_tools import register_native_logic_tools
    from .tools.mcplusplus import register_native_mcplusplus_tools
    from .tools.media_tools import register_native_media_tools
    from .tools.medical_research_scrapers import register_native_medical_research_scrapers
    from .tools.monitoring_tools import register_native_monitoring_tools
    from .tools.p2p import register_native_p2p_tools
    from .tools.p2p_workflow_tools import register_native_p2p_workflow_tools
    from .tools.pdf_tools import register_native_pdf_tools
    from .tools.provenance_tools import register_native_provenance_tools
    from .tools.rate_limiting import register_native_rate_limiting_tools
    from .tools.search_tools import register_native_search_tools
    from .tools.security_tools import register_native_security_tools
    from .tools.session_tools import register_native_session_tools
    from .tools.software_engineering_tools import register_native_software_engineering_tools
    from .tools.sparse_embedding_tools import register_native_sparse_embedding_tools
    from .tools.storage_tools import register_native_storage_tools
    from .tools.vector_store_tools import register_native_vector_store_tools
    from .tools.vector_tools import register_native_vector_tools
    from .tools.web_archive_tools import register_native_web_archive_tools
    from .tools.web_scraping_tools import register_native_web_scraping_tools
    from .tools.workflow import register_native_workflow_tools

    registrars: List[Callable[[Any], None]] = [
        register_native_admin_tools,
        register_native_alert_tools,
        register_native_analysis_tools,
        register_native_auth_tools,
        register_native_background_task_tools,
        register_native_bespoke_tools,
        register_native_cache_tools,
        register_native_cli_tools,
        register_native_dashboard_tools,
        register_native_data_processing_tools,
        register_native_dataset_tools,
        register_native_development_tools,
        register_native_discord_tools,
        register_native_embedding_tools,
        register_native_email_tools,
        register_native_file_converter_tools,
        register_native_file_detection_tools,
        register_native_finance_data_tools,
        register_native_function_tools,
        register_native_geospatial_tools,
        register_native_graph_tools,
        register_native_index_management_tools,
        register_native_investigation_tools,
        register_native_ipfs_cluster_tools,
        register_native_ipfs_tools,
        register_native_legal_dataset_tools,
        register_native_legacy_mcp_tools,
        register_native_lizardperson_argparse_programs,
        register_native_lizardpersons_function_tools,
        register_native_logic_tools,
        register_native_mcplusplus_tools,
        register_native_media_tools,
        register_native_medical_research_scrapers,
        register_native_monitoring_tools,
        register_native_p2p_workflow_tools,
        register_native_pdf_tools,
        register_native_provenance_tools,
        register_native_rate_limiting_tools,
        register_native_search_tools,
        register_native_security_tools,
        register_native_session_tools,
        register_native_software_engineering_tools,
        register_native_sparse_embedding_tools,
        register_native_storage_tools,
        register_native_vector_store_tools,
        register_native_vector_tools,
        register_native_web_archive_tools,
        register_native_web_scraping_tools,
        register_native_workflow_tools,
    ]
    if include_p2p_taskqueue_tools:
        registrars.append(register_native_p2p_tools)
    return registrars


def register_legacy_tools_into_manager(
    manager: HierarchicalToolManager,
    default_category: str = "legacy_mcp",
    include_p2p_taskqueue_tools: bool = True,
) -> int:
    """Register collected legacy tools into the hierarchical manager.

    Category mapping heuristic:
    - If a tool has a prefix like `github_foo`, category = `github`.
    - Else category = `default_category`.
    """
    records = collect_legacy_mcp_tools(include_p2p_taskqueue_tools=include_p2p_taskqueue_tools)

    count = 0
    for record in records.values():
        category = record.category or _category_from_tool_name(record.name, fallback=default_category)
        manager.register_tool(
            category=category,
            name=record.name,
            func=record.function,
            description=record.description,
            input_schema=record.input_schema,
            runtime=_runtime_from_execution_context(record.execution_context),
            tags=record.tags,
        )
        count += 1

    logger.info("Registered %d legacy tools into hierarchical manager", count)
    return count


def _category_from_tool_name(name: str, fallback: str) -> str:
    """Infer category from tool naming convention."""
    allowed_prefixes = {
        "github",
        "docker",
        "hardware",
        "runner",
        "ipfs",
        "network",
        "model",
        "models",
        "inference",
        "endpoint",
        "endpoints",
        "status",
        "workflow",
        "workflows",
        "dashboard",
        "manifest",
        "p2p",
    }
    if "_" in name:
        prefix = name.split("_", 1)[0].strip().lower()
        if prefix in allowed_prefixes:
            return prefix
    return fallback


def _runtime_from_execution_context(execution_context: Optional[str]) -> Optional[str]:
    """Map legacy execution context to runtime-router values."""
    if execution_context == "worker":
        return "trio"
    if execution_context == "server":
        return "fastapi"
    return None


__all__ = [
    "LegacyCollectorMCP",
    "LegacyToolRecord",
    "collect_legacy_mcp_tools",
    "register_legacy_tools_into_manager",
]
