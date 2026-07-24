"""Canonical bootstrap server for MCP unification.

This module provides the stable runtime construction path for the unified MCP
server package and attaches canonical bootstrap services when enabled.
"""

from __future__ import annotations

import time
from typing import Any

from .logger import configure_root_logging, get_logger

configure_root_logging()

from .configs import UnifiedMCPServerConfig, parse_preload_categories
from .hierarchical_tool_manager import HierarchicalToolManager
from .runtime_router import RuntimeRouter
from .dispatch_pipeline import (
    coerce_dispatch_bool,
    coerce_dispatch_dict,
    coerce_dispatch_list,
    compute_dispatch_intent_cid,
    normalize_dispatch_parameters,
)
from .server_context import UnifiedServerContext
from .wave_a_loaders import configure_wave_a_loaders
from .tools.idl import load_idl_tools
from .tools.admin_tools import register_native_admin_tools
from .tools.alert_tools import register_native_alert_tools
from .tools.analysis_tools import register_native_analysis_tools
from .tools.auth_tools import register_native_auth_tools
from .tools.audit_tools import register_native_audit_tools
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
from .tools.finance_data_tools import register_native_finance_data_tools
from .tools.file_converter_tools import register_native_file_converter_tools
from .tools.geospatial_tools import register_native_geospatial_tools
from .tools.graph_tools import register_native_graph_tools
from .tools.file_detection_tools import register_native_file_detection_tools
from .tools.functions import register_native_function_tools
from .tools.index_management_tools import register_native_index_management_tools
from .tools.ipfs_cluster_tools import register_native_ipfs_cluster_tools
from .tools.ipfs_tools import register_native_ipfs_tools_category
from .tools.investigation_tools import register_native_investigation_tools
from .tools.legal_dataset_tools import register_native_legal_dataset_tools
from .tools.legacy_mcp_tools import register_native_legacy_mcp_tools
from .tools.lizardperson_argparse_programs import register_native_lizardperson_argparse_programs
from .tools.lizardpersons_function_tools import register_native_lizardpersons_function_tools
from .tools.logic_tools import register_native_logic_tools
from .tools.media_tools import register_native_media_tools
from .tools.mcplusplus import register_native_mcplusplus_tools
from .tools.medical_research_scrapers import register_native_medical_research_scrapers
from .tools.monitoring_tools import register_native_monitoring_tools
from .tools.p2p_workflow_tools import register_native_p2p_workflow_tools
from .tools.p2p_tools import register_native_p2p_tools_category
from .tools.pdf_tools import register_native_pdf_tools
from .tools.provenance_tools import register_native_provenance_tools
from .tools.search_tools import register_native_search_tools
from .tools.security_tools import register_native_security_tools
from .tools.session_tools import register_native_session_tools
from .tools.sparse_embedding_tools import register_native_sparse_embedding_tools
from .tools.software_engineering_tools import register_native_software_engineering_tools
from .tools.docker_tools import register_native_docker_tools
from .tools.storage_tools import register_native_storage_tools
from .tools.vector_store_tools import register_native_vector_store_tools
from .tools.vector_tools import register_native_vector_tools
from .tools.web_archive_tools import register_native_web_archive_tools
from .tools.web_scraping_tools import register_native_web_scraping_tools
from .tools.workflow_tools import register_native_workflow_tools_category
from .tools.vllm_tools import register_native_vllm_tools
from .tools.rate_limiting import register_native_rate_limiting_tools
from .tools.rate_limiting_tools import register_native_rate_limiting_tools_category
# Migrated legacy mcp/tools categories
from .tools.hardware_tools import register_native_hardware_tools
from .tools.inference_tools import register_native_inference_tools
from .tools.model_tools import register_native_model_tools
from .tools.acceleration_tools import register_native_acceleration_tools
from .tools.backend_management_tools import register_native_backend_management_tools
from .tools.endpoint_tools import register_native_endpoint_tools
from .tools.github_tools import register_native_github_tools
from .tools.copilot_tools import register_native_copilot_tools
from .tools.cli_endpoint_tools import register_native_cli_endpoint_tools
from .tools.ipfs_network_tools import register_native_ipfs_network_tools
from .tools.enhanced_inference_tools import register_native_enhanced_inference_tools
from .tools.workflow_management_tools import register_native_workflow_management_tools
from .tools.shared_tools import register_native_shared_tools
from .mcplusplus.artifacts import ArtifactStore, build_decision, compute_artifact_cid, envelope_from_payloads
from .mcplusplus.delegation import validate_raw_delegation_chain
from .mcplusplus.policy_engine import evaluate_with_ipfs_datasets_policy
from .mcplusplus.event_dag import EventDAGStore
from .mcplusplus.risk_scheduler import RiskScheduler
from .risk_scorer import RiskScorer, RiskScoringPolicy
from .policy_audit_log import PolicyAuditLog
from .audit_metrics_bridge import connect_audit_to_prometheus
from .secrets_vault import SecretsVault
from .monitoring import EnhancedMetricsCollector, P2PMetricsCollector
from .otel_tracing import MCPTracer, configure_tracing
from .prometheus_exporter import PrometheusExporter
from .validators import validate_dispatch_inputs
from .exceptions import ValidationError

logger = get_logger(__name__)


async def _invoke_maybe_async(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Invoke callable and await it when it returns an awaitable."""
    result = func(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result


def get_unified_meta_tool_names() -> list[str]:
    """Return canonical control-plane meta-tool names for unified MCP."""
    return [
        "tools_list_categories",
        "tools_list_tools",
        "tools_get_schema",
        "tools_dispatch",
        "tools_runtime_metrics",
    ]


def get_unified_wave_a_categories() -> list[str]:
    """Return canonical Wave A categories supported by unified bootstrap."""
    return ["ipfs", "workflow", "p2p"]


def get_unified_supported_profiles() -> list[str]:
    """Return MCP++ profile capabilities advertised by unified bootstrap."""
    return [
        "mcp++/idl",
        "mcp++/cid-envelope",
        "mcp++/ucan",
        "mcp++/deontic-policy",
        "mcp++/p2p-transport",
        "mcp++/risk-scheduling",
    ]


def _parse_preload_categories(value: str | None) -> list[str]:
    """Parse preload categories from env var.

    Accepts comma-separated category names or the special value `all`.
    """
    return parse_preload_categories(value, get_unified_wave_a_categories())


def _preload_configured_categories(manager: HierarchicalToolManager, preload_categories: list[str]) -> list[str]:
    """Preload selected categories (if configured) by triggering list_tools()."""
    loaded: list[str] = []
    for category in preload_categories:
        try:
            manager.list_tools(category)
            loaded.append(category)
        except Exception as exc:
            logger.warning("Failed to preload unified category '%s': %s", category, exc)

    return loaded


def _build_unified_services() -> dict[str, Any]:
    """Build lazy MCP++ service factories for unified runtime composition.

    Services are attached as callables to avoid heavy startup side-effects from
    optional runtime dependencies.
    """
    return {
        "task_queue_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_task_queue"]
        ).create_task_queue(**kwargs),
        "workflow_scheduler_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_workflow_scheduler"]
        ).create_workflow_scheduler(**kwargs),
        "workflow_engine_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["WorkflowEngine"]
        ).WorkflowEngine(**kwargs),
        "workflow_dag_executor_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["WorkflowDAGExecutor"]
        ).WorkflowDAGExecutor(**kwargs),
        "peer_registry_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_peer_registry"]
        ).create_peer_registry(**kwargs),
        "peer_bootstrap_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_peer_bootstrap"]
        ).create_peer_bootstrap(**kwargs),
        "peer_discovery_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_peer_discovery"]
        ).create_peer_discovery(**kwargs),
        "result_cache_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["ResultCache", "MemoryCacheBackend"]
        ).ResultCache(backend=__import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["MemoryCacheBackend"]
        ).MemoryCacheBackend(), **kwargs),
        "risk_scheduler_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["RiskScheduler"]
        ).RiskScheduler(**kwargs),
    }


def _attach_unified_bootstrap(server: Any, config: UnifiedMCPServerConfig) -> None:
    """Attach unified migration components to a legacy MCP server instance.

    This is intentionally non-invasive: no legacy registration paths are replaced.
    The unified components are attached as attributes for incremental integration.
    """
    runtime_router = RuntimeRouter(default_runtime="fastapi")
    manager = HierarchicalToolManager(runtime_router=runtime_router)
    event_store = EventDAGStore()
    artifact_store_backend = str(config.artifact_store_backend or "memory")
    artifact_store_path = str(config.artifact_store_path or "")
    if artifact_store_backend == "json" and artifact_store_path:
        artifact_store = ArtifactStore.load_json(artifact_store_path)
    else:
        artifact_store = ArtifactStore()
    artifact_store_runtime_meta: dict[str, Any] = {
        "backend": artifact_store_backend,
        "path": artifact_store_path,
        "durable": artifact_store_backend == "json" and bool(artifact_store_path),
        "loaded": int((artifact_store.stats() or {}).get("artifact_count") or 0),
    }
    if artifact_store_backend == "json" and not artifact_store_path:
        artifact_store_runtime_meta["warning"] = "artifact_store_path_required"

    # Audit storage: persists event_dag entries to ipfs_kit_py or disk using CIDs.
    _audit_storage: Any = None
    try:
        from .ipfs_kit_bridge import get_audit_storage
        _audit_storage = get_audit_storage()
    except Exception:
        pass
    if _audit_storage is None:
        try:
            from ..ipfs_kit_integration import IPFSKitStorage
            _audit_storage = IPFSKitStorage(enable_ipfs_kit=True)
        except Exception:
            _audit_storage = None
    risk_scheduler = None
    risk_scorer = RiskScorer()
    policy_audit = PolicyAuditLog(enabled=config.enable_policy_audit)
    metrics_collector = EnhancedMetricsCollector(enabled=config.enable_monitoring)
    metrics_collector.start_monitoring()
    p2p_metrics_collector = P2PMetricsCollector(base_collector=metrics_collector)
    tracer = MCPTracer()
    tracing_status: dict[str, Any] = {
        "enabled": False,
        "configured": False,
        "service_name": str(config.otel_service_name),
        "endpoint": str(config.otel_exporter_endpoint or ""),
        "protocol": str(config.otel_export_protocol or "grpc"),
        "error": "",
        "info": tracer.get_info(),
    }
    if config.enable_otel_tracing:
        try:
            tracing_status["configured"] = bool(
                configure_tracing(
                    service_name=config.otel_service_name,
                    otlp_endpoint=config.otel_exporter_endpoint or None,
                    export_protocol=config.otel_export_protocol,
                )
            )
            tracing_status["enabled"] = bool(tracing_status["configured"])
        except Exception as exc:
            tracing_status["error"] = str(exc)

    prometheus_exporter = (
        PrometheusExporter(
            collector=metrics_collector,
            port=config.prometheus_port,
            namespace=config.prometheus_namespace,
        )
        if config.enable_prometheus_exporter
        else None
    )
    prometheus_status: dict[str, Any] = {
        "enabled": prometheus_exporter is not None,
        "http_started": False,
        "error": "",
        "info": prometheus_exporter.get_info() if prometheus_exporter is not None else {
            "exporter": "prometheus",
            "namespace": config.prometheus_namespace,
            "port": config.prometheus_port,
            "prometheus_available": False,
            "http_server_running": False,
        },
    }
    if prometheus_exporter is not None and config.enable_prometheus_http_server:
        try:
            prometheus_exporter.start_http_server()
            prometheus_status["http_started"] = True
            prometheus_status["info"] = prometheus_exporter.get_info()
        except Exception as exc:
            prometheus_status["error"] = str(exc)
    audit_metrics_bridge = None
    audit_metrics_status: dict[str, Any] = {
        "enabled": False,
        "attached": False,
        "forwarded_count": 0,
        "error": "",
        "category": "policy",
    }
    if policy_audit.enabled and prometheus_exporter is not None:
        try:
            audit_metrics_bridge = connect_audit_to_prometheus(
                policy_audit,
                prometheus_exporter,
                category="policy",
            )
            audit_metrics_status = {
                "enabled": True,
                **audit_metrics_bridge.get_info(),
                "error": "",
            }
        except Exception as exc:
            audit_metrics_status["enabled"] = True
            audit_metrics_status["error"] = str(exc)
    secrets_vault = SecretsVault() if config.enable_secrets_vault else None
    secrets_status: dict[str, Any] = {
        "attached": secrets_vault is not None,
        "env_autoload_enabled": bool(config.enable_secrets_env_autoload),
        "env_overwrite": bool(config.enable_secrets_env_overwrite),
        "env_loaded": [],
        "error": "",
    }
    if secrets_vault is not None and config.enable_secrets_env_autoload:
        try:
            loaded_names = secrets_vault.load_into_env(overwrite=config.enable_secrets_env_overwrite)
            secrets_status["env_loaded"] = list(loaded_names)
        except Exception as exc:
            secrets_status["error"] = str(exc)
    configure_wave_a_loaders(manager)
    manager.register_category_loader(
        "admin_tools",
        lambda mgr: register_native_admin_tools(mgr),
    )
    manager.register_category_loader(
        "alert_tools",
        lambda mgr: register_native_alert_tools(mgr),
    )
    manager.register_category_loader(
        "audit_tools",
        lambda mgr: register_native_audit_tools(mgr),
    )
    manager.register_category_loader(
        "analysis_tools",
        lambda mgr: register_native_analysis_tools(mgr),
    )
    manager.register_category_loader(
        "auth_tools",
        lambda mgr: register_native_auth_tools(mgr),
    )
    manager.register_category_loader(
        "bespoke_tools",
        lambda mgr: register_native_bespoke_tools(mgr),
    )
    manager.register_category_loader(
        "cache_tools",
        lambda mgr: register_native_cache_tools(mgr),
    )
    manager.register_category_loader(
        "cli",
        lambda mgr: register_native_cli_tools(mgr),
    )
    manager.register_category_loader(
        "background_task_tools",
        lambda mgr: register_native_background_task_tools(mgr),
    )
    manager.register_category_loader(
        "dashboard_tools",
        lambda mgr: register_native_dashboard_tools(mgr),
    )
    manager.register_category_loader(
        "dataset_tools",
        lambda mgr: register_native_dataset_tools(mgr),
    )
    manager.register_category_loader(
        "development_tools",
        lambda mgr: register_native_development_tools(mgr),
    )
    manager.register_category_loader(
        "discord_tools",
        lambda mgr: register_native_discord_tools(mgr),
    )
    manager.register_category_loader(
        "embedding_tools",
        lambda mgr: register_native_embedding_tools(mgr),
    )
    manager.register_category_loader(
        "monitoring_tools",
        lambda mgr: register_native_monitoring_tools(mgr),
    )
    manager.register_category_loader(
        "ipfs_cluster_tools",
        lambda mgr: register_native_ipfs_cluster_tools(mgr),
    )
    manager.register_category_loader(
        "ipfs_tools",
        lambda mgr: register_native_ipfs_tools_category(mgr),
    )
    manager.register_category_loader(
        "investigation_tools",
        lambda mgr: register_native_investigation_tools(mgr),
    )
    manager.register_category_loader(
        "legal_dataset_tools",
        lambda mgr: register_native_legal_dataset_tools(mgr),
    )
    manager.register_category_loader(
        "legacy_mcp_tools",
        lambda mgr: register_native_legacy_mcp_tools(mgr),
    )
    manager.register_category_loader(
        "lizardperson_argparse_programs",
        lambda mgr: register_native_lizardperson_argparse_programs(mgr),
    )
    manager.register_category_loader(
        "lizardpersons_function_tools",
        lambda mgr: register_native_lizardpersons_function_tools(mgr),
    )
    manager.register_category_loader(
        "logic_tools",
        lambda mgr: register_native_logic_tools(mgr),
    )
    manager.register_category_loader(
        "media_tools",
        lambda mgr: register_native_media_tools(mgr),
    )
    manager.register_category_loader(
        "mcplusplus",
        lambda mgr: register_native_mcplusplus_tools(mgr),
    )
    manager.register_category_loader(
        "medical_research_scrapers",
        lambda mgr: register_native_medical_research_scrapers(mgr),
    )
    manager.register_category_loader(
        "p2p_workflow_tools",
        lambda mgr: register_native_p2p_workflow_tools(mgr),
    )
    manager.register_category_loader(
        "p2p_tools",
        lambda mgr: register_native_p2p_tools_category(mgr),
    )
    manager.register_category_loader(
        "pdf_tools",
        lambda mgr: register_native_pdf_tools(mgr),
    )
    manager.register_category_loader(
        "functions",
        lambda mgr: register_native_function_tools(mgr),
    )
    manager.register_category_loader(
        "workflow_tools",
        lambda mgr: register_native_workflow_tools_category(mgr),
    )
    manager.register_category_loader(
        "sparse_embedding_tools",
        lambda mgr: register_native_sparse_embedding_tools(mgr),
    )
    manager.register_category_loader(
        "web_scraping_tools",
        lambda mgr: register_native_web_scraping_tools(mgr),
    )
    manager.register_category_loader(
        "data_processing_tools",
        lambda mgr: register_native_data_processing_tools(mgr),
    )
    manager.register_category_loader(
        "email_tools",
        lambda mgr: register_native_email_tools(mgr),
    )
    manager.register_category_loader(
        "finance_data_tools",
        lambda mgr: register_native_finance_data_tools(mgr),
    )
    manager.register_category_loader(
        "file_converter_tools",
        lambda mgr: register_native_file_converter_tools(mgr),
    )
    manager.register_category_loader(
        "file_detection_tools",
        lambda mgr: register_native_file_detection_tools(mgr),
    )
    manager.register_category_loader(
        "geospatial_tools",
        lambda mgr: register_native_geospatial_tools(mgr),
    )
    manager.register_category_loader(
        "graph_tools",
        lambda mgr: register_native_graph_tools(mgr),
    )
    manager.register_category_loader(
        "index_management_tools",
        lambda mgr: register_native_index_management_tools(mgr),
    )
    manager.register_category_loader(
        "provenance_tools",
        lambda mgr: register_native_provenance_tools(mgr),
    )
    manager.register_category_loader(
        "security_tools",
        lambda mgr: register_native_security_tools(mgr),
    )
    manager.register_category_loader(
        "search_tools",
        lambda mgr: register_native_search_tools(mgr),
    )
    manager.register_category_loader(
        "software_engineering_tools",
        lambda mgr: register_native_software_engineering_tools(mgr),
    )
    manager.register_category_loader(
        "docker_tools",
        lambda mgr: register_native_docker_tools(mgr),
    )
    manager.register_category_loader(
        "session_tools",
        lambda mgr: register_native_session_tools(mgr),
    )
    manager.register_category_loader(
        "storage_tools",
        lambda mgr: register_native_storage_tools(mgr),
    )
    manager.register_category_loader(
        "vector_store_tools",
        lambda mgr: register_native_vector_store_tools(mgr),
    )
    manager.register_category_loader(
        "vector_tools",
        lambda mgr: register_native_vector_tools(mgr),
    )
    manager.register_category_loader(
        "web_archive_tools",
        lambda mgr: register_native_web_archive_tools(mgr),
    )
    manager.register_category_loader(
        "rate_limiting",
        lambda mgr: register_native_rate_limiting_tools(mgr),
    )
    manager.register_category_loader(
        "rate_limiting_tools",
        lambda mgr: register_native_rate_limiting_tools_category(mgr),
    )
    manager.register_category_loader(
        "vllm_tools",
        lambda mgr: register_native_vllm_tools(mgr),
    )
    # Migrated legacy mcp/tools categories
    manager.register_category_loader(
        "hardware_tools",
        lambda mgr: register_native_hardware_tools(mgr),
    )
    manager.register_category_loader(
        "inference_tools",
        lambda mgr: register_native_inference_tools(mgr),
    )
    manager.register_category_loader(
        "model_tools",
        lambda mgr: register_native_model_tools(mgr),
    )
    manager.register_category_loader(
        "acceleration_tools",
        lambda mgr: register_native_acceleration_tools(mgr),
    )
    manager.register_category_loader(
        "backend_management_tools",
        lambda mgr: register_native_backend_management_tools(mgr),
    )
    manager.register_category_loader(
        "endpoint_tools",
        lambda mgr: register_native_endpoint_tools(mgr),
    )
    manager.register_category_loader(
        "github_tools",
        lambda mgr: register_native_github_tools(mgr),
    )
    manager.register_category_loader(
        "copilot_tools",
        lambda mgr: register_native_copilot_tools(mgr),
    )
    manager.register_category_loader(
        "cli_endpoint_tools",
        lambda mgr: register_native_cli_endpoint_tools(mgr),
    )
    manager.register_category_loader(
        "ipfs_network_tools",
        lambda mgr: register_native_ipfs_network_tools(mgr),
    )
    manager.register_category_loader(
        "enhanced_inference_tools",
        lambda mgr: register_native_enhanced_inference_tools(mgr),
    )
    manager.register_category_loader(
        "workflow_management_tools",
        lambda mgr: register_native_workflow_management_tools(mgr),
    )
    manager.register_category_loader(
        "shared_tools",
        lambda mgr: register_native_shared_tools(mgr),
    )
    manager.register_category_loader(
        "idl",
        lambda mgr: load_idl_tools(mgr, supported_capabilities=get_unified_supported_profiles()),
    )
    preloaded_categories = _preload_configured_categories(manager, config.preload_categories)

    async def _bind_frontier_execution(item: Any) -> dict[str, Any]:
        """Bind a frontier item to workflow/task queue execution adapters."""
        binding = {
            "attempted": False,
            "scheduled": False,
            "route": "",
            "workflow_id": "",
            "task_id": "",
            "event_cid": "",
            "error": "",
        }

        if item is None:
            binding["error"] = "frontier_empty"
            return binding

        binding["attempted"] = True
        binding["event_cid"] = str(getattr(item, "event_cid", "") or "")

        services = getattr(server, "_unified_services", {})
        task_payload = {
            "event_cid": str(getattr(item, "event_cid", "") or ""),
            "actor": str(getattr(item, "actor", "") or "*"),
            "priority": float(getattr(item, "priority", 0.0) or 0.0),
            "metadata": dict(getattr(item, "metadata", {}) or {}),
            "source": "mcpplusplus.risk_frontier",
        }

        scheduler_factory = services.get("workflow_scheduler_factory") if isinstance(services, dict) else None
        if callable(scheduler_factory):
            try:
                scheduler = scheduler_factory()
                if scheduler is not None:
                    submit_fn = None
                    for method_name in ("submit_workflow", "create_workflow", "submit"):
                        maybe = getattr(scheduler, method_name, None)
                        if callable(maybe):
                            submit_fn = maybe
                            break

                    if submit_fn is not None:
                        workflow_result = await _invoke_maybe_async(
                            submit_fn,
                            workflow_name="risk_frontier_dispatch",
                            tasks=[
                                {
                                    "task_type": "mcp.frontier.execute",
                                    "payload": task_payload,
                                }
                            ],
                            metadata={
                                "event_cid": task_payload["event_cid"],
                                "risk_priority": task_payload["priority"],
                            },
                        )

                        workflow_id = ""
                        if isinstance(workflow_result, str):
                            workflow_id = workflow_result
                        elif isinstance(workflow_result, dict):
                            workflow_id = str(workflow_result.get("workflow_id") or "")

                        if workflow_id:
                            binding.update(
                                {
                                    "scheduled": True,
                                    "route": "workflow_scheduler",
                                    "workflow_id": workflow_id,
                                }
                            )
                            return binding
            except Exception as exc:
                binding["error"] = str(exc)

        task_queue_factory = services.get("task_queue_factory") if isinstance(services, dict) else None
        if callable(task_queue_factory):
            try:
                task_queue = task_queue_factory()
                submit = getattr(task_queue, "submit", None)
                if callable(submit):
                    task_id = await _invoke_maybe_async(
                        submit,
                        task_type="mcp_frontier_event",
                        payload=task_payload,
                        priority=int(round(float(task_payload["priority"]) * 1000.0)),
                    )
                    if task_id:
                        binding.update(
                            {
                                "scheduled": True,
                                "route": "task_queue",
                                "task_id": str(task_id),
                                "error": "",
                            }
                        )
                        return binding
                    binding["error"] = binding["error"] or "task_queue_submit_failed"
            except Exception as exc:
                binding["error"] = str(exc)

        if not binding["error"]:
            binding["error"] = "no_scheduler_or_task_queue"
        return binding

    async def tools_list_categories() -> dict[str, Any]:
        return {"categories": manager.list_categories()}

    async def tools_list_tools(category: str) -> dict[str, Any]:
        return {"category": category, "tools": manager.list_tools(category)}

    async def tools_get_schema(category: str, tool_name: str) -> dict[str, Any]:
        return manager.get_tool_schema(category, tool_name)

    async def tools_dispatch(category: str, tool_name: str, parameters: dict[str, Any]) -> Any:
        try:
            category, tool_name, payload = validate_dispatch_inputs(
                category=category,
                tool_name=tool_name,
                parameters=parameters,
            )
        except (ValueError, ValidationError) as exc:
            return {
                "ok": False,
                "error": "invalid_dispatch_parameter",
                "details": str(exc),
            }

        dispatch_started = time.perf_counter()

        def _record_observability(status: str) -> None:
            latency_seconds = max(0.0, time.perf_counter() - dispatch_started)
            success = status == "success"
            metrics_collector.track_tool_execution(
                tool_name=f"{category}.{tool_name}",
                execution_time_ms=latency_seconds * 1000.0,
                success=success,
            )
            if prometheus_exporter is not None:
                prometheus_exporter.record_tool_call(
                    category=category,
                    tool=tool_name,
                    status=status,
                    latency_seconds=latency_seconds,
                )
                prometheus_exporter.update()

        def _build_success_response(
            *,
            result: Any,
            risk_record: Any,
            risk_assessment_obj: Any,
            policy_obj: Any = None,
            policy_decision_obj: dict[str, Any] | None = None,
            passthrough_result_fields: bool = False,
            extra_fields: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            response: dict[str, Any] = {
                "ok": True,
                "result": result,
                "risk_assessment": risk_assessment_obj.to_dict() if risk_assessment_obj is not None else None,
                "risk": risk_record.to_dict(),
                "audit": policy_audit.stats() if policy_audit.enabled else None,
            }
            if policy_obj is not None:
                response["policy"] = policy_obj.to_dict()
                response["policy_decision"] = dict(policy_decision_obj or {})
            if passthrough_result_fields and isinstance(result, dict):
                for key, value in result.items():
                    response.setdefault(str(key), value)
            if isinstance(extra_fields, dict) and extra_fields:
                response.update(extra_fields)
            return response

        def _persist_artifact_store_snapshot() -> dict[str, Any]:
            meta = dict(artifact_store_runtime_meta)
            if meta.get("backend") != "json":
                meta["saved"] = 0
                return meta
            path = str(meta.get("path") or "")
            if not path:
                raise ValueError("artifact_store_path_required")
            meta["saved"] = int(artifact_store.save_json(path))
            meta["loaded"] = int((artifact_store.stats() or {}).get("artifact_count") or 0)
            return meta

        dispatch_intent_cid = compute_dispatch_intent_cid(category, tool_name, payload)

        def _persist_to_audit_storage(cid: str, payload_obj: dict[str, Any]) -> dict[str, Any]:
            """Persist a single artifact blob to audit storage (ipfs_kit / disk) by CID."""
            meta: dict[str, Any] = {"cid": cid, "persisted": False, "backend": "none", "error": ""}
            if _audit_storage is None:
                meta["error"] = "audit_storage_unavailable"
                return meta
            try:
                import json as _json
                blob = _json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
                store_fn = getattr(_audit_storage, "store", None)
                if callable(store_fn):
                    stored_cid = store_fn(blob, cid)
                    meta["persisted"] = True
                    meta["backend"] = "ipfs_kit"
                    meta["stored_cid"] = str(stored_cid or cid)
                    return meta
                # Fallback: write to cache directory on disk
                write_fn = getattr(_audit_storage, "_write_bytes_to_cache", None)
                if callable(write_fn):
                    path = write_fn(cid, blob)
                    meta["persisted"] = True
                    meta["backend"] = "disk"
                    meta["path"] = str(path)
                    return meta
                meta["error"] = "no_suitable_store_method"
            except Exception as exc:
                meta["error"] = str(exc)
            return meta

        def _record_event_dag_always(
            input_payload: dict[str, Any],
            output_payload: dict[str, Any],
            parent_cids: list[str],
        ) -> dict[str, Any]:
            """Compute CIDs, add to in-memory event_store, and persist to audit storage.

            This runs unconditionally on every successful tool invocation so that
            inputs and outputs are always auditable regardless of the emit_artifacts flag.
            """
            dag_meta: dict[str, Any] = {
                "persisted": False,
                "input_cid": "",
                "output_cid": "",
                "event_cid": "",
                "lineage": [],
                "storage": {},
                "error": "",
            }
            try:
                input_cid = compute_artifact_cid(input_payload)
                output_cid = compute_artifact_cid(output_payload)
                minimal_event: dict[str, Any] = {
                    "input_cid": input_cid,
                    "output_cid": output_cid,
                    "intent_cid": dispatch_intent_cid,
                    "category": category,
                    "tool_name": tool_name,
                    "parents": list(parent_cids or []),
                }
                event_cid = compute_artifact_cid(minimal_event)
                minimal_event["event_cid"] = event_cid

                dag_meta["input_cid"] = input_cid
                dag_meta["output_cid"] = output_cid
                dag_meta["event_cid"] = event_cid

                # Add to in-memory event DAG (best-effort; parent validation may fail)
                try:
                    event_store.add_event(event_cid, minimal_event)
                    dag_meta["persisted"] = True
                    dag_meta["lineage"] = event_store.get_lineage(event_cid)
                except Exception as dag_exc:
                    dag_meta["error"] = str(dag_exc)

                # Persist input, output, event to audit storage
                storage_results: dict[str, Any] = {}
                for label, cid_key, obj in (
                    ("input", input_cid, input_payload),
                    ("output", output_cid, output_payload),
                    ("event", event_cid, minimal_event),
                ):
                    storage_results[label] = _persist_to_audit_storage(cid_key, obj)
                dag_meta["storage"] = storage_results
            except Exception as exc:
                dag_meta["error"] = dag_meta.get("error") or str(exc)
            return dag_meta

        try:
            emit_artifacts = coerce_dispatch_bool(
                payload.pop("__emit_artifacts", config.enable_cid_artifact_emission),
                field_name="__emit_artifacts",
            )
            proof_cid = str(payload.pop("__proof_cid", "") or "")
            policy_cid = str(payload.pop("__policy_cid", "") or "")
            policy_version = str(payload.pop("__policy_version", "") or "")
            correlation_id = str(payload.pop("__correlation_id", "") or "")
            enforce_ucan = coerce_dispatch_bool(
                payload.pop("__enforce_ucan", config.enable_ucan_validation),
                field_name="__enforce_ucan",
            )
            ucan_actor = str(payload.pop("__ucan_actor", "") or "")
            ucan_proof_chain = coerce_dispatch_list(
                payload.pop("__ucan_proof_chain", []),
                field_name="__ucan_proof_chain",
            )
            ucan_require_signatures = coerce_dispatch_bool(
                payload.pop("__ucan_require_signatures", False),
                field_name="__ucan_require_signatures",
            )
            ucan_issuer_public_keys = coerce_dispatch_dict(
                payload.pop("__ucan_issuer_public_keys", {}),
                field_name="__ucan_issuer_public_keys",
            )
            ucan_revoked_proof_cids = coerce_dispatch_list(
                payload.pop("__ucan_revoked_proof_cids", []),
                field_name="__ucan_revoked_proof_cids",
            )
            ucan_context_cids = coerce_dispatch_list(
                payload.pop("__ucan_context_cids", []),
                field_name="__ucan_context_cids",
            )
            enforce_policy = coerce_dispatch_bool(
                payload.pop("__enforce_policy", config.enable_policy_evaluation),
                field_name="__enforce_policy",
            )
            policy_actor = str(payload.pop("__policy_actor", "") or ucan_actor or "*")
            policy_clauses = coerce_dispatch_list(
                payload.pop("__policy_clauses", []),
                field_name="__policy_clauses",
            )
            policy_resource = payload.pop("__policy_resource", None)
            if policy_resource is not None:
                policy_resource = str(policy_resource)
            parent_event_cids = coerce_dispatch_list(
                payload.pop("__parent_event_cids", []),
                field_name="__parent_event_cids",
            )
            risk_actor = str(payload.pop("__risk_actor", "") or policy_actor or ucan_actor or "*")
            enforce_risk = coerce_dispatch_bool(
                payload.pop("__enforce_risk", config.enable_risk_scoring),
                field_name="__enforce_risk",
            )
            raw_risk_policy = coerce_dispatch_dict(
                payload.pop("__risk_policy", {}),
                field_name="__risk_policy",
            )
            frontier_consensus_signal = coerce_dispatch_dict(
                payload.pop("__frontier_consensus_signal", {}),
                field_name="__frontier_consensus_signal",
            )
            enable_frontier_consensus_signal = coerce_dispatch_bool(
                payload.pop("__enable_frontier_consensus_signal", False),
                field_name="__enable_frontier_consensus_signal",
            )
            execute_frontier = coerce_dispatch_bool(
                payload.pop("__execute_frontier", config.enable_risk_frontier_execution),
                field_name="__execute_frontier",
            )
            use_result_cache = coerce_dispatch_bool(
                payload.pop("__use_result_cache", False),
                field_name="__use_result_cache",
            )
            cache_ttl_raw = payload.pop("__cache_ttl", None)
            cache_ttl: float | None = None
            if cache_ttl_raw is not None:
                cache_ttl = float(cache_ttl_raw)
                if cache_ttl <= 0:
                    raise ValueError("__cache_ttl must be > 0 when provided")
            discover_peers = coerce_dispatch_bool(
                payload.pop("__discover_peers", False),
                field_name="__discover_peers",
            )
            resolve_bootstrap_addrs = coerce_dispatch_bool(
                payload.pop("__resolve_bootstrap_addrs", False),
                field_name="__resolve_bootstrap_addrs",
            )
            peer_probe_limit_raw = payload.pop("__peer_probe_limit", 25)
            peer_probe_limit = int(peer_probe_limit_raw)
            if peer_probe_limit < 1:
                raise ValueError("__peer_probe_limit must be >= 1")
        except ValueError as exc:
            _record_observability("error")
            return {
                "ok": False,
                "error": "invalid_dispatch_parameter",
                "details": str(exc),
                "intent_cid": dispatch_intent_cid,
            }

        policy_decision_binding: dict[str, Any] | None = None
        risk_assessment = None

        services = getattr(server, "_unified_services", {})
        result_cache = None
        cache_meta: dict[str, Any] = {
            "enabled": use_result_cache,
            "hit": False,
            "stored": False,
            "error": "",
            "key": f"{category}.{tool_name}",
        }
        peer_registry_meta: dict[str, Any] | None = None
        peer_bootstrap_meta: dict[str, Any] | None = None

        if use_result_cache and isinstance(services, dict):
            cache_factory = services.get("result_cache_factory")
            if callable(cache_factory):
                try:
                    result_cache = cache_factory()
                except Exception as exc:
                    cache_meta["error"] = str(exc)

        async def _probe_peer_registry() -> dict[str, Any] | None:
            if not discover_peers:
                return None

            response: dict[str, Any] = {
                "enabled": True,
                "factory_used": False,
                "peer_count": 0,
                "peers": [],
                "error": "",
            }

            if not isinstance(services, dict):
                response["error"] = "services_unavailable"
                return response

            try:
                discover_fn = None
                discovery_error = ""

                discovery_factory = services.get("peer_discovery_factory")
                if callable(discovery_factory):
                    response["factory_used"] = True
                    try:
                        discovery = discovery_factory()
                    except Exception as exc:
                        discovery = None
                        discovery_error = str(exc)
                    if discovery is not None:
                        discover_fn = getattr(discovery, "discover_peers", None)
                        if discover_fn is None:
                            discovery_error = "peer_discovery_unavailable"
                    elif not discovery_error:
                        discovery_error = "peer_discovery_unavailable"

                if not callable(discover_fn):
                    registry_factory = services.get("peer_registry_factory")
                    if not callable(registry_factory):
                        response["error"] = discovery_error or "peer_registry_factory_unavailable"
                        return response

                    response["factory_used"] = True
                    registry = registry_factory()
                    if registry is None:
                        response["error"] = discovery_error or "peer_registry_unavailable"
                        return response

                    discover_fn = getattr(registry, "discover_peers", None)
                    if not callable(discover_fn):
                        discover_fn = getattr(registry, "list_connected_peers", None)
                    if not callable(discover_fn):
                        response["error"] = discovery_error or "peer_registry_discovery_unavailable"
                        return response

                    response["error"] = ""

                try:
                    peers_result = await _invoke_maybe_async(discover_fn, max_peers=peer_probe_limit)
                except TypeError:
                    peers_result = await _invoke_maybe_async(discover_fn)
                peers: list[dict[str, Any]] = []
                if isinstance(peers_result, list):
                    peers = [p for p in peers_result if isinstance(p, dict)]
                elif isinstance(peers_result, dict):
                    maybe_peers = peers_result.get("peers")
                    if isinstance(maybe_peers, list):
                        peers = [p for p in maybe_peers if isinstance(p, dict)]
                response["peers"] = peers[:peer_probe_limit]
                response["peer_count"] = len(response["peers"])
            except Exception as exc:
                response["error"] = str(exc)

            return response

        async def _probe_peer_bootstrap() -> dict[str, Any] | None:
            if not resolve_bootstrap_addrs:
                return None

            response: dict[str, Any] = {
                "enabled": True,
                "factory_used": False,
                "address_count": 0,
                "addresses": [],
                "error": "",
            }

            if not isinstance(services, dict):
                response["error"] = "services_unavailable"
                return response

            try:
                addrs_fn = None
                bootstrap_error = ""

                bootstrap_factory = services.get("peer_bootstrap_factory")
                if callable(bootstrap_factory):
                    response["factory_used"] = True
                    try:
                        bootstrap = bootstrap_factory()
                    except Exception as exc:
                        bootstrap = None
                        bootstrap_error = str(exc)

                    if bootstrap is not None:
                        addrs_fn = getattr(bootstrap, "get_bootstrap_addrs", None)
                        if not callable(addrs_fn):
                            addrs_fn = getattr(bootstrap, "get_bootstrap_nodes", None)
                        if addrs_fn is None:
                            bootstrap_error = "peer_bootstrap_resolution_unavailable"
                    elif not bootstrap_error:
                        bootstrap_error = "peer_bootstrap_unavailable"

                if not callable(addrs_fn):
                    registry_factory = services.get("peer_registry_factory")
                    if not callable(registry_factory):
                        response["error"] = bootstrap_error or "peer_bootstrap_factory_unavailable"
                        return response

                    response["factory_used"] = True
                    registry = registry_factory()
                    if registry is None:
                        response["error"] = bootstrap_error or "peer_registry_unavailable"
                        return response

                    addrs_fn = getattr(registry, "get_bootstrap_addrs", None)
                    if not callable(addrs_fn):
                        addrs_fn = getattr(registry, "get_bootstrap_nodes", None)
                    if not callable(addrs_fn):
                        response["error"] = bootstrap_error or "peer_bootstrap_resolution_unavailable"
                        return response

                    response["error"] = ""

                try:
                    addrs_result = await _invoke_maybe_async(addrs_fn, max_peers=peer_probe_limit)
                except TypeError:
                    addrs_result = await _invoke_maybe_async(addrs_fn)

                if isinstance(addrs_result, list):
                    response["addresses"] = [
                        item for item in addrs_result[:peer_probe_limit] if isinstance(item, str) and item
                    ]
                    response["address_count"] = len(response["addresses"])
            except Exception as exc:
                response["error"] = str(exc)

            return response

        if enforce_risk:
            scorer = risk_scorer
            if raw_risk_policy:
                scorer = RiskScorer(
                    RiskScoringPolicy(
                        tool_risk_overrides=dict(raw_risk_policy.get("tool_risk_overrides") or {}),
                        default_risk=float(raw_risk_policy.get("default_risk", 0.3) or 0.3),
                        actor_trust_levels=dict(raw_risk_policy.get("actor_trust_levels") or {}),
                        max_acceptable_risk=float(raw_risk_policy.get("max_acceptable_risk", 0.75) or 0.75),
                    )
                )

            risk_assessment = scorer.score_intent(
                tool=f"{category}.{tool_name}",
                actor=risk_actor,
                params=payload,
            )
            if not risk_assessment.is_acceptable:
                record = risk_scheduler.record_outcome(actor=risk_actor, allowed=False)
                policy_audit.record(
                    decision="risk_denied",
                    tool=f"{category}.{tool_name}",
                    actor=risk_actor,
                    intent_cid=dispatch_intent_cid,
                    policy_cid=policy_cid,
                    justification="risk score exceeds acceptable threshold",
                    extra={"score": risk_assessment.score, "level": risk_assessment.level.value},
                )
                _record_observability("error")
                return {
                    "ok": False,
                    "error": "risk_denied",
                    "risk_assessment": risk_assessment.to_dict(),
                    "risk": record.to_dict(),
                    "audit": policy_audit.stats() if policy_audit.enabled else None,
                }

        ucan_verdict = None
        if enforce_ucan:
            verdict = validate_raw_delegation_chain(
                raw_chain=ucan_proof_chain,
                resource=f"{category}.{tool_name}",
                ability="invoke",
                actor=ucan_actor,
                require_signatures=ucan_require_signatures,
                issuer_public_keys=ucan_issuer_public_keys,
                revoked_proof_cids=ucan_revoked_proof_cids,
                context_cids=ucan_context_cids,
            )
            ucan_verdict = verdict
            if not verdict.allowed:
                record = risk_scheduler.record_outcome(actor=risk_actor, allowed=False)
                policy_audit.record(
                    decision="authorization_denied",
                    tool=f"{category}.{tool_name}",
                    actor=ucan_actor or risk_actor,
                    intent_cid=dispatch_intent_cid,
                    policy_cid=policy_cid,
                    justification=verdict.reason,
                    extra={"scheme": "ucan", "chain_length": verdict.chain_length},
                )
                _record_observability("error")
                return {
                    "ok": False,
                    "error": "authorization_denied",
                    "authorization": {
                        "scheme": "ucan",
                        **verdict.to_dict(),
                    },
                    "risk_assessment": risk_assessment.to_dict() if risk_assessment is not None else None,
                    "risk": record.to_dict(),
                    "audit": policy_audit.stats() if policy_audit.enabled else None,
                }

        def _authorization_success_fields() -> dict[str, Any]:
            if ucan_verdict is None:
                return {}
            return {
                "authorization": {
                    "scheme": "ucan",
                    **ucan_verdict.to_dict(),
                }
            }

        policy_decision = None
        if enforce_policy:
            policy_decision = evaluate_with_ipfs_datasets_policy(
                raw_clauses=policy_clauses,
                actor=policy_actor,
                action=f"{category}.{tool_name}",
                resource=policy_resource,
            )

            try:
                policy_decision_payload = build_decision(
                    decision=policy_decision.decision,
                    intent_cid="",
                    policy_cid=policy_cid,
                    justification=policy_decision.justification,
                    obligations=policy_decision.obligations,
                    policy_version=policy_version,
                )
                policy_decision_cid = compute_artifact_cid(policy_decision_payload)
                artifact_store.put(policy_decision_cid, policy_decision_payload)
                policy_decision_binding = {
                    "decision_cid": policy_decision_cid,
                    "persisted": True,
                    "stats": artifact_store.stats(),
                    **_persist_artifact_store_snapshot(),
                }
            except Exception as exc:
                policy_decision_binding = {
                    "decision_cid": "",
                    "persisted": False,
                    "error": str(exc),
                    "stats": artifact_store.stats(),
                    **dict(artifact_store_runtime_meta),
                }

            if policy_decision.decision == "deny":
                record = risk_scheduler.record_outcome(actor=risk_actor, allowed=False)
                policy_audit.record(
                    decision="policy_denied",
                    tool=f"{category}.{tool_name}",
                    actor=policy_actor or risk_actor,
                    intent_cid=dispatch_intent_cid,
                    policy_cid=policy_cid,
                    justification=policy_decision.justification,
                    obligations=[str(x.get("type") or "") for x in policy_decision.obligations],
                    extra={"scheme": "temporal_policy"},
                )
                _record_observability("error")
                return {
                    "ok": False,
                    "error": "policy_denied",
                    "policy": policy_decision.to_dict(),
                    "policy_decision": dict(policy_decision_binding or {}),
                    "risk_assessment": risk_assessment.to_dict() if risk_assessment is not None else None,
                    "risk": record.to_dict(),
                    "audit": policy_audit.stats() if policy_audit.enabled else None,
                }

        cache_hit = False
        result: Any = None
        try:
            if result_cache is not None:
                get_fn = getattr(result_cache, "get", None)
                if callable(get_fn):
                    cached = await _invoke_maybe_async(
                        get_fn,
                        task_id=f"{category}.{tool_name}",
                        inputs=payload,
                    )
                    if cached is not None:
                        cache_meta["hit"] = True
                        if not emit_artifacts:
                            peer_registry_meta = await _probe_peer_registry()
                            peer_bootstrap_meta = await _probe_peer_bootstrap()
                            obligations = len(policy_decision.obligations) if policy_decision is not None else 0
                            record = risk_scheduler.record_outcome(actor=risk_actor, allowed=True, obligations=obligations)
                            policy_decision_label = "allow"
                            policy_justification = ""
                            policy_obligations: list[str] = []
                            if policy_decision is not None:
                                policy_decision_label = policy_decision.decision
                                policy_justification = policy_decision.justification
                                policy_obligations = [str(x.get("type") or "") for x in policy_decision.obligations]
                            policy_audit.record(
                                decision=policy_decision_label,
                                tool=f"{category}.{tool_name}",
                                actor=policy_actor or ucan_actor or risk_actor,
                                intent_cid=dispatch_intent_cid,
                                policy_cid=policy_cid,
                                justification=policy_justification,
                                obligations=policy_obligations,
                                extra={"cache_hit": True},
                            )
                            _record_observability("success")
                            extra_fields: dict[str, Any] = {}
                            if use_result_cache:
                                extra_fields["cache"] = dict(cache_meta)
                            if peer_registry_meta is not None:
                                extra_fields["peer_registry"] = dict(peer_registry_meta)
                            if peer_bootstrap_meta is not None:
                                extra_fields["peer_bootstrap"] = dict(peer_bootstrap_meta)
                            if policy_decision is None:
                                return _build_success_response(
                                    result=cached,
                                    risk_record=record,
                                    risk_assessment_obj=risk_assessment,
                                    passthrough_result_fields=isinstance(cached, dict),
                                    extra_fields={
                                        **extra_fields,
                                        **_authorization_success_fields(),
                                    },
                                )
                            return _build_success_response(
                                result=cached,
                                risk_record=record,
                                risk_assessment_obj=risk_assessment,
                                policy_obj=policy_decision,
                                policy_decision_obj=policy_decision_binding,
                                extra_fields={
                                    **extra_fields,
                                    **_authorization_success_fields(),
                                },
                            )
                        cache_hit = True
                        result = cached

            if not cache_hit:
                result = await manager.dispatch(category, tool_name, payload)

            if result_cache is not None and not cache_hit:
                put_fn = getattr(result_cache, "put", None)
                if callable(put_fn):
                    try:
                        await _invoke_maybe_async(
                            put_fn,
                            task_id=f"{category}.{tool_name}",
                            value=result,
                            ttl=cache_ttl,
                            inputs=payload,
                        )
                        cache_meta["stored"] = True
                    except Exception as exc:
                        cache_meta["error"] = str(exc)
        except Exception:
            _record_observability("error")
            raise

        peer_registry_meta = await _probe_peer_registry()
        peer_bootstrap_meta = await _probe_peer_bootstrap()

        # Always record inputs and outputs to the event DAG and audit storage,
        # regardless of the emit_artifacts flag.  This ensures every tool
        # invocation is auditable by its content ID.
        _input_payload_for_dag = dict(payload) if isinstance(payload, dict) else {"input": payload}
        _output_payload_for_dag = result if isinstance(result, dict) else {"result": result}
        _always_dag_meta = _record_event_dag_always(
            input_payload={
                "category": category,
                "tool_name": tool_name,
                "intent_cid": dispatch_intent_cid,
                "payload": _input_payload_for_dag,
            },
            output_payload=_output_payload_for_dag,
            parent_cids=parent_event_cids,
        )

        if not emit_artifacts:
            obligations = len(policy_decision.obligations) if policy_decision is not None else 0
            record = risk_scheduler.record_outcome(actor=risk_actor, allowed=True, obligations=obligations)
            policy_decision_label = "allow"
            policy_justification = ""
            policy_obligations: list[str] = []
            if policy_decision is not None:
                policy_decision_label = policy_decision.decision
                policy_justification = policy_decision.justification
                policy_obligations = [str(x.get("type") or "") for x in policy_decision.obligations]
            policy_audit.record(
                decision=policy_decision_label,
                tool=f"{category}.{tool_name}",
                actor=policy_actor or ucan_actor or risk_actor,
                intent_cid=dispatch_intent_cid,
                policy_cid=policy_cid,
                justification=policy_justification,
                obligations=policy_obligations,
                extra={"emit_artifacts": False},
            )
            if policy_decision is None:
                _record_observability("success")
                return _build_success_response(
                    result=result,
                    risk_record=record,
                    risk_assessment_obj=risk_assessment,
                    passthrough_result_fields=isinstance(result, dict),
                    extra_fields={
                        **_authorization_success_fields(),
                        "event_dag": _always_dag_meta,
                        **({"cache": dict(cache_meta)} if use_result_cache else {}),
                        **({"peer_registry": dict(peer_registry_meta)} if peer_registry_meta is not None else {}),
                        **({"peer_bootstrap": dict(peer_bootstrap_meta)} if peer_bootstrap_meta is not None else {}),
                    },
                )
            _record_observability("success")
            return _build_success_response(
                result=result,
                risk_record=record,
                risk_assessment_obj=risk_assessment,
                policy_obj=policy_decision,
                policy_decision_obj=policy_decision_binding,
                extra_fields={
                    **_authorization_success_fields(),
                    "event_dag": _always_dag_meta,
                    **({"cache": dict(cache_meta)} if use_result_cache else {}),
                    **({"peer_registry": dict(peer_registry_meta)} if peer_registry_meta is not None else {}),
                    **({"peer_bootstrap": dict(peer_bootstrap_meta)} if peer_bootstrap_meta is not None else {}),
                },
            )

        # Build immutable artifact references without changing default dispatch
        # shape unless artifact emission is explicitly requested.
        tool_schema = manager.get_tool_schema(category, tool_name)
        interface_cid = compute_artifact_cid(
            {
                "category": category,
                "tool_name": tool_name,
                "input_schema": tool_schema.get("input_schema", {}),
            }
        )
        output_payload = result if isinstance(result, dict) else {"result": result}
        envelope = envelope_from_payloads(
            interface_cid=interface_cid,
            input_payload=payload,
            tool=f"{category}.{tool_name}",
            output_payload=output_payload,
            decision=policy_decision.decision if policy_decision is not None else "allow",
            decision_justification=policy_decision.justification if policy_decision is not None else "",
            decision_obligations=policy_decision.obligations if policy_decision is not None else [],
            proof_cid=proof_cid,
            policy_cid=policy_cid,
            correlation_id=correlation_id,
            parent_event_cids=parent_event_cids,
        )

        persisted_artifacts_meta: dict[str, Any]
        try:
            written = artifact_store.put_many(
                {
                    envelope["input_cid"]: envelope["input"],
                    envelope["intent_cid"]: envelope["intent"],
                    envelope["decision_cid"]: envelope["decision"],
                    envelope["output_cid"]: envelope["output"],
                    envelope["receipt_cid"]: envelope["receipt"],
                    envelope["event_cid"]: envelope["event"],
                }
            )
            persisted_artifacts_meta = {
                "persisted": True,
                "written": int(written),
                "stats": artifact_store.stats(),
                **_persist_artifact_store_snapshot(),
            }
        except Exception as exc:
            persisted_artifacts_meta = {
                "persisted": False,
                "written": 0,
                "error": str(exc),
                "stats": artifact_store.stats(),
                **dict(artifact_store_runtime_meta),
            }

        event_dag_meta: dict[str, Any]
        try:
            event_store.add_event(envelope["event_cid"], envelope["event"])
            obligations = len(policy_decision.obligations) if policy_decision is not None else 0
            record = risk_scheduler.record_outcome(
                actor=risk_actor,
                allowed=True,
                obligations=obligations,
                event_cid=envelope["event_cid"],
            )
            frontier_item = risk_scheduler.enqueue_frontier(
                event_cid=envelope["event_cid"],
                actor=risk_actor,
                expected_value=0.75,
                dependency_ready=True,
                consensus_signal=frontier_consensus_signal,
                enable_consensus_signal=enable_frontier_consensus_signal,
                metadata={
                    "category": category,
                    "tool_name": tool_name,
                    "consensus_signal": dict(frontier_consensus_signal),
                    "enable_consensus_signal": enable_frontier_consensus_signal,
                },
            )
            # Persist all artifact blobs to audit storage using their CIDs.
            _artifact_storage_results: dict[str, Any] = {}
            for _label, _cid_key in (
                ("input", envelope["input_cid"]),
                ("output", envelope["output_cid"]),
                ("event", envelope["event_cid"]),
                ("receipt", envelope["receipt_cid"]),
            ):
                _artifact_storage_results[_label] = _persist_to_audit_storage(
                    _cid_key, envelope[_label if _label != "receipt" else "receipt"]
                )
            event_dag_meta = {
                "persisted": True,
                "lineage": event_store.get_lineage(envelope["event_cid"]),
                "stats": event_store.stats(),
                "storage": _artifact_storage_results,
            }
        except Exception as exc:
            event_dag_meta = {
                "persisted": False,
                "error": str(exc),
                "lineage": [],
                "stats": event_store.stats(),
            }
            record = risk_scheduler.record_outcome(actor=risk_actor, allowed=True)
            frontier_item = None

        frontier_execution = {
            "attempted": False,
            "scheduled": False,
            "route": "",
            "workflow_id": "",
            "task_id": "",
            "event_cid": str(envelope["event_cid"]),
            "error": "",
        }
        if execute_frontier:
            popped = risk_scheduler.pop_next()
            frontier_execution = await _bind_frontier_execution(popped)

        policy_decision_label = "allow"
        policy_justification = ""
        policy_obligations: list[str] = []
        if policy_decision is not None:
            policy_decision_label = policy_decision.decision
            policy_justification = policy_decision.justification
            policy_obligations = [str(x.get("type") or "") for x in policy_decision.obligations]
        policy_audit.record(
            decision=policy_decision_label,
            tool=f"{category}.{tool_name}",
            actor=policy_actor or ucan_actor or risk_actor,
            intent_cid=envelope["intent_cid"],
            policy_cid=policy_cid,
            justification=policy_justification,
            obligations=policy_obligations,
            extra={"emit_artifacts": True, "event_cid": envelope["event_cid"]},
        )
        _record_observability("success")

        policy_decision_response = {
            "decision_cid": envelope["decision_cid"],
            "persisted": bool(persisted_artifacts_meta.get("persisted")),
            "stats": dict((persisted_artifacts_meta.get("stats") or {})),
        }
        if (
            policy_decision is not None
            and policy_decision_binding is not None
            and bool(policy_decision_binding.get("persisted"))
            and str(policy_decision_binding.get("decision_cid") or "")
        ):
            policy_decision_response = dict(policy_decision_binding)

        return _build_success_response(
            result=result,
            risk_record=record,
            risk_assessment_obj=risk_assessment,
            policy_obj=policy_decision,
            policy_decision_obj=policy_decision_response,
            extra_fields={
                **_authorization_success_fields(),
                "artifacts": {
                    "input_cid": envelope["input_cid"],
                    "intent_cid": envelope["intent_cid"],
                    "decision_cid": envelope["decision_cid"],
                    "output_cid": envelope["output_cid"],
                    "receipt_cid": envelope["receipt_cid"],
                    "event_cid": envelope["event_cid"],
                },
                "artifact_payloads": {
                    "intent": envelope["intent"],
                    "decision": envelope["decision"],
                    "receipt": envelope["receipt"],
                    "event": envelope["event"],
                },
                "artifact_store": persisted_artifacts_meta,
                "event_dag": event_dag_meta,
                "frontier": {
                    "enqueued": frontier_item is not None,
                    "priority": round(frontier_item.priority, 5) if frontier_item is not None else None,
                    "event_cid": frontier_item.event_cid if frontier_item is not None else None,
                    "execution": frontier_execution,
                    "stats": risk_scheduler.stats(),
                },
                **({"cache": dict(cache_meta)} if use_result_cache else {}),
                **({"peer_registry": dict(peer_registry_meta)} if peer_registry_meta is not None else {}),
                **({"peer_bootstrap": dict(peer_bootstrap_meta)} if peer_bootstrap_meta is not None else {}),
            },
        )

        # unreachable placeholder to keep static analyzers calm about branch
        # structure; response is returned above.

    async def tools_runtime_metrics() -> dict[str, Any]:
        return {
            "runtimes": runtime_router.get_metrics(),
            "observability": {
                "monitoring": {
                    "info": metrics_collector.get_info(),
                    "snapshot": metrics_collector.get_snapshot(),
                },
                "tracing": dict(tracing_status),
                "prometheus": {
                    **dict(prometheus_status),
                    "info": prometheus_exporter.get_info() if prometheus_exporter is not None else dict(prometheus_status.get("info") or {}),
                },
                "audit_metrics": dict(audit_metrics_status),
            },
        }

    # Attach migration components for callers that want the unified surface.
    unified_services = _build_unified_services()
    risk_scheduler_factory = unified_services.get("risk_scheduler_factory") if isinstance(unified_services, dict) else None
    if callable(risk_scheduler_factory):
        try:
            risk_scheduler = risk_scheduler_factory()
        except Exception:
            risk_scheduler = None
    if risk_scheduler is None:
        risk_scheduler = RiskScheduler()

    workflow_engine_factory = unified_services.get("workflow_engine_factory") if isinstance(unified_services, dict) else None
    workflow_engine = None
    if callable(workflow_engine_factory):
        try:
            workflow_engine = workflow_engine_factory()
        except Exception:
            workflow_engine = None
    if workflow_engine is None:
        workflow_engine = WorkflowEngine()

    workflow_dag_executor_factory = unified_services.get("workflow_dag_executor_factory") if isinstance(unified_services, dict) else None
    workflow_dag_executor = None
    if callable(workflow_dag_executor_factory):
        try:
            workflow_dag_executor = workflow_dag_executor_factory()
        except Exception:
            workflow_dag_executor = None
    if workflow_dag_executor is None:
        workflow_dag_executor = WorkflowDAGExecutor()

    unified_context = UnifiedServerContext(
        runtime_router=runtime_router,
        tool_manager=manager,
        services=unified_services,
        preloaded_categories=list(preloaded_categories),
        supported_profiles=get_unified_supported_profiles(),
        bootstrap_enabled=True,
    )

    setattr(server, "_unified_runtime_router", runtime_router)
    setattr(server, "_unified_tool_manager", manager)
    setattr(server, "_unified_bootstrap_enabled", True)
    setattr(server, "_unified_meta_tools", get_unified_meta_tool_names())
    setattr(server, "_unified_preloaded_categories", preloaded_categories)
    setattr(server, "_unified_services", unified_services)
    setattr(server, "_unified_server_context", unified_context)
    setattr(server, "_unified_server_context_snapshot", unified_context.snapshot())
    setattr(server, "_unified_artifact_store", artifact_store)
    setattr(server, "_unified_artifact_store_meta", dict(artifact_store_runtime_meta))
    setattr(server, "_unified_event_dag", event_store)
    setattr(server, "_unified_workflow_engine", workflow_engine)
    setattr(server, "_unified_workflow_dag_executor", workflow_dag_executor)
    setattr(server, "_unified_risk_scheduler", risk_scheduler)
    setattr(server, "_unified_risk_scorer", risk_scorer)
    setattr(server, "_unified_policy_audit", policy_audit)
    setattr(server, "_unified_metrics_collector", metrics_collector)
    setattr(server, "_unified_p2p_metrics_collector", p2p_metrics_collector)
    setattr(server, "_unified_tracer", tracer)
    setattr(server, "_unified_tracing_status", tracing_status)
    setattr(server, "_unified_prometheus_exporter", prometheus_exporter)
    setattr(server, "_unified_prometheus_status", prometheus_status)
    setattr(server, "_unified_audit_metrics_bridge", audit_metrics_bridge)
    setattr(server, "_unified_audit_metrics_status", audit_metrics_status)
    setattr(server, "_unified_secrets_vault", secrets_vault)
    setattr(server, "_unified_secrets_status", secrets_status)
    setattr(server, "_unified_audit_storage", _audit_storage)
    setattr(server, "_unified_supported_profiles", list(unified_context.supported_profiles))
    from .mcplusplus.profile_g_transport import get_profile_g_dispatcher, profile_metadata
    setattr(server, "_unified_profile_g_dispatcher", get_profile_g_dispatcher())
    setattr(server, "_unified_profile_g_metadata", profile_metadata())
    setattr(
        server,
        "_unified_profile_negotiation",
        {
            "supports_profile_negotiation": True,
            "mode": "optional_additive",
            "profiles": list(unified_context.supported_profiles),
        },
    )

    # Register compact hierarchical meta-tools if legacy server supports it.
    if hasattr(server, "register_tool") and callable(getattr(server, "register_tool")):
        meta_tool_specs = {
            "tools_list_categories": {
                "function": tools_list_categories,
                "description": "List available unified MCP tool categories.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_list_tools": {
                "function": tools_list_tools,
                "description": "List tools in a unified MCP category.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                    },
                    "required": ["category"],
                },
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_get_schema": {
                "function": tools_get_schema,
                "description": "Get schema for a unified MCP tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "tool_name": {"type": "string"},
                    },
                    "required": ["category", "tool_name"],
                },
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_dispatch": {
                "function": tools_dispatch,
                "description": "Dispatch a unified MCP tool call by category and name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "tool_name": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["category", "tool_name", "parameters"],
                },
                "tags": ["unified", "dispatch", "meta"],
            },
            "tools_runtime_metrics": {
                "function": tools_runtime_metrics,
                "description": "Get unified runtime router metrics (latency/error/timeout).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "tags": ["unified", "metrics", "meta"],
            },
        }

        for tool_name in get_unified_meta_tool_names():
            spec = meta_tool_specs[tool_name]
            server.register_tool(
                name=tool_name,
                function=spec["function"],
                description=spec["description"],
                input_schema=spec["input_schema"],
                execution_context="server",
                tags=spec["tags"],
            )


class StandaloneMCP:
    """Canonical standalone MCP implementation for unified mcp_server.

    Provides a lightweight tool/resource/prompt registry compatible with the
    legacy ``ipfs_accelerate_py.mcp.server.StandaloneMCP`` interface so that
    callers migrating away from the deprecated ``mcp`` package can swap the
    import path without any further code changes.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}
        self._error_handler: Any = None
        self.lifespan_start_handler: Any = None
        self.lifespan_stop_handler: Any = None
        self._init_error_handler()
        logger.info("Using standalone MCP implementation: %s", name)

    def _init_error_handler(self) -> None:
        """Initialize optional auto-healing error handler from environment."""
        import os
        try:
            from ipfs_accelerate_py.error_handler import CLIErrorHandler  # type: ignore

            enable_auto_issue = os.environ.get("IPFS_AUTO_ISSUE", "").lower() in ("1", "true", "yes")
            enable_auto_pr = os.environ.get("IPFS_AUTO_PR", "").lower() in ("1", "true", "yes")
            enable_auto_heal = os.environ.get("IPFS_AUTO_HEAL", "").lower() in ("1", "true", "yes")
            repo = os.environ.get("IPFS_REPO", "endomorphosis/ipfs_accelerate_py")

            if enable_auto_issue or enable_auto_pr or enable_auto_heal:
                self._error_handler = CLIErrorHandler(
                    repo=repo,
                    enable_auto_issue=enable_auto_issue,
                    enable_auto_pr=enable_auto_pr,
                    enable_auto_heal=enable_auto_heal,
                    log_context_lines=50,
                )
                logger.info(
                    "MCP auto-healing enabled: issue=%s, pr=%s, heal=%s",
                    enable_auto_issue,
                    enable_auto_pr,
                    enable_auto_heal,
                )
        except ImportError:
            pass
        except Exception:
            pass

    def register_tool(
        self,
        name: str,
        function: Any,
        description: str = "",
        input_schema: Any = None,
        execution_context: str = "server",
        **_kwargs: Any,
    ) -> None:
        """Register a callable tool in the local registry."""
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema or {"type": "object", "properties": {}, "required": []},
            "execution_context": execution_context,
        }

    def register_resource(
        self,
        uri: str,
        function: Any,
        description: str = "",
        **_kwargs: Any,
    ) -> None:
        """Register a resource endpoint in the local registry."""
        self.resources[uri] = {"function": function, "description": description}

    def register_prompt(
        self,
        name: str,
        template: str = "",
        description: str = "",
        input_schema: Any = None,
        **_kwargs: Any,
    ) -> None:
        """Register a prompt template in the local registry."""
        self.prompts[name] = {
            "template": template,
            "description": description,
            "input_schema": input_schema or {"type": "object", "properties": {}, "required": []},
        }

    def create_fastapi_app(self, **_kwargs: Any) -> Any:
        """Create a minimal FastAPI-compatible app backed by this registry."""
        title = "IPFS Accelerate MCP API"
        try:
            from fastapi import FastAPI  # type: ignore
            app: Any = FastAPI(
                title=title,
                description="IPFS Accelerate MCP",
                version="0.1.0",
            )

            @app.get("/healthz")
            async def _healthz():
                return {"status": "ok", "service": self.name}

        except ImportError:
            class _MinimalApp:
                def __init__(self, title: str) -> None:
                    self.title = title
            app = _MinimalApp(title=title)

        app._standalone_mcp = self  # type: ignore[attr-defined]
        return app

    def _report_tool_error(self, tool_name: str, exc: Exception, params: dict) -> None:
        if not self._error_handler:
            return
        try:
            context = {
                "mcp_server": self.name,
                "tool_name": tool_name,
                "tool_params": str(params),
                "error_source": "mcp_tool",
            }
            self._error_handler.capture_error(exc, context=context)
        except Exception:
            pass

    def _report_resource_error(self, resource_uri: str, exc: Exception) -> None:
        if not self._error_handler:
            return
        try:
            context = {"mcp_server": self.name, "resource_uri": resource_uri, "error_source": "mcp_resource"}
            self._error_handler.capture_error(exc, context=context)
        except Exception:
            pass

    def _report_client_error(self, error_data: dict) -> None:
        if not self._error_handler:
            return
        try:
            context = {"mcp_server": self.name, "error_source": "mcp_client", **dict(error_data or {})}
            exc = RuntimeError(str(error_data.get("error_message", "client error")))
            self._error_handler.capture_error(exc, context=context)
        except Exception:
            pass


class MCPServerWrapper:
    """Canonical MCP server wrapper for unified mcp_server.

    Replaces the legacy ``ipfs_accelerate_py.mcp.server.MCPServerWrapper``
    with an implementation that does not import from the deprecated ``mcp``
    package.  The public interface (``name``, ``host``, ``port``, ``app``,
    ``run()``) is preserved for compatibility.
    """

    def __init__(
        self,
        name: str = "ipfs-accelerate",
        description: str = "",
        accelerate_instance: Any = None,
        host: str = "0.0.0.0",
        port: int = 9000,
        mount_path: str = "/mcp",
        debug: bool = False,
        **_extra: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.host = host
        self.port = port
        self.mount_path = mount_path
        self.debug = debug

        from types import SimpleNamespace
        self.state = SimpleNamespace(accelerate=accelerate_instance)

        self.mcp = StandaloneMCP(name=self.name)
        self.mcp.state = SimpleNamespace(accelerate=accelerate_instance)  # type: ignore[attr-defined]

        # Default lifespan handlers for compatibility with callers that check them.
        self.lifespan_start_handler: Any = lambda ctx: None
        self.lifespan_stop_handler: Any = lambda ctx, state: None

        # Build a minimal FastAPI app directly without routing through create_server()
        # to avoid recursive construction.
        self.app = self._build_app()

        # Register canonical hardware info tools as compatibility aliases.
        try:
            from .tools.hardware_tools.native_hardware_tools import hardware_get_info, hardware_recommend

            if "detect_hardware" not in self.mcp.tools:
                self.mcp.register_tool(
                    name="detect_hardware",
                    function=hardware_get_info,
                    description="Detect available hardware",
                    input_schema={"type": "object", "properties": {}, "required": []},
                    execution_context="server",
                )

            if "get_optimal_hardware" not in self.mcp.tools:
                self.mcp.register_tool(
                    name="get_optimal_hardware",
                    function=hardware_recommend,
                    description="Get optimal hardware for a model",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string", "description": "Model name"},
                            "task": {
                                "type": "string",
                                "description": "Task type",
                                "enum": ["inference", "training", "fine-tuning"],
                                "default": "inference",
                            },
                        },
                        "required": ["model_name"],
                    },
                    execution_context="server",
                )
        except Exception:
            pass

        # Register basic system resources.
        try:
            import platform
            if "system://info" not in self.mcp.resources:
                self.mcp.register_resource(
                    uri="system://info",
                    function=lambda: {
                        "platform": platform.platform(),
                        "python_version": platform.python_version(),
                    },
                    description="Basic system information",
                )
            if "system://capabilities" not in self.mcp.resources:
                self.mcp.register_resource(
                    uri="system://capabilities",
                    function=lambda: {"accelerators": {}, "features": {}},
                    description="System capabilities",
                )
        except Exception:
            pass

    def _build_app(self) -> Any:
        """Build a minimal app without routing through create_server() (avoids recursion)."""
        title = "IPFS Accelerate MCP API"
        description = self.description or "IPFS Accelerate MCP"
        try:
            from fastapi import FastAPI  # type: ignore
            app: Any = FastAPI(
                title=title,
                description=description,
                version="0.1.0",
                docs_url="/docs",
                redoc_url="/redoc",
            )

            @app.get("/healthz")
            async def _healthz():
                return {"status": "ok", "service": self.name}

        except ImportError:
            # Fallback when FastAPI is not installed.
            class _MinimalApp:
                def __init__(self, title: str, description: str) -> None:
                    self.title = title
                    self.description = description

            app = _MinimalApp(title=title, description=description)

        app._standalone_mcp = self.mcp  # type: ignore[attr-defined]
        return app

    @property
    def tools(self) -> dict[str, Any]:
        """Expose the underlying MCP tool registry for compatibility."""

        return self.mcp.tools

    @property
    def resources(self) -> dict[str, Any]:
        """Expose the underlying MCP resource registry for compatibility."""

        return self.mcp.resources

    @property
    def prompts(self) -> dict[str, Any]:
        """Expose the underlying MCP prompt registry for compatibility."""

        return self.mcp.prompts

    def register_tool(
        self,
        name: str,
        function: Any,
        description: str = "",
        input_schema: Any = None,
        execution_context: str = "server",
        **kwargs: Any,
    ) -> None:
        """Register a tool on the underlying standalone MCP registry."""

        self.mcp.register_tool(
            name=name,
            function=function,
            description=description,
            input_schema=input_schema,
            execution_context=execution_context,
            **kwargs,
        )

    def register_resource(self, uri: str, function: Any, description: str = "", **kwargs: Any) -> None:
        """Register a resource on the underlying standalone MCP registry."""

        self.mcp.register_resource(uri=uri, function=function, description=description, **kwargs)

    def register_prompt(
        self,
        name: str,
        template: str = "",
        description: str = "",
        input_schema: Any = None,
        **kwargs: Any,
    ) -> None:
        """Register a prompt on the underlying standalone MCP registry."""

        self.mcp.register_prompt(
            name=name,
            template=template,
            description=description,
            input_schema=input_schema,
            **kwargs,
        )

    def run(
        self,
        host: Any = None,
        port: Any = None,
        reload: bool = False,
    ) -> None:
        """Run the MCP server via uvicorn."""
        try:
            import uvicorn
            uvicorn.run(
                self.app,
                host=host or self.host,
                port=int(port or self.port),
                log_level="debug" if self.debug else "info",
                reload=reload,
            )
        except ImportError:
            raise RuntimeError(
                "uvicorn is required to run the MCP server. Install it with: pip install uvicorn"
            )

    def stop(self) -> None:
        """No-op stop hook for compatibility with callers that call server.stop()."""


# Module-level MCP server instance registry (mirrors legacy mcp.server pattern).
_MCP_SERVER_INSTANCE: Any = None
_MCP_LIKE_INSTANCE: Any = None


def set_mcp_like_instance(mcp_like: Any) -> None:
    """Register an MCP-like registry instance for shared tool access."""
    global _MCP_LIKE_INSTANCE
    _MCP_LIKE_INSTANCE = mcp_like


def get_mcp_server_instance() -> Any:
    """Return the last created MCP server instance or MCP-like registry."""
    return _MCP_SERVER_INSTANCE or _MCP_LIKE_INSTANCE


def register_tools(
    mcp: Any,
    *,
    include_p2p_taskqueue_tools: bool = True,
) -> None:
    """Register canonical tool set with an MCP-like registry instance.

    Convenience wrapper used by callers that expect a ``register_tools(mcp)``
    function analogous to the legacy ``ipfs_accelerate_py.mcp.tools.register_all_tools``.
    """
    _ = include_p2p_taskqueue_tools
    try:
        from .tools.hardware_tools.native_hardware_tools import hardware_get_info, hardware_recommend
        mcp.register_tool(
            name="hardware_get_info",
            function=hardware_get_info,
            description="Get hardware acceleration capabilities.",
            input_schema={"type": "object", "properties": {}, "required": []},
            execution_context="server",
        )
        mcp.register_tool(
            name="hardware_recommend",
            function=hardware_recommend,
            description="Recommend hardware for a model type and task.",
            input_schema={
                "type": "object",
                "properties": {
                    "model_type": {"type": "string", "default": "general"},
                    "model_size": {"type": "string", "default": "medium"},
                    "task": {"type": "string", "default": "inference"},
                },
                "required": [],
            },
            execution_context="server",
        )
    except Exception:
        pass

    try:
        from .tools.inference_tools.native_inference_tools import inference_run_inference
        mcp.register_tool(
            name="inference_run_inference",
            function=inference_run_inference,
            description="Run inference on a model.",
            input_schema={
                "type": "object",
                "properties": {
                    "model_cid": {"type": "string"},
                    "input_data": {"type": "string"},
                    "device": {"type": "string", "default": "auto"},
                },
                "required": ["model_cid", "input_data"],
            },
            execution_context="server",
        )
    except Exception:
        pass

    try:
        from .tools.model_tools.native_model_tools import (
            model_get_served,
            model_list_served,
        )
        mcp.register_tool(
            name="model_list_served",
            function=model_list_served,
            description="List models currently available from configured inference servers.",
            input_schema={
                "type": "object",
                "properties": {
                    "endpoint_url": {"type": "string"},
                    "timeout": {"type": "number", "default": 2.0, "minimum": 0.1},
                },
                "required": [],
            },
            execution_context="server",
        )
        mcp.register_tool(
            name="model_get_served",
            function=model_get_served,
            description="Get live serving information for a model ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "endpoint_url": {"type": "string"},
                    "timeout": {"type": "number", "default": 2.0, "minimum": 0.1},
                },
                "required": ["model_id"],
            },
            execution_context="server",
        )
    except Exception:
        pass

    try:
        from .tools.shared_tools.native_shared_tools import generate_text
        mcp.register_tool(
            name="generate_text",
            function=generate_text,
            description="Generate text through the configured LLM router, including local Leanstral.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "model": {"type": "string", "default": "auto"},
                    "max_tokens": {"type": "integer", "default": 512, "minimum": 1},
                    "temperature": {"type": "number", "default": 0.7, "minimum": 0},
                },
                "required": ["prompt"],
            },
            execution_context="server",
        )
    except Exception:
        pass


def _create_base_server(*args: Any, **kwargs: Any) -> Any:
    """Construct the canonical MCP server wrapper."""
    server_kwargs = dict(kwargs)
    server_kwargs.pop("_skip_unified_bridge", None)

    server = MCPServerWrapper(*args, **server_kwargs)
    global _MCP_SERVER_INSTANCE
    _MCP_SERVER_INSTANCE = server
    try:
        set_mcp_like_instance(getattr(server, "mcp", None) or server)
    except Exception:
        pass

    return server


def create_server(*args: Any, **kwargs: Any) -> Any:
    """Create and return an MCP server instance.

    Builds the canonical MCP server wrapper directly while keeping unified
    bootstrap attachment in this package.
    """

    config = UnifiedMCPServerConfig.from_env(
        allowed_preload_categories=get_unified_wave_a_categories()
    )

    server = _create_base_server(*args, **kwargs)

    if config.enable_unified_bootstrap:
        try:
            _attach_unified_bootstrap(server, config)
            logger.info("Attached unified MCP server bootstrap components")
        except Exception as exc:
            # Keep old behavior even if bootstrap init fails.
            logger.warning("Unified bootstrap initialization failed: %s", exc)

    return server


#: Alias matching the legacy ``ipfs_accelerate_py.mcp.server.create_ipfs_mcp_server`` name.
create_ipfs_mcp_server = create_server

#: Alias matching the legacy ``ipfs_accelerate_py.mcp.server.IPFSAccelerateMCPServer`` name.
IPFSAccelerateMCPServer = MCPServerWrapper

#: Alias matching the legacy ``ipfs_accelerate_py.mcp.tools.register_all_tools`` name.
register_all_tools = register_tools


def populate_unified_registry() -> None:
    """Populate the global tool registry with all canonical tools.

    Drop-in replacement for the legacy
    ``ipfs_accelerate_py.mcp.tool_migration.populate_unified_registry``.
    """
    from ipfs_accelerate_py.mcp_server.tool_registry import get_global_registry  # noqa: PLC0415

    registry = get_global_registry()
    mcp_stub = StandaloneMCP(name="registry-seed")
    register_tools(mcp_stub)
    for name, func in mcp_stub.tools.items():
        try:
            registry.register_function(func, name=name)
        except Exception:
            pass


def main() -> None:
    """Start the MCP server using the canonical standalone CLI behavior."""
    from ipfs_accelerate_py.mcp_server.standalone_server import main as standalone_main

    standalone_main()


#: Alias matching ``run_server`` names used in various scripts.
run_server = main


if __name__ == "__main__":
    main()
