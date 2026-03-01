"""Compatibility bootstrap server for MCP unification.

This module provides a stable import location for the new canonical MCP server
package while delegating runtime behavior to the existing
`ipfs_accelerate_py.mcp.server` implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from .configs import UnifiedMCPServerConfig, parse_preload_categories
from .hierarchical_tool_manager import HierarchicalToolManager
from .runtime_router import RuntimeRouter
from .wave_a_loaders import configure_wave_a_loaders
from .tools.idl import load_idl_tools
from .mcplusplus.artifacts import compute_artifact_cid, envelope_from_payloads
from .mcplusplus.delegation import validate_raw_delegation_chain
from .mcplusplus.policy_engine import evaluate_raw_policy
from .mcplusplus.event_dag import EventDAGStore
from .mcplusplus.risk_scheduler import RiskScheduler

logger = logging.getLogger(__name__)


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
        "mcp++/profile-a-idl",
        "mcp++/profile-b-cid-artifacts",
        "mcp++/profile-c-ucan",
        "mcp++/profile-d-temporal-policy",
        "mcp++/profile-e-mcp-p2p",
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
        "peer_discovery_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["PeerDiscoveryManager"]
        ).PeerDiscoveryManager(**kwargs),
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
    risk_scheduler = RiskScheduler()
    configure_wave_a_loaders(manager)
    manager.register_category_loader(
        "idl",
        lambda mgr: load_idl_tools(mgr, supported_capabilities=get_unified_supported_profiles()),
    )
    preloaded_categories = _preload_configured_categories(manager, config.preload_categories)

    async def tools_list_categories() -> dict[str, Any]:
        return {"categories": manager.list_categories()}

    async def tools_list_tools(category: str) -> dict[str, Any]:
        return {"category": category, "tools": manager.list_tools(category)}

    async def tools_get_schema(category: str, tool_name: str) -> dict[str, Any]:
        return manager.get_tool_schema(category, tool_name)

    async def tools_dispatch(category: str, tool_name: str, parameters: dict[str, Any]) -> Any:
        payload = dict(parameters) if isinstance(parameters, dict) else {}

        emit_artifacts = bool(payload.pop("__emit_artifacts", config.enable_cid_artifact_emission))
        proof_cid = str(payload.pop("__proof_cid", "") or "")
        policy_cid = str(payload.pop("__policy_cid", "") or "")
        correlation_id = str(payload.pop("__correlation_id", "") or "")
        enforce_ucan = bool(payload.pop("__enforce_ucan", config.enable_ucan_validation))
        ucan_actor = str(payload.pop("__ucan_actor", "") or "")
        ucan_proof_chain = payload.pop("__ucan_proof_chain", [])
        if not isinstance(ucan_proof_chain, list):
            ucan_proof_chain = []
        enforce_policy = bool(payload.pop("__enforce_policy", config.enable_policy_evaluation))
        policy_actor = str(payload.pop("__policy_actor", "") or ucan_actor or "*")
        policy_clauses = payload.pop("__policy_clauses", [])
        if not isinstance(policy_clauses, list):
            policy_clauses = []
        policy_resource = payload.pop("__policy_resource", None)
        if policy_resource is not None:
            policy_resource = str(policy_resource)
        parent_event_cids = payload.pop("__parent_event_cids", [])
        if not isinstance(parent_event_cids, list):
            parent_event_cids = []
        risk_actor = str(payload.pop("__risk_actor", "") or policy_actor or ucan_actor or "*")

        if enforce_ucan:
            verdict = validate_raw_delegation_chain(
                raw_chain=ucan_proof_chain,
                resource=f"{category}.{tool_name}",
                ability="invoke",
                actor=ucan_actor,
            )
            if not verdict.allowed:
                record = risk_scheduler.record_outcome(actor=risk_actor, allowed=False)
                return {
                    "ok": False,
                    "error": "authorization_denied",
                    "authorization": {
                        "scheme": "ucan",
                        **verdict.to_dict(),
                    },
                    "risk": record.to_dict(),
                }

        policy_decision = None
        if enforce_policy:
            policy_decision = evaluate_raw_policy(
                raw_clauses=policy_clauses,
                actor=policy_actor,
                action=f"{category}.{tool_name}",
                resource=policy_resource,
            )
            if policy_decision.decision == "deny":
                record = risk_scheduler.record_outcome(actor=risk_actor, allowed=False)
                return {
                    "ok": False,
                    "error": "policy_denied",
                    "policy": policy_decision.to_dict(),
                    "risk": record.to_dict(),
                }

        result = await manager.dispatch(category, tool_name, payload)
        if not emit_artifacts:
            obligations = len(policy_decision.obligations) if policy_decision is not None else 0
            record = risk_scheduler.record_outcome(actor=risk_actor, allowed=True, obligations=obligations)
            if policy_decision is None:
                if isinstance(result, dict):
                    enriched = dict(result)
                    enriched.setdefault("risk", record.to_dict())
                    return enriched
                return {
                    "ok": True,
                    "result": result,
                    "risk": record.to_dict(),
                }
            return {
                "ok": True,
                "result": result,
                "policy": policy_decision.to_dict(),
                "risk": record.to_dict(),
            }

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
            decision="allow",
            proof_cid=proof_cid,
            policy_cid=policy_cid,
            correlation_id=correlation_id,
            parent_event_cids=parent_event_cids,
        )

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
                metadata={"category": category, "tool_name": tool_name},
            )
            event_dag_meta = {
                "persisted": True,
                "lineage": event_store.get_lineage(envelope["event_cid"]),
                "stats": event_store.stats(),
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

        return {
            "ok": True,
            "result": result,
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
            "event_dag": event_dag_meta,
            "risk": record.to_dict(),
            "frontier": {
                "enqueued": frontier_item is not None,
                "priority": round(frontier_item.priority, 5) if frontier_item is not None else None,
                "event_cid": frontier_item.event_cid if frontier_item is not None else None,
                "stats": risk_scheduler.stats(),
            },
            "policy": policy_decision.to_dict() if policy_decision is not None else None,
        }

        # unreachable placeholder to keep static analyzers calm about branch
        # structure; response is returned above.

    async def tools_runtime_metrics() -> dict[str, Any]:
        return {"runtimes": runtime_router.get_metrics()}

    # Attach migration components for callers that want the unified surface.
    setattr(server, "_unified_runtime_router", runtime_router)
    setattr(server, "_unified_tool_manager", manager)
    setattr(server, "_unified_bootstrap_enabled", True)
    setattr(server, "_unified_meta_tools", get_unified_meta_tool_names())
    setattr(server, "_unified_preloaded_categories", preloaded_categories)
    setattr(server, "_unified_services", _build_unified_services())
    setattr(server, "_unified_event_dag", event_store)
    setattr(server, "_unified_risk_scheduler", risk_scheduler)
    setattr(server, "_unified_supported_profiles", get_unified_supported_profiles())
    setattr(
        server,
        "_unified_profile_negotiation",
        {
            "supports_profile_negotiation": True,
            "mode": "optional_additive",
            "profiles": get_unified_supported_profiles(),
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


def create_server(*args: Any, **kwargs: Any) -> Any:
    """Create and return an MCP server instance.

    Delegates to the current stable implementation under
    `ipfs_accelerate_py.mcp.server`. This allows callers to migrate imports to
    `ipfs_accelerate_py.mcp_server.server` immediately while preserving behavior.
    """
    from ipfs_accelerate_py.mcp.server import create_mcp_server

    # Prevent recursive bridging if legacy create_mcp_server is configured to
    # route back through this unified package.
    kwargs.setdefault("_skip_unified_bridge", True)

    config = UnifiedMCPServerConfig.from_env(
        allowed_preload_categories=get_unified_wave_a_categories()
    )

    server = create_mcp_server(*args, **kwargs)

    if config.enable_unified_bootstrap:
        try:
            _attach_unified_bootstrap(server, config)
            logger.info("Attached unified MCP server bootstrap components")
        except Exception as exc:
            # Keep old behavior even if bootstrap init fails.
            logger.warning("Unified bootstrap initialization failed: %s", exc)

    return server


def main() -> None:
    """Start the MCP server using existing CLI behavior.

    Delegates to `ipfs_accelerate_py.mcp.server.main` until the unified server
    implementation is ported in place.
    """
    from ipfs_accelerate_py.mcp.server import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    main()
