"""Compatibility bootstrap server for MCP unification.

This module provides a stable import location for the new canonical MCP server
package while delegating runtime behavior to the existing
`ipfs_accelerate_py.mcp.server` implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .configs import UnifiedMCPServerConfig, parse_preload_categories
from .hierarchical_tool_manager import HierarchicalToolManager
from .runtime_router import RuntimeRouter
from .wave_a_loaders import configure_wave_a_loaders
from .tools.idl import load_idl_tools
from .mcplusplus.artifacts import ArtifactStore, build_decision, compute_artifact_cid, envelope_from_payloads
from .mcplusplus.delegation import validate_raw_delegation_chain
from .mcplusplus.policy_engine import evaluate_raw_policy
from .mcplusplus.event_dag import EventDAGStore
from .mcplusplus.risk_scheduler import RiskScheduler
from .risk_scorer import RiskScorer, RiskScoringPolicy
from .policy_audit_log import PolicyAuditLog
from .secrets_vault import SecretsVault
from .monitoring import EnhancedMetricsCollector, P2PMetricsCollector
from .otel_tracing import MCPTracer, configure_tracing
from .prometheus_exporter import PrometheusExporter

logger = logging.getLogger(__name__)


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
    artifact_store = ArtifactStore()
    risk_scheduler = RiskScheduler()
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
        payload = dict(parameters) if isinstance(parameters, dict) else {}
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
        dispatch_intent_cid = compute_artifact_cid(
            {
                "category": category,
                "tool_name": tool_name,
                "parameters": payload,
            }
        )

        emit_artifacts = bool(payload.pop("__emit_artifacts", config.enable_cid_artifact_emission))
        proof_cid = str(payload.pop("__proof_cid", "") or "")
        policy_cid = str(payload.pop("__policy_cid", "") or "")
        correlation_id = str(payload.pop("__correlation_id", "") or "")
        enforce_ucan = bool(payload.pop("__enforce_ucan", config.enable_ucan_validation))
        ucan_actor = str(payload.pop("__ucan_actor", "") or "")
        ucan_proof_chain = payload.pop("__ucan_proof_chain", [])
        if not isinstance(ucan_proof_chain, list):
            ucan_proof_chain = []
        ucan_require_signatures = bool(payload.pop("__ucan_require_signatures", False))
        ucan_issuer_public_keys = payload.pop("__ucan_issuer_public_keys", {})
        if not isinstance(ucan_issuer_public_keys, dict):
            ucan_issuer_public_keys = {}
        ucan_revoked_proof_cids = payload.pop("__ucan_revoked_proof_cids", [])
        if not isinstance(ucan_revoked_proof_cids, list):
            ucan_revoked_proof_cids = []
        ucan_context_cids = payload.pop("__ucan_context_cids", [])
        if not isinstance(ucan_context_cids, list):
            ucan_context_cids = []
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
        enforce_risk = bool(payload.pop("__enforce_risk", config.enable_risk_scoring))
        raw_risk_policy = payload.pop("__risk_policy", {})
        if not isinstance(raw_risk_policy, dict):
            raw_risk_policy = {}
        execute_frontier = bool(payload.pop("__execute_frontier", config.enable_risk_frontier_execution))
        policy_decision_binding: dict[str, Any] | None = None
        risk_assessment = None

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

        policy_decision = None
        if enforce_policy:
            policy_decision = evaluate_raw_policy(
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
                )
                policy_decision_cid = compute_artifact_cid(policy_decision_payload)
                artifact_store.put(policy_decision_cid, policy_decision_payload)
                policy_decision_binding = {
                    "decision_cid": policy_decision_cid,
                    "persisted": True,
                    "stats": artifact_store.stats(),
                }
            except Exception as exc:
                policy_decision_binding = {
                    "decision_cid": "",
                    "persisted": False,
                    "error": str(exc),
                    "stats": artifact_store.stats(),
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

        try:
            result = await manager.dispatch(category, tool_name, payload)
        except Exception:
            _record_observability("error")
            raise
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
                if isinstance(result, dict):
                    enriched = dict(result)
                    enriched.setdefault("risk", record.to_dict())
                    enriched.setdefault(
                        "risk_assessment",
                        risk_assessment.to_dict() if risk_assessment is not None else None,
                    )
                    if policy_audit.enabled:
                        enriched.setdefault("audit", policy_audit.stats())
                    _record_observability("success")
                    return enriched
                _record_observability("success")
                return {
                    "ok": True,
                    "result": result,
                    "risk_assessment": risk_assessment.to_dict() if risk_assessment is not None else None,
                    "risk": record.to_dict(),
                    "audit": policy_audit.stats() if policy_audit.enabled else None,
                }
            _record_observability("success")
            return {
                "ok": True,
                "result": result,
                "policy": policy_decision.to_dict(),
                "policy_decision": dict(policy_decision_binding or {}),
                "risk_assessment": risk_assessment.to_dict() if risk_assessment is not None else None,
                "risk": record.to_dict(),
                "audit": policy_audit.stats() if policy_audit.enabled else None,
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
            }
        except Exception as exc:
            persisted_artifacts_meta = {
                "persisted": False,
                "written": 0,
                "error": str(exc),
                "stats": artifact_store.stats(),
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
            "policy_decision": {
                "decision_cid": envelope["decision_cid"],
                "persisted": bool(persisted_artifacts_meta.get("persisted")),
                "stats": dict((persisted_artifacts_meta.get("stats") or {})),
            },
            "artifact_payloads": {
                "intent": envelope["intent"],
                "decision": envelope["decision"],
                "receipt": envelope["receipt"],
                "event": envelope["event"],
            },
            "artifact_store": persisted_artifacts_meta,
            "event_dag": event_dag_meta,
            "risk_assessment": risk_assessment.to_dict() if risk_assessment is not None else None,
            "risk": record.to_dict(),
            "frontier": {
                "enqueued": frontier_item is not None,
                "priority": round(frontier_item.priority, 5) if frontier_item is not None else None,
                "event_cid": frontier_item.event_cid if frontier_item is not None else None,
                "execution": frontier_execution,
                "stats": risk_scheduler.stats(),
            },
            "policy": policy_decision.to_dict() if policy_decision is not None else None,
            "audit": policy_audit.stats() if policy_audit.enabled else None,
        }

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
            },
        }

    # Attach migration components for callers that want the unified surface.
    setattr(server, "_unified_runtime_router", runtime_router)
    setattr(server, "_unified_tool_manager", manager)
    setattr(server, "_unified_bootstrap_enabled", True)
    setattr(server, "_unified_meta_tools", get_unified_meta_tool_names())
    setattr(server, "_unified_preloaded_categories", preloaded_categories)
    setattr(server, "_unified_services", _build_unified_services())
    setattr(server, "_unified_artifact_store", artifact_store)
    setattr(server, "_unified_event_dag", event_store)
    setattr(server, "_unified_risk_scheduler", risk_scheduler)
    setattr(server, "_unified_risk_scorer", risk_scorer)
    setattr(server, "_unified_policy_audit", policy_audit)
    setattr(server, "_unified_metrics_collector", metrics_collector)
    setattr(server, "_unified_p2p_metrics_collector", p2p_metrics_collector)
    setattr(server, "_unified_tracer", tracer)
    setattr(server, "_unified_tracing_status", tracing_status)
    setattr(server, "_unified_prometheus_exporter", prometheus_exporter)
    setattr(server, "_unified_prometheus_status", prometheus_status)
    setattr(server, "_unified_secrets_vault", secrets_vault)
    setattr(server, "_unified_secrets_status", secrets_status)
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
