"""
Unified Inference Backend Manager for IPFS Accelerate

This module provides a comprehensive system for managing, discovering, and routing
inference requests across multiple backends including:
- GPU backends (local CUDA, ROCm, etc.)
- API backends (OpenAI, Anthropic, HuggingFace, etc.)
- CLI backends (Claude CLI, OpenAI CLI, etc.)
- P2P/libp2p distributed backends
- WebSocket-enabled backends
- MCP server integration

Key Features:
- Automatic backend discovery and registration
- Health monitoring and status reporting
- Intelligent request routing and load balancing
- Priority-based scheduling
- Multi-protocol support (HTTP, WebSocket, libp2p)
- Resource-aware model loading
"""

import asyncio
import logging
import time
import threading
import inspect
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Types of inference backends"""
    GPU = "gpu"  # Local GPU inference (CUDA, ROCm, etc.)
    API = "api"  # Remote API endpoints
    CLI = "cli"  # CLI tool integrations
    P2P = "p2p"  # libp2p distributed backends
    WEBSOCKET = "websocket"  # WebSocket-enabled backends
    MCP = "mcp"  # MCP server backends
    HYBRID = "hybrid"  # Supports multiple protocols


class BackendStatus(Enum):
    """Backend health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    OFFLINE = "offline"


@dataclass
class BackendCapabilities:
    """Describes what a backend can do"""
    supported_tasks: Set[str] = field(default_factory=set)  # e.g., "text-generation", "embedding"
    supported_models: Set[str] = field(default_factory=set)
    max_batch_size: int = 1
    supports_streaming: bool = False
    supports_batching: bool = False
    hardware_types: Set[str] = field(default_factory=set)  # e.g., "cuda", "cpu", "mps"
    protocols: Set[str] = field(default_factory=set)  # e.g., "http", "websocket", "libp2p"


@dataclass
class BackendMetrics:
    """Runtime metrics for a backend"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    current_queue_size: int = 0
    active_connections: int = 0
    models_loaded: int = 0
    last_health_check: Optional[float] = None
    uptime_seconds: float = 0.0


@dataclass
class BackendInfo:
    """Complete information about a backend"""
    backend_id: str
    backend_type: BackendType
    name: str
    endpoint: Optional[str] = None
    status: BackendStatus = BackendStatus.UNKNOWN
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    metrics: BackendMetrics = field(default_factory=BackendMetrics)
    instance: Optional[Any] = None  # The actual backend instance
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_selected_at: Optional[float] = None
    last_selected_task: Optional[str] = None
    last_selection_reason: Optional[str] = None
    selection_count: int = 0


class InferenceBackendManager:
    """
    Unified manager for all inference backends
    
    Responsibilities:
    - Backend registration and discovery
    - Health monitoring
    - Request routing
    - Load balancing
    - Status reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        state_path = self.config.get('registry_state_path') or self.config.get('state_path')
        if state_path is None:
            state_path = Path.home() / '.cache' / 'ipfs_accelerate' / 'backend_registry.json'
        self._state_path = Path(state_path).expanduser()
        capability_registry_path = self.config.get('capability_registry_path')
        if capability_registry_path is None:
            capability_registry_path = Path.home() / '.cache' / 'ipfs_accelerate' / 'peer_capability_registry.json'
        self._capability_registry_path = str(Path(capability_registry_path).expanduser())
        self._persist_registry = bool(self.config.get('persist_registry', True))
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._persist_registry = False
        
        # Backend registry
        self.backends: Dict[str, BackendInfo] = {}
        self._lock = threading.RLock()
        
        # Backend type mapping
        self.backends_by_type: Dict[BackendType, List[str]] = defaultdict(list)
        
        # Task mapping (which backends can handle which tasks)
        self.task_routing: Dict[str, List[str]] = defaultdict(list)
        
        # Health check configuration
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.health_check_enabled = self.config.get('enable_health_checks', True)
        self._health_check_task = None
        
        # Load balancing strategy
        self.load_balancing_strategy = self.config.get('load_balancing', 'round_robin')
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        self._result_recorder: Optional[Callable[..., Dict[str, Any]]] = self.config.get('result_recorder')

        self._load_registry_state()
        
        logger.info("InferenceBackendManager initialized")

    def _serialize_backend_info(self, backend_info: BackendInfo) -> Dict[str, Any]:
        return {
            "backend_id": backend_info.backend_id,
            "backend_type": backend_info.backend_type.value,
            "name": backend_info.name,
            "endpoint": backend_info.endpoint,
            "status": backend_info.status.value,
            "capabilities": {
                "supported_tasks": sorted(backend_info.capabilities.supported_tasks),
                "supported_models": sorted(backend_info.capabilities.supported_models),
                "max_batch_size": backend_info.capabilities.max_batch_size,
                "supports_streaming": backend_info.capabilities.supports_streaming,
                "supports_batching": backend_info.capabilities.supports_batching,
                "hardware_types": sorted(backend_info.capabilities.hardware_types),
                "protocols": sorted(backend_info.capabilities.protocols),
            },
            "metrics": {
                "total_requests": backend_info.metrics.total_requests,
                "successful_requests": backend_info.metrics.successful_requests,
                "failed_requests": backend_info.metrics.failed_requests,
                "average_latency_ms": backend_info.metrics.average_latency_ms,
                "current_queue_size": backend_info.metrics.current_queue_size,
                "active_connections": backend_info.metrics.active_connections,
                "models_loaded": backend_info.metrics.models_loaded,
                "last_health_check": backend_info.metrics.last_health_check,
                "uptime_seconds": backend_info.metrics.uptime_seconds,
            },
            "metadata": backend_info.metadata,
            "registered_at": backend_info.registered_at,
            "last_seen": backend_info.last_seen,
            "last_selected_at": backend_info.last_selected_at,
            "last_selected_task": backend_info.last_selected_task,
            "last_selection_reason": backend_info.last_selection_reason,
            "selection_count": backend_info.selection_count,
        }

    def _deserialize_backend_info(self, payload: Dict[str, Any]) -> BackendInfo:
        capabilities_data = payload.get("capabilities", {}) or {}
        metrics_data = payload.get("metrics", {}) or {}
        backend_info = BackendInfo(
            backend_id=str(payload.get("backend_id", "")),
            backend_type=BackendType(payload.get("backend_type", BackendType.API.value)),
            name=str(payload.get("name", payload.get("backend_id", "backend"))),
            endpoint=payload.get("endpoint"),
            status=BackendStatus(payload.get("status", BackendStatus.UNKNOWN.value)),
            capabilities=BackendCapabilities(
                supported_tasks=set(capabilities_data.get("supported_tasks", [])),
                supported_models=set(capabilities_data.get("supported_models", [])),
                max_batch_size=int(capabilities_data.get("max_batch_size", 1)),
                supports_streaming=bool(capabilities_data.get("supports_streaming", False)),
                supports_batching=bool(capabilities_data.get("supports_batching", False)),
                hardware_types=set(capabilities_data.get("hardware_types", [])),
                protocols=set(capabilities_data.get("protocols", [])),
            ),
            metrics=BackendMetrics(
                total_requests=int(metrics_data.get("total_requests", 0)),
                successful_requests=int(metrics_data.get("successful_requests", 0)),
                failed_requests=int(metrics_data.get("failed_requests", 0)),
                average_latency_ms=float(metrics_data.get("average_latency_ms", 0.0)),
                current_queue_size=int(metrics_data.get("current_queue_size", 0)),
                active_connections=int(metrics_data.get("active_connections", 0)),
                models_loaded=int(metrics_data.get("models_loaded", 0)),
                last_health_check=metrics_data.get("last_health_check"),
                uptime_seconds=float(metrics_data.get("uptime_seconds", 0.0)),
            ),
            instance=None,
            metadata=dict(payload.get("metadata", {}) or {}),
            registered_at=float(payload.get("registered_at", time.time())),
            last_seen=float(payload.get("last_seen", time.time())),
            last_selected_at=payload.get("last_selected_at"),
            last_selected_task=payload.get("last_selected_task"),
            last_selection_reason=payload.get("last_selection_reason"),
            selection_count=int(payload.get("selection_count", 0)),
        )
        return backend_info

    def _load_registry_state(self) -> None:
        if not self._persist_registry or not self._state_path.exists():
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.debug(f"Failed to load backend registry state: {exc}")
            return

        self.backends.clear()
        self.backends_by_type.clear()
        self.task_routing.clear()

        for backend_payload in payload.get("backends", []):
            try:
                backend_info = self._deserialize_backend_info(backend_payload)
            except Exception as exc:
                logger.debug(f"Skipping backend registry entry during load: {exc}")
                continue

            self.backends[backend_info.backend_id] = backend_info
            self.backends_by_type[backend_info.backend_type].append(backend_info.backend_id)
            for task in backend_info.capabilities.supported_tasks:
                if backend_info.backend_id not in self.task_routing[task]:
                    self.task_routing[task].append(backend_info.backend_id)

    def _save_registry_state(self) -> None:
        if not self._persist_registry:
            return

        payload = {
            "backends": [self._serialize_backend_info(backend_info) for backend_info in self.backends.values()],
            "load_balancing_strategy": self.load_balancing_strategy,
            "timestamp": time.time(),
        }

        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as exc:
            logger.debug(f"Failed to save backend registry state: {exc}")

    async def execute_task(
        self,
        *,
        task: str,
        model: str,
        inputs: List[Any],
        preferred_types: Optional[List[BackendType]] = None,
        required_protocols: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Select a backend, invoke it, record metrics, and finalize the result."""
        backend = self.select_backend_for_task(
            task=task,
            model=model,
            preferred_types=preferred_types,
            required_protocols=required_protocols,
        )
        if backend is None:
            raise RuntimeError(f"No backend available for task '{task}'")

        instance = backend.instance
        if instance is None:
            raise RuntimeError(f"Backend '{backend.backend_id}' has no executable instance")

        method_name = self._resolve_execution_method_name(task=task, instance=instance)
        if method_name is None:
            raise RuntimeError(f"Backend '{backend.backend_id}' does not support executable method for task '{task}'")

        method = getattr(instance, method_name)
        call_kwargs = self._build_execution_kwargs(
            task=task,
            model=model,
            inputs=inputs,
            parameters=parameters or {},
        )

        started = time.time()
        success = False
        try:
            if inspect.iscoroutinefunction(method):
                raw_result = await method(**call_kwargs)
            else:
                raw_result = method(**call_kwargs)
            latency_ms = (time.time() - started) * 1000.0
            success = True
            self.record_request(backend.backend_id, success=True, latency_ms=latency_ms)

            if isinstance(raw_result, dict):
                result = dict(raw_result)
            else:
                result = {"result": raw_result}

            result.setdefault("processing_time", latency_ms / 1000.0)
            result.setdefault("device", getattr(instance, "device", None))

            return self.finalize_inference_result(
                backend_id=backend.backend_id,
                task=task,
                model=model,
                inputs=inputs,
                result=result,
            )
        except Exception:
            latency_ms = (time.time() - started) * 1000.0
            self.record_request(backend.backend_id, success=False, latency_ms=latency_ms)
            raise

    def _resolve_execution_method_name(self, *, task: str, instance: Any) -> Optional[str]:
        candidates: List[str] = []
        if task == "text-generation":
            candidates = ["run_inference", "generate_text", "generate", "chat", "completion"]
        elif task in {"text-embedding", "embedding"}:
            candidates = ["run_inference", "generate_embedding", "embedding", "embed", "batch_embed"]
        else:
            candidates = ["run_inference", "infer", "predict", "generate"]

        for name in candidates:
            if hasattr(instance, name) and callable(getattr(instance, name)):
                return name
        return None

    def _build_execution_kwargs(
        self,
        *,
        task: str,
        model: str,
        inputs: List[Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(parameters)
        if task == "text-generation":
            payload.setdefault("model_id", model)
            payload.setdefault("model", model)
            payload.setdefault("inputs", inputs[0] if len(inputs) == 1 else inputs)
            payload.setdefault("prompt", inputs[0] if inputs else "")
        elif task in {"text-embedding", "embedding"}:
            payload.setdefault("model_id", model)
            payload.setdefault("model", model)
            payload.setdefault("text", inputs[0] if inputs else "")
            payload.setdefault("texts", inputs)
        else:
            payload.setdefault("model_id", model)
            payload.setdefault("model", model)
            payload.setdefault("data", inputs[0] if len(inputs) == 1 else inputs)
            payload.setdefault("inputs", inputs)
        return payload

    def finalize_inference_result(
        self,
        *,
        backend_id: str,
        task: str,
        model: str,
        inputs: List[Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach canonical backend metadata and run the configured result recorder.

        This creates a single post-execution seam so backend callers do not need
        to hand-roll persistence/provenance behavior.
        """
        backend_info = self.get_backend(backend_id)
        backend_type = backend_info.backend_type.value if backend_info else None
        endpoint = backend_info.endpoint if backend_info else None

        merged = dict(result)
        merged.setdefault("backend_id", backend_id)
        merged.setdefault("backend_type", backend_type)
        merged.setdefault("endpoint", endpoint)
        merged.setdefault("task", task)
        merged.setdefault("model", model)
        if backend_info and backend_info.last_selection_reason:
            merged.setdefault("selection_reason", backend_info.last_selection_reason)

        if callable(self._result_recorder):
            try:
                recorded = self._result_recorder(
                    model=model,
                    inputs=inputs,
                    result=merged,
                    backend_id=backend_id,
                    backend_type=backend_type,
                    endpoint=endpoint,
                    device=merged.get("device"),
                )
                if isinstance(recorded, dict):
                    merged = recorded
            except Exception as exc:
                logger.warning(f"Result recorder failed for backend {backend_id}: {exc}")

        return merged
    
    def register_backend(
        self,
        backend_id: str,
        backend_type: BackendType,
        name: str,
        instance: Any,
        capabilities: Optional[BackendCapabilities] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new inference backend
        
        Args:
            backend_id: Unique identifier for the backend
            backend_type: Type of backend
            name: Human-readable name
            instance: The actual backend instance
            capabilities: What the backend can do
            endpoint: Optional endpoint URL
            metadata: Additional metadata
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if backend_id in self.backends:
                logger.warning(f"Backend {backend_id} already registered, updating")
            
            backend_info = BackendInfo(
                backend_id=backend_id,
                backend_type=backend_type,
                name=name,
                instance=instance,
                endpoint=endpoint,
                capabilities=capabilities or BackendCapabilities(),
                metadata=metadata or {},
                status=BackendStatus.INITIALIZING
            )
            
            self.backends[backend_id] = backend_info
            self.backends_by_type[backend_type].append(backend_id)
            
            # Update task routing
            for task in backend_info.capabilities.supported_tasks:
                if backend_id not in self.task_routing[task]:
                    self.task_routing[task].append(backend_id)
            
            logger.info(f"Registered backend: {backend_id} ({name}) - Type: {backend_type.value}")
            
            # Set to healthy after registration (can be overridden by health check)
            self._update_backend_status(backend_id, BackendStatus.HEALTHY)
            self._save_registry_state()
            
            return True
    
    def unregister_backend(self, backend_id: str) -> bool:
        """Unregister a backend"""
        with self._lock:
            if backend_id not in self.backends:
                logger.warning(f"Backend {backend_id} not found")
                return False
            
            backend_info = self.backends[backend_id]
            
            # Remove from type mapping
            if backend_id in self.backends_by_type[backend_info.backend_type]:
                self.backends_by_type[backend_info.backend_type].remove(backend_id)
            
            # Remove from task routing
            for task in backend_info.capabilities.supported_tasks:
                if backend_id in self.task_routing[task]:
                    self.task_routing[task].remove(backend_id)
            
            # Remove from registry
            del self.backends[backend_id]
            self._save_registry_state()
            
            logger.info(f"Unregistered backend: {backend_id}")
            return True

    def prune_stale_backends(self, max_age_s: float = 300.0, *, statuses: Optional[Set[BackendStatus]] = None) -> List[str]:
        """Remove backends that have not been seen recently.

        Args:
            max_age_s: Maximum allowed age since last_seen before pruning.
            statuses: Optional status filter. When provided, only backends with
                these statuses are eligible for pruning.

        Returns:
            List of backend IDs that were removed.
        """
        removed: List[str] = []
        cutoff = time.time() - float(max_age_s)
        eligible_statuses = set(statuses) if statuses is not None else {
            BackendStatus.OFFLINE,
            BackendStatus.UNHEALTHY,
            BackendStatus.UNKNOWN,
        }

        with self._lock:
            backend_ids = list(self.backends.keys())
            for backend_id in backend_ids:
                backend_info = self.backends.get(backend_id)
                if backend_info is None:
                    continue
                if backend_info.status not in eligible_statuses:
                    continue
                if backend_info.last_seen >= cutoff:
                    continue

                if backend_id in self.backends_by_type.get(backend_info.backend_type, []):
                    self.backends_by_type[backend_info.backend_type].remove(backend_id)

                for task in list(backend_info.capabilities.supported_tasks):
                    if backend_id in self.task_routing.get(task, []):
                        self.task_routing[task].remove(backend_id)
                        if not self.task_routing[task]:
                            del self.task_routing[task]

                del self.backends[backend_id]
                removed.append(backend_id)

            if removed:
                self._save_registry_state()

        if removed:
            logger.info("Pruned stale backends: %s", ", ".join(removed))
        return removed
    
    def get_backend(self, backend_id: str) -> Optional[BackendInfo]:
        """Get information about a specific backend"""
        return self.backends.get(backend_id)
    
    def list_backends(
        self,
        backend_type: Optional[BackendType] = None,
        status: Optional[BackendStatus] = None,
        task: Optional[str] = None
    ) -> List[BackendInfo]:
        """
        List backends with optional filtering
        
        Args:
            backend_type: Filter by backend type
            status: Filter by status
            task: Filter by supported task
            
        Returns:
            List of matching backends
        """
        with self._lock:
            backends = list(self.backends.values())
            
            if backend_type:
                backends = [b for b in backends if b.backend_type == backend_type]
            
            if status:
                backends = [b for b in backends if b.status == status]
            
            if task:
                backend_ids = self.task_routing.get(task, [])
                backends = [b for b in backends if b.backend_id in backend_ids]
            
            return backends
    
    def select_backend_for_task(
        self,
        task: str,
        model: Optional[str] = None,
        preferred_types: Optional[List[BackendType]] = None,
        required_protocols: Optional[List[str]] = None
    ) -> Optional[BackendInfo]:
        """
        Select the best backend for a given task
        
        Args:
            task: The inference task type
            model: Optional specific model required
            preferred_types: Preferred backend types (in order)
            required_protocols: Required protocol support
            
        Returns:
            Selected backend or None if no suitable backend found
        """
        with self._lock:
            # Get backends that support this task
            candidate_ids = self.task_routing.get(task, [])
            if not candidate_ids:
                logger.warning(f"No backends found for task: {task}")
                return None
            
            candidates = [self.backends[bid] for bid in candidate_ids if bid in self.backends]
            selection_reasons: Dict[str, str] = {}

            for candidate in candidates:
                reasons = [f"supports_task:{task}"]
                if model:
                    if candidate.capabilities.supported_models and model in candidate.capabilities.supported_models:
                        reasons.append(f"supports_model:{model}")
                    elif candidate.capabilities.supported_models:
                        reasons.append(f"model_mismatch:{model}")
                if required_protocols:
                    missing_protocols = [proto for proto in required_protocols if proto not in candidate.capabilities.protocols]
                    if missing_protocols:
                        reasons.append(f"missing_protocols:{','.join(missing_protocols)}")
                    else:
                        reasons.append(f"protocols:{','.join(required_protocols)}")
                if preferred_types and candidate.backend_type in preferred_types:
                    reasons.append(f"preferred_type:{candidate.backend_type.value}")
                if self.load_balancing_strategy:
                    reasons.append(f"strategy:{self.load_balancing_strategy}")
                if self._capability_registry_path:
                    reasons.append(f"capability_registry:{self._capability_registry_path}")
                selection_reasons[candidate.backend_id] = ";".join(reasons)
            
            # Filter by status (only healthy backends)
            candidates = [b for b in candidates if b.status == BackendStatus.HEALTHY]
            
            if not candidates:
                logger.warning(f"No healthy backends found for task: {task}")
                return None
            
            # Filter by model if specified
            if model:
                candidates = [
                    b for b in candidates
                    if not b.capabilities.supported_models or model in b.capabilities.supported_models
                ]
            
            # Filter by protocol if specified
            if required_protocols:
                candidates = [
                    b for b in candidates
                    if all(proto in b.capabilities.protocols for proto in required_protocols)
                ]
            
            if not candidates:
                logger.warning(f"No backends match requirements for task: {task}")
                return None
            
            # Sort by preferred types if specified
            if preferred_types:
                type_priority = {t: i for i, t in enumerate(preferred_types)}
                candidates.sort(key=lambda b: type_priority.get(b.backend_type, len(preferred_types)))
            
            # Apply load balancing strategy
            if self.load_balancing_strategy == 'round_robin':
                # Round-robin within task
                idx = self._round_robin_counters[task] % len(candidates)
                self._round_robin_counters[task] += 1
                selected = candidates[idx]
            
            elif self.load_balancing_strategy == 'least_loaded':
                # Select backend with smallest queue
                candidates.sort(key=lambda b: b.metrics.current_queue_size)
                selected = candidates[0]
            
            elif self.load_balancing_strategy == 'best_performance':
                # Select backend with best average latency
                candidates.sort(key=lambda b: b.metrics.average_latency_ms or float('inf'))
                selected = candidates[0]
            
            else:
                # Default: return first candidate
                selected = candidates[0]

            selected.last_selected_at = time.time()
            selected.last_selected_task = task
            selected.last_selection_reason = selection_reasons.get(
                selected.backend_id,
                f"task:{task};strategy:{self.load_balancing_strategy}",
            )
            selected.selection_count += 1
            self._save_registry_state()
            return selected
    
    def get_backend_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report of all backends
        
        Returns:
            Status report dictionary
        """
        with self._lock:
            return {
                "total_backends": len(self.backends),
                "backends_by_type": {
                    bt.value: len(ids) for bt, ids in self.backends_by_type.items()
                },
                "backends_by_status": {
                    status.value: len([
                        b for b in self.backends.values() if b.status == status
                    ])
                    for status in BackendStatus
                },
                "total_requests": sum(b.metrics.total_requests for b in self.backends.values()),
                "total_successful": sum(b.metrics.successful_requests for b in self.backends.values()),
                "total_failed": sum(b.metrics.failed_requests for b in self.backends.values()),
                "supported_tasks": list(self.task_routing.keys()),
                "backends": [
                    {
                        "id": b.backend_id,
                        "name": b.name,
                        "type": b.backend_type.value,
                        "status": b.status.value,
                        "endpoint": b.endpoint,
                        "tasks": list(b.capabilities.supported_tasks),
                        "protocols": list(b.capabilities.protocols),
                        "last_selection_reason": b.last_selection_reason,
                        "metrics": {
                            "requests": b.metrics.total_requests,
                            "success_rate": (
                                b.metrics.successful_requests / b.metrics.total_requests * 100
                                if b.metrics.total_requests > 0 else 0
                            ),
                            "avg_latency_ms": b.metrics.average_latency_ms,
                            "queue_size": b.metrics.current_queue_size,
                            "models_loaded": b.metrics.models_loaded,
                        }
                    }
                    for b in self.backends.values()
                ],
                "timestamp": time.time()
            }
    
    def _update_backend_status(self, backend_id: str, status: BackendStatus):
        """Update backend status"""
        if backend_id in self.backends:
            self.backends[backend_id].status = status
            self.backends[backend_id].last_seen = time.time()
            self._save_registry_state()
    
    def record_request(self, backend_id: str, success: bool, latency_ms: float):
        """Record metrics for a request"""
        with self._lock:
            if backend_id not in self.backends:
                return
            
            metrics = self.backends[backend_id].metrics
            metrics.total_requests += 1
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            # Update average latency (exponential moving average)
            if metrics.average_latency_ms == 0:
                metrics.average_latency_ms = latency_ms
            else:
                alpha = 0.3  # Weight for new samples
                metrics.average_latency_ms = (
                    alpha * latency_ms + (1 - alpha) * metrics.average_latency_ms
                )
            self._save_registry_state()
    
    async def health_check_loop(self):
        """Periodic health check for all backends"""
        while self.health_check_enabled:
            try:
                await self.run_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            await asyncio.sleep(self.health_check_interval)
    
    async def run_health_checks(self):
        """Run health checks on all backends"""
        with self._lock:
            backend_ids = list(self.backends.keys())
        
        for backend_id in backend_ids:
            try:
                await self.check_backend_health(backend_id)
            except Exception as e:
                logger.error(f"Health check failed for {backend_id}: {e}")
                self._update_backend_status(backend_id, BackendStatus.UNHEALTHY)
    
    async def check_backend_health(self, backend_id: str) -> bool:
        """
        Check health of a specific backend
        
        Args:
            backend_id: Backend to check
            
        Returns:
            True if healthy, False otherwise
        """
        backend_info = self.get_backend(backend_id)
        if not backend_info:
            return False
        
        # Update last health check time
        backend_info.metrics.last_health_check = time.time()
        
        # Check if backend has a health check method
        instance = backend_info.instance
        if instance and hasattr(instance, 'health_check'):
            try:
                if asyncio.iscoroutinefunction(instance.health_check):
                    result = await instance.health_check()
                else:
                    result = instance.health_check()
                
                if result:
                    self._update_backend_status(backend_id, BackendStatus.HEALTHY)
                    return True
                else:
                    self._update_backend_status(backend_id, BackendStatus.UNHEALTHY)
                    return False
            except Exception as e:
                logger.error(f"Health check error for {backend_id}: {e}")
                self._update_backend_status(backend_id, BackendStatus.UNHEALTHY)
                return False
        
        # If no health check method, assume healthy if recently seen
        time_since_seen = time.time() - backend_info.last_seen
        if time_since_seen > 300:  # 5 minutes
            self._update_backend_status(backend_id, BackendStatus.OFFLINE)
            return False
        
        return True
    
    def start_health_monitoring(self):
        """Start the background health check loop"""
        if self._health_check_task is None and self.health_check_enabled:
            logger.info("Starting health monitoring")
            self._health_check_task = asyncio.create_task(self.health_check_loop())
    
    def stop_health_monitoring(self):
        """Stop the background health check loop"""
        if self._health_check_task:
            logger.info("Stopping health monitoring")
            self._health_check_task.cancel()
            self._health_check_task = None


# Global singleton instance
_global_manager: Optional[InferenceBackendManager] = None


def get_backend_manager(config: Optional[Dict[str, Any]] = None) -> InferenceBackendManager:
    """Get the global backend manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = InferenceBackendManager(config)
    return _global_manager


def register_backend_from_config(backend_config: Dict[str, Any]) -> bool:
    """
    Register a backend from a configuration dictionary
    
    Args:
        backend_config: Backend configuration containing:
            - backend_id: Unique identifier
            - backend_type: Type (gpu, api, cli, etc.)
            - name: Display name
            - instance: Backend instance
            - capabilities: Optional capabilities dict
            - endpoint: Optional endpoint URL
            - metadata: Optional metadata
            
    Returns:
        True if registration successful
    """
    manager = get_backend_manager()
    
    backend_type_str = backend_config.get('backend_type', 'api')
    try:
        backend_type = BackendType(backend_type_str)
    except ValueError:
        logger.error(f"Invalid backend type: {backend_type_str}")
        return False
    
    capabilities = None
    if 'capabilities' in backend_config:
        cap_dict = backend_config['capabilities']
        capabilities = BackendCapabilities(
            supported_tasks=set(cap_dict.get('supported_tasks', [])),
            supported_models=set(cap_dict.get('supported_models', [])),
            max_batch_size=cap_dict.get('max_batch_size', 1),
            supports_streaming=cap_dict.get('supports_streaming', False),
            supports_batching=cap_dict.get('supports_batching', False),
            hardware_types=set(cap_dict.get('hardware_types', [])),
            protocols=set(cap_dict.get('protocols', ['http']))
        )
    
    return manager.register_backend(
        backend_id=backend_config['backend_id'],
        backend_type=backend_type,
        name=backend_config['name'],
        instance=backend_config.get('instance'),
        capabilities=capabilities,
        endpoint=backend_config.get('endpoint'),
        metadata=backend_config.get('metadata', {})
    )
