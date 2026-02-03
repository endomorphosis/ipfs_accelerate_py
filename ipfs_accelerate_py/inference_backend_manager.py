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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
import json

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
        
        logger.info("InferenceBackendManager initialized")
    
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
            
            logger.info(f"Unregistered backend: {backend_id}")
            return True
    
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
                return candidates[idx]
            
            elif self.load_balancing_strategy == 'least_loaded':
                # Select backend with smallest queue
                candidates.sort(key=lambda b: b.metrics.current_queue_size)
                return candidates[0]
            
            elif self.load_balancing_strategy == 'best_performance':
                # Select backend with best average latency
                candidates.sort(key=lambda b: b.metrics.average_latency_ms or float('inf'))
                return candidates[0]
            
            else:
                # Default: return first candidate
                return candidates[0]
    
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
