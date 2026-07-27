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
import anyio
import logging
import os
import time
import threading
import inspect
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Mapping, Sequence, Set, Tuple
from collections import defaultdict

from .model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from .model_catalog.catalog import AIServiceCatalog
from .model_catalog.sources.deployments import BackendDeploymentSource
from .model_catalog.sources.static import CatalogSourceResult, SourceMetadata

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

    def __post_init__(self) -> None:
        for field_name in (
            "supported_tasks",
            "supported_models",
            "hardware_types",
            "protocols",
        ):
            values = getattr(self, field_name)
            if isinstance(values, str) or not isinstance(
                values, (set, frozenset, list, tuple)
            ):
                raise TypeError(f"{field_name} must be a collection of strings")
            normalized = {
                item.strip()
                for item in values
                if isinstance(item, str) and item.strip()
            }
            if len(normalized) != len(values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            setattr(self, field_name, normalized)
        if (
            isinstance(self.max_batch_size, bool)
            or not isinstance(self.max_batch_size, int)
            or self.max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(self.supports_streaming, bool):
            raise TypeError("supports_streaming must be boolean")
        if not isinstance(self.supports_batching, bool):
            raise TypeError("supports_batching must be boolean")


_TASK_OPERATIONS: Dict[str, Operation] = {
    "text-generation": Operation.TEXT_GENERATE,
    "text_generation": Operation.TEXT_GENERATE,
    "generate": Operation.TEXT_GENERATE,
    "chat": Operation.TEXT_CHAT,
    "conversational": Operation.TEXT_CHAT,
    "embedding": Operation.EMBEDDING_GENERATE,
    "embeddings": Operation.EMBEDDING_GENERATE,
    "text-embedding": Operation.EMBEDDING_GENERATE,
    "feature-extraction": Operation.EMBEDDING_GENERATE,
    "vision": Operation.VISION_GENERATE,
    "audio": Operation.AUDIO_TRANSCRIBE,
    "transcription": Operation.AUDIO_TRANSCRIBE,
    "text-to-speech": Operation.AUDIO_SYNTHESIZE,
}


def _catalog_capabilities(
    tasks: Sequence[str],
    *,
    streaming: bool = False,
    batching: bool = False,
    max_batch_size: Optional[int] = None,
) -> Tuple[CapabilityDescriptor, ...]:
    """Translate legacy task labels once at the registration boundary."""

    operations = {
        _TASK_OPERATIONS[item.strip().casefold()]
        for item in tasks
        if isinstance(item, str) and item.strip().casefold() in _TASK_OPERATIONS
    }
    if not operations:
        return ()
    if streaming:
        operations.add(Operation.STREAM)
    if batching:
        operations.add(Operation.BATCH)
    inputs = {Modality.TEXT}
    outputs = {Modality.TEXT}
    if Operation.EMBEDDING_GENERATE in operations:
        outputs.add(Modality.EMBEDDING)
    if Operation.VISION_GENERATE in operations:
        inputs.add(Modality.IMAGE)
    if Operation.AUDIO_TRANSCRIBE in operations:
        inputs.add(Modality.AUDIO)
    if Operation.AUDIO_SYNTHESIZE in operations:
        outputs.add(Modality.AUDIO)
    return (
        CapabilityDescriptor(
            operations=tuple(sorted(operations, key=lambda item: item.value)),
            input_modalities=tuple(sorted(inputs, key=lambda item: item.value)),
            output_modalities=tuple(sorted(outputs, key=lambda item: item.value)),
            max_batch_size=max_batch_size if batching else None,
        ),
    )


@dataclass(frozen=True)
class ProviderRegistration:
    """Named construction data for a lazily instantiated API provider."""

    name: str
    backend_module_path: str
    backend_class_name: str
    env_key_primary: Optional[str]
    env_key_secondary: Optional[str]
    default_base_url: Optional[str]
    display_name: str
    supported_tasks: frozenset[str]
    descriptor: Optional[ProviderDescriptor] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip().casefold()
        if not name:
            raise ValueError("provider registration name must not be empty")
        for field_name in (
            "backend_module_path",
            "backend_class_name",
            "display_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
            object.__setattr__(self, field_name, value.strip())
        for field_name in (
            "env_key_primary",
            "env_key_secondary",
            "default_base_url",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be non-empty text or None")
        tasks = frozenset(
            item.strip()
            for item in self.supported_tasks
            if isinstance(item, str) and item.strip()
        )
        if not tasks or len(tasks) != len(self.supported_tasks):
            raise ValueError("supported_tasks must contain non-empty task names")
        descriptor = self.descriptor or ProviderDescriptor(
            name=name,
            display_name=self.display_name,
            capabilities=_catalog_capabilities(tuple(tasks), streaming=True),
            lifecycle=LifecycleState.DECLARED,
            state=OperationalState(known=True),
            provenance=(
                Provenance(source="inference-backend-manager.providers"),
            ),
        )
        if descriptor.name != name:
            raise ValueError("provider descriptor name must match registration name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "supported_tasks", tasks)
        object.__setattr__(self, "descriptor", descriptor)

    # Alternate named spellings used by the first typed API draft.
    @property
    def module_path(self) -> str:
        return self.backend_module_path

    @property
    def class_name(self) -> str:
        return self.backend_class_name

    @property
    def base_url(self) -> Optional[str]:
        return self.default_base_url

    @property
    def legacy_tuple(self) -> Tuple[Any, ...]:
        return (
            self.backend_module_path,
            self.backend_class_name,
            self.env_key_primary,
            self.env_key_secondary,
            self.default_base_url,
            self.display_name,
            set(self.supported_tasks),
        )

    def __getitem__(self, index: int) -> Any:
        warnings.warn(
            "indexed provider registrations are deprecated; use named fields",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.legacy_tuple[index]

    def __iter__(self):
        warnings.warn(
            "tuple-unpacking provider registrations is deprecated; use named fields",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter(self.legacy_tuple)

    def __len__(self) -> int:
        return 7


ProviderSpec = ProviderRegistration
ProviderBackendSpec = ProviderRegistration
ProviderConfiguration = ProviderRegistration


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


@dataclass(frozen=True)
class BackendRegistration:
    """Typed input accepted by :meth:`register_backend`."""

    backend_id: str
    backend_type: BackendType
    name: str
    instance: Optional[Any] = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    endpoint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    status: Optional[BackendStatus] = None
    configured: Optional[bool] = True
    authorized: Optional[bool] = None
    reachable: Optional[bool] = None
    live: Optional[bool] = None
    ready: Optional[bool] = None
    healthy: Optional[bool] = None
    routable: Optional[bool] = None
    provider: Optional[ProviderDescriptor] = None
    models: Tuple[ModelDescriptor, ...] = ()
    deployments: Tuple[DeploymentDescriptor, ...] = ()
    bindings: Tuple[RouterBinding, ...] = ()


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
    aliases: Tuple[str, ...] = ()
    configured: Optional[bool] = True
    authorized: Optional[bool] = None
    reachable: Optional[bool] = None
    live: Optional[bool] = None
    ready: Optional[bool] = None
    healthy: Optional[bool] = None
    routable: Optional[bool] = None
    provider: Optional[ProviderDescriptor] = None
    models: Tuple[ModelDescriptor, ...] = ()
    deployments: Tuple[DeploymentDescriptor, ...] = ()
    bindings: Tuple[RouterBinding, ...] = ()


class BackendManagerCatalogSource:
    """Thread-safe, side-effect-free source published by the manager."""

    source = "inference-backend-manager"
    precedence = 40
    side_effecting = False

    def __init__(self, source: str = source, precedence: int = precedence) -> None:
        self.source = source
        self.precedence = precedence
        self._lock = threading.RLock()
        self._snapshot = CatalogSnapshot()

    def replace(self, snapshot: CatalogSnapshot) -> None:
        if not isinstance(snapshot, CatalogSnapshot):
            raise TypeError("catalog source accepts only CatalogSnapshot values")
        with self._lock:
            self._snapshot = snapshot

    def load(self) -> CatalogSourceResult:
        with self._lock:
            snapshot = self._snapshot
        return CatalogSourceResult(
            snapshot=snapshot,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=snapshot.revision,
            ),
        )

    snapshot = load
    read = load


BackendCatalogSource = BackendManagerCatalogSource


def _provider_spec(
    name: str,
    module_path: str,
    class_name: str,
    env_primary: Optional[str],
    env_secondary: Optional[str],
    base_url: Optional[str],
    display_name: str,
    tasks: Set[str],
    *,
    aliases: Tuple[str, ...] = (),
) -> ProviderRegistration:
    return ProviderRegistration(
        name=name,
        backend_module_path=module_path,
        backend_class_name=class_name,
        env_key_primary=env_primary,
        env_key_secondary=env_secondary,
        default_base_url=base_url,
        display_name=display_name,
        supported_tasks=frozenset(tasks),
        descriptor=ProviderDescriptor(
            name=name,
            display_name=display_name,
            aliases=aliases,
            capabilities=_catalog_capabilities(tuple(tasks), streaming=True),
            lifecycle=LifecycleState.DECLARED,
            state=OperationalState(known=True),
            provenance=(
                Provenance(source="inference-backend-manager.providers"),
            ),
        ),
    )


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
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        catalog: Optional[Any] = None,
        catalog_source: Optional[Any] = None,
        deployment_source: Optional[Any] = None,
        provider_registry: Optional[Mapping[str, Any]] = None,
    ):
        self.config = dict(config or {})
        catalog = self.config.get("catalog", catalog)
        catalog_source = self.config.get(
            "catalog_source",
            self.config.get("deployment_source", catalog_source or deployment_source),
        )
        provider_registry = self.config.get("provider_registry", provider_registry)

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
        self._backend_aliases: Dict[str, str] = {}
        
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

        registry = self._PROVIDER_REGISTRY if provider_registry is None else provider_registry
        self._provider_registry = {
            str(name).strip().casefold(): self._adapt_provider_registration(
                str(name).strip().casefold(), value
            )
            for name, value in registry.items()
        }

        self._load_registry_state()
        self._catalog = AIServiceCatalog() if catalog is None else catalog
        source_name = self.config.get(
            "catalog_source_name", BackendManagerCatalogSource.source
        )
        source_precedence = self.config.get(
            "catalog_source_precedence", BackendManagerCatalogSource.precedence
        )
        self._catalog_source = (
            BackendManagerCatalogSource(source_name, source_precedence)
            if catalog_source is None
            else catalog_source
        )
        if not callable(getattr(self._catalog_source, "replace", None)):
            raise TypeError("catalog_source must provide replace(CatalogSnapshot)")
        if not callable(getattr(self._catalog_source, "load", None)):
            raise TypeError("catalog_source must provide load()")
        self._catalog_source_registered = False
        with self._lock:
            for backend in self.backends.values():
                self._project_backend_catalog_records(backend)
            self._publish_catalog_locked(refresh=False)
        self._register_catalog_source()
        
        logger.info("InferenceBackendManager initialized")

    @property
    def catalog(self) -> Any:
        """Canonical catalog receiving this manager's deployment projection."""

        return self._catalog

    @property
    def catalog_source(self) -> Any:
        """Side-effect-free source containing this manager's latest snapshot."""

        return self._catalog_source

    @property
    def deployment_source(self) -> Any:
        """Compatibility name for :attr:`catalog_source`."""

        return self._catalog_source

    @property
    def catalog_revision(self) -> str:
        return self.get_catalog_snapshot().revision  # type: ignore[return-value]

    def get_catalog_snapshot(self) -> CatalogSnapshot:
        """Return the immutable current snapshot without probing a backend."""

        result = self._catalog_source.load()
        return result.snapshot if hasattr(result, "snapshot") else result

    catalog_snapshot = get_catalog_snapshot

    def _register_catalog_source(self) -> None:
        register = getattr(self._catalog, "register_source", None)
        if not callable(register):
            raise TypeError("catalog must provide register_source()")
        source_name = self._catalog_source.source
        states = getattr(self._catalog, "source_states", None)
        existing = (
            {item.name for item in states()}
            if callable(states)
            else set()
        )
        if source_name in existing:
            logger.warning(
                "Catalog source %s is already registered; projection remains local",
                source_name,
            )
            return
        register(
            source_name,
            self._catalog_source,
            precedence=self._catalog_source.precedence,
            side_effecting=False,
            load=True,
        )
        self._catalog_source_registered = True

    def _publish_catalog_locked(self, *, refresh: bool = True) -> None:
        """Atomically publish one deterministic generation."""

        providers: Dict[str, ProviderDescriptor] = {}
        models: Dict[str, ModelDescriptor] = {}
        deployments: Dict[str, DeploymentDescriptor] = {}
        bindings: Dict[str, RouterBinding] = {}
        for backend_id in sorted(self.backends):
            backend = self.backends[backend_id]
            if backend.provider is not None:
                providers[backend.provider.provider_id] = backend.provider  # type: ignore[index]
            for model in backend.models:
                models[model.model_id] = model  # type: ignore[index]
            for deployment in backend.deployments:
                deployments[deployment.deployment_id] = deployment  # type: ignore[index]
            for binding in backend.bindings:
                bindings[binding.binding_id] = binding  # type: ignore[index]
        snapshot = CatalogSnapshot(
            providers=tuple(providers.values()),
            models=tuple(models.values()),
            deployments=tuple(deployments.values()),
            bindings=tuple(bindings.values()),
        )
        self._catalog_source.replace(snapshot)
        if not refresh or not self._catalog_source_registered:
            return
        try:
            result = self._catalog.refresh(
                (self._catalog_source.source,),
                raise_on_error=False,
            )
            if result.failed:
                logger.warning(
                    "Backend catalog source retained its prior catalog generation"
                )
        except Exception as exc:
            # Publishing is derived state and must not roll back a successful
            # runtime registration or endpoint lifecycle operation.
            logger.warning(
                "Backend catalog source could not be synchronized: %s",
                type(exc).__name__,
            )

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
            "aliases": list(backend_info.aliases),
            "configured": backend_info.configured,
            "authorized": backend_info.authorized,
            "reachable": backend_info.reachable,
            "live": backend_info.live,
            "ready": backend_info.ready,
            "healthy": backend_info.healthy,
            "routable": backend_info.routable,
            "provider": (
                backend_info.provider.to_dict()
                if backend_info.provider is not None
                else None
            ),
            "models": [item.to_dict() for item in backend_info.models],
            "deployments": [item.to_dict() for item in backend_info.deployments],
            "bindings": [item.to_dict() for item in backend_info.bindings],
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
            aliases=tuple(payload.get("aliases", ())),
            configured=payload.get("configured", True),
            authorized=payload.get("authorized"),
            reachable=payload.get("reachable"),
            live=payload.get("live"),
            ready=payload.get("ready"),
            healthy=payload.get("healthy"),
            routable=payload.get("routable"),
            provider=(
                ProviderDescriptor.from_dict(payload["provider"])
                if payload.get("provider")
                else None
            ),
            models=tuple(
                ModelDescriptor.from_dict(item)
                for item in payload.get("models", ())
            ),
            deployments=tuple(
                DeploymentDescriptor.from_dict(item)
                for item in payload.get("deployments", ())
            ),
            bindings=tuple(
                RouterBinding.from_dict(item)
                for item in payload.get("bindings", ())
            ),
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
        self._backend_aliases.clear()

        for backend_payload in payload.get("backends", []):
            try:
                backend_info = self._deserialize_backend_info(backend_payload)
            except Exception as exc:
                logger.debug(f"Skipping backend registry entry during load: {exc}")
                continue

            self.backends[backend_info.backend_id] = backend_info
            self.backends_by_type[backend_info.backend_type].append(backend_info.backend_id)
            for alias in backend_info.aliases:
                self._backend_aliases[alias] = backend_info.backend_id
            for task in backend_info.capabilities.supported_tasks:
                if backend_info.backend_id not in self.task_routing[task]:
                    self.task_routing[task].append(backend_info.backend_id)

    def _save_registry_state(self) -> None:
        if not self._persist_registry:
            return

        payload = {
            "backends": [
                self._serialize_backend_info(self.backends[backend_id])
                for backend_id in sorted(self.backends)
            ],
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
                raw_result = await anyio.to_thread.run_sync(lambda: method(**call_kwargs))
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
        protocols = sorted(backend_info.capabilities.protocols) if backend_info else []
        hardware_types = sorted(backend_info.capabilities.hardware_types) if backend_info else []
        placement_node = None
        if backend_info:
            placement_node = (
                backend_info.metadata.get("placement_node")
                or backend_info.metadata.get("node_name")
                or backend_info.metadata.get("peer_id")
                or backend_info.metadata.get("worker_id")
            )

        merged = dict(result)
        merged.setdefault("backend_id", backend_id)
        merged.setdefault("backend_type", backend_type)
        merged.setdefault("endpoint", endpoint)
        merged.setdefault("protocol", protocols[0] if protocols else None)
        merged.setdefault("protocols", protocols)
        merged.setdefault("hardware_type", hardware_types[0] if hardware_types else merged.get("device"))
        merged.setdefault("hardware_types", hardware_types)
        merged.setdefault("placement_node", placement_node)
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
                    protocol=merged.get("protocol"),
                    protocols=merged.get("protocols"),
                    hardware_type=merged.get("hardware_type"),
                    hardware_types=merged.get("hardware_types"),
                    placement_node=merged.get("placement_node"),
                )
                if isinstance(recorded, dict):
                    merged = recorded
            except Exception as exc:
                logger.warning(f"Result recorder failed for backend {backend_id}: {exc}")

        return merged
    
    @staticmethod
    def _coerce_backend_type(value: Any) -> BackendType:
        if isinstance(value, BackendType):
            return value
        if isinstance(value, str):
            return BackendType(value.strip().casefold())
        raise TypeError("backend_type must be a BackendType or string value")

    @staticmethod
    def _coerce_backend_status(value: Any) -> BackendStatus:
        if isinstance(value, BackendStatus):
            return value
        if isinstance(value, str):
            return BackendStatus(value.strip().casefold())
        raise TypeError("status must be a BackendStatus or string value")

    @staticmethod
    def _coerce_capabilities(value: Any) -> BackendCapabilities:
        if value is None:
            return BackendCapabilities()
        if isinstance(value, BackendCapabilities):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("capabilities must be BackendCapabilities or a mapping")
        return BackendCapabilities(
            supported_tasks=set(value.get("supported_tasks", ())),
            supported_models=set(value.get("supported_models", ())),
            max_batch_size=value.get("max_batch_size", 1),
            supports_streaming=value.get("supports_streaming", False),
            supports_batching=value.get("supports_batching", False),
            hardware_types=set(value.get("hardware_types", ())),
            protocols=set(value.get("protocols", ())),
        )

    @staticmethod
    def _typed_records(
        values: Any, record_type: Any, field_name: str
    ) -> Tuple[Any, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes, Mapping)) or not isinstance(
            values, Sequence
        ):
            raise TypeError(f"{field_name} must be a sequence")
        return tuple(
            item
            if isinstance(item, record_type)
            else record_type.from_dict(item)
            for item in values
        )

    @classmethod
    def _coerce_backend_registration(
        cls,
        backend_id: Any,
        backend_type: Any,
        name: Any,
        instance: Any,
        capabilities: Any,
        endpoint: Any,
        metadata: Any,
        aliases: Any,
        status: Any,
        configured: Any,
        authorized: Any,
        reachable: Any,
        live: Any,
        ready: Any,
        healthy: Any,
        routable: Any,
        provider: Any,
        models: Any,
        deployments: Any,
        bindings: Any,
    ) -> BackendRegistration:
        if isinstance(backend_id, BackendRegistration):
            if any(
                value is not None
                for value in (
                    backend_type,
                    name,
                    capabilities,
                    endpoint,
                    metadata,
                    status,
                    provider,
                )
            ) or aliases or models or deployments or bindings:
                raise ValueError(
                    "a BackendRegistration cannot be combined with other fields"
                )
            record = backend_id
            return cls._coerce_backend_registration(
                record.backend_id,
                record.backend_type,
                record.name,
                record.instance,
                record.capabilities,
                record.endpoint,
                record.metadata,
                record.aliases,
                record.status,
                record.configured,
                record.authorized,
                record.reachable,
                record.live,
                record.ready,
                record.healthy,
                record.routable,
                record.provider,
                record.models,
                record.deployments,
                record.bindings,
            )
        if isinstance(backend_id, Mapping):
            if any(value is not None for value in (backend_type, name, capabilities)):
                raise ValueError(
                    "a registration mapping cannot be combined with other fields"
                )
            values = dict(backend_id)
            allowed = {
                "backend_id", "backend_type", "name", "instance", "capabilities",
                "endpoint", "metadata", "aliases", "status", "configured",
                "authorized", "reachable", "live", "ready", "healthy",
                "routable", "provider", "models", "deployment", "deployments",
                "bindings",
            }
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(
                    "unknown backend registration fields: %s"
                    % ", ".join(sorted(unknown))
                )
            if "deployment" in values:
                if "deployments" in values:
                    raise ValueError(
                        "registration cannot set deployment and deployments"
                    )
                values["deployments"] = (values.pop("deployment"),)
            return cls._coerce_backend_registration(
                values.get("backend_id"),
                values.get("backend_type"),
                values.get("name"),
                values.get("instance"),
                values.get("capabilities"),
                values.get("endpoint"),
                values.get("metadata"),
                values.get("aliases", ()),
                values.get("status"),
                values.get("configured", True),
                values.get("authorized"),
                values.get("reachable"),
                values.get("live"),
                values.get("ready"),
                values.get("healthy"),
                values.get("routable"),
                values.get("provider"),
                values.get("models", ()),
                values.get("deployments", ()),
                values.get("bindings", ()),
            )
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id must be non-empty text")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be non-empty text")
        if endpoint is not None and (
            not isinstance(endpoint, str) or not endpoint.strip()
        ):
            raise ValueError("endpoint must be non-empty text or None")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if isinstance(aliases, str) or not isinstance(
            aliases, (list, tuple, set, frozenset)
        ):
            raise TypeError("aliases must be a collection of strings")
        normalized_aliases = tuple(
            sorted(
                {
                    item.strip().casefold()
                    for item in aliases
                    if isinstance(item, str) and item.strip()
                }
            )
        )
        if len(normalized_aliases) != len(aliases):
            raise ValueError("aliases must contain unique non-empty strings")
        if backend_id.strip().casefold() in normalized_aliases:
            raise ValueError("backend_id cannot also be an alias")
        for field_name, value in (
            ("configured", configured),
            ("authorized", authorized),
            ("reachable", reachable),
            ("live", live),
            ("ready", ready),
            ("healthy", healthy),
            ("routable", routable),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be boolean or None")
        if provider is not None and not isinstance(provider, ProviderDescriptor):
            if isinstance(provider, Mapping):
                provider = ProviderDescriptor.from_dict(provider)
            else:
                raise TypeError("provider must be a ProviderDescriptor")
        return BackendRegistration(
            backend_id=backend_id.strip(),
            backend_type=cls._coerce_backend_type(backend_type),
            name=name.strip(),
            instance=instance,
            capabilities=cls._coerce_capabilities(capabilities),
            endpoint=endpoint.strip() if endpoint is not None else None,
            metadata=dict(metadata),
            aliases=normalized_aliases,
            status=(
                cls._coerce_backend_status(status)
                if status is not None
                else None
            ),
            configured=configured,
            authorized=authorized,
            reachable=reachable,
            live=live,
            ready=ready,
            healthy=healthy,
            routable=routable,
            provider=provider,
            models=cls._typed_records(models, ModelDescriptor, "models"),
            deployments=cls._typed_records(
                deployments, DeploymentDescriptor, "deployments"
            ),
            bindings=cls._typed_records(bindings, RouterBinding, "bindings"),
        )

    def _provider_registration(
        self, name: str
    ) -> Optional[ProviderRegistration]:
        return self._provider_registry.get(name)

    def _project_backend_catalog_records(self, backend: BackendInfo) -> None:
        """Create or update canonical records for one runtime backend."""

        observed = any(
            value is not None
            for value in (
                backend.reachable,
                backend.live,
                backend.ready,
                backend.healthy,
                backend.routable,
            )
        )
        if backend.ready is True:
            catalog_status = "ready"
        elif backend.status == BackendStatus.INITIALIZING:
            catalog_status = "initializing"
        elif backend.status == BackendStatus.OFFLINE:
            catalog_status = "stopped"
        elif backend.status == BackendStatus.DEGRADED:
            catalog_status = "degraded"
        else:
            # HEALTHY and UNHEALTHY are health observations, not readiness.
            # Keeping lifecycle configured prevents either from being silently
            # promoted into deployment readiness.
            catalog_status = "configured"
        row = {
            "backend_id": backend.backend_id,
            "backend_type": (
                backend.backend_type.value
                if backend.endpoint is not None
                else "in-process"
            ),
            "name": backend.name,
            "endpoint": backend.endpoint,
            "provider": (
                backend.metadata.get("provider")
                or (
                    backend.backend_id.removeprefix("api_").removeprefix("api-")
                    if backend.backend_type == BackendType.API
                    else None
                )
            ),
            "status": catalog_status,
            "configured": backend.configured,
            "authorized": backend.authorized,
            "reachable": backend.reachable,
            "healthy": backend.healthy,
            "routable": backend.routable,
            "ready": backend.ready,
            "capabilities": {
                "supported_tasks": sorted(
                    {
                        _TASK_OPERATIONS[item.strip().casefold()].value
                        for item in backend.capabilities.supported_tasks
                        if (
                            isinstance(item, str)
                            and item.strip().casefold() in _TASK_OPERATIONS
                        )
                    }
                ),
                "supported_models": sorted(backend.capabilities.supported_models),
                "max_batch_size": backend.capabilities.max_batch_size,
                "supports_streaming": backend.capabilities.supports_streaming,
                "supports_batching": backend.capabilities.supports_batching,
                "protocols": sorted(backend.capabilities.protocols),
            },
            "metadata": {
                key: backend.metadata[key]
                for key in ("provider", "locality")
                if key in backend.metadata
            },
        }
        projection = BackendDeploymentSource([row]).load()
        if projection.error_count or not projection.deployments:
            message = (
                projection.diagnostics[0].message
                if projection.diagnostics
                else "registration did not produce a deployment"
            )
            raise ValueError(
                "backend cannot be represented as a catalog deployment: %s"
                % message
            )

        generated_provider = projection.providers[0]
        provider_spec = self._provider_registration(generated_provider.name)
        if backend.provider is None:
            backend.provider = (
                replace(
                    provider_spec.descriptor,
                    lifecycle=LifecycleState.CONFIGURED,
                    state=OperationalState(known=True, configured=True),
                )
                if provider_spec is not None
                else generated_provider
            )
        if backend.provider.provider_id != generated_provider.provider_id:
            raise ValueError("provider descriptor does not match backend provider")

        if not backend.models:
            backend.models = projection.models
        if any(
            model.provider_id != backend.provider.provider_id
            for model in backend.models
        ):
            raise ValueError("model descriptor provider_id does not match provider")

        if not backend.deployments:
            backend.deployments = projection.deployments
        else:
            generated_by_model = {
                item.model_id: item for item in projection.deployments
            }
            updated = []
            for deployment in backend.deployments:
                if deployment.provider_id != backend.provider.provider_id:
                    raise ValueError(
                        "deployment descriptor provider_id does not match provider"
                    )
                current = generated_by_model.get(deployment.model_id)
                updated.append(
                    replace(
                        deployment,
                        lifecycle=(
                            current.lifecycle
                            if observed and current is not None
                            else deployment.lifecycle
                        ),
                        state=(
                            current.state
                            if observed and current is not None
                            else deployment.state
                        ),
                    )
                )
            backend.deployments = tuple(updated)

        model_ids = {item.model_id for item in backend.models}
        if any(
            item.model_id is not None and item.model_id not in model_ids
            for item in backend.deployments
        ):
            raise ValueError("deployment model_id does not match registered models")

        operations = tuple(
            sorted(
                {
                    operation
                    for deployment in backend.deployments
                    for capability in deployment.capabilities
                    for operation in capability.operations
                    if operation not in (Operation.BATCH, Operation.STREAM)
                },
                key=lambda item: item.value,
            )
        )
        state = OperationalState(
            known=True,
            configured=backend.configured,
            authorized=backend.authorized,
            reachable=backend.reachable,
            healthy=backend.healthy,
            routable=backend.routable,
        )
        if not backend.bindings and operations:
            backend.bindings = tuple(
                RouterBinding(
                    router=str(
                        backend.metadata.get(
                            "router", "inference_backend_manager"
                        )
                    ).strip().casefold(),
                    provider_id=backend.provider.provider_id,
                    model_id=deployment.model_id,
                    deployment_id=deployment.deployment_id,
                    operations=operations,
                    priority=int(backend.metadata.get("priority", 0)),
                    state=state,
                    provenance=(
                        Provenance(
                            source="inference-backend-manager.bindings",
                            source_record_id=backend.backend_id,
                        ),
                    ),
                )
                for deployment in backend.deployments
            )
        elif observed:
            backend.bindings = tuple(
                replace(binding, state=state)
                for binding in backend.bindings
            )
        deployment_ids = {item.deployment_id for item in backend.deployments}
        if any(
            binding.provider_id != backend.provider.provider_id
            or (
                binding.deployment_id is not None
                and binding.deployment_id not in deployment_ids
            )
            for binding in backend.bindings
        ):
            raise ValueError("router binding does not match registered deployment")

    def register_backend(
        self,
        backend_id: Any,
        backend_type: Any = None,
        name: Optional[str] = None,
        instance: Any = None,
        capabilities: Optional[Any] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        aliases: Sequence[str] = (),
        status: Optional[Any] = None,
        configured: Optional[bool] = True,
        authorized: Optional[bool] = None,
        reachable: Optional[bool] = None,
        live: Optional[bool] = None,
        ready: Optional[bool] = None,
        healthy: Optional[bool] = None,
        routable: Optional[bool] = None,
        provider: Optional[ProviderDescriptor] = None,
        models: Sequence[ModelDescriptor] = (),
        deployment: Optional[DeploymentDescriptor] = None,
        deployments: Sequence[DeploymentDescriptor] = (),
        bindings: Sequence[RouterBinding] = (),
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
        if deployment is not None:
            if deployments:
                logger.warning(
                    "registration cannot set both deployment and deployments"
                )
                return False
            deployments = (deployment,)
        try:
            registration = self._coerce_backend_registration(
                backend_id, backend_type, name, instance, capabilities, endpoint,
                metadata, aliases, status, configured, authorized, reachable,
                live, ready, healthy, routable, provider, models, deployments,
                bindings,
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Rejected malformed backend registration: %s", exc)
            return False

        with self._lock:
            backend_id = registration.backend_id
            previous = self.backends.get(backend_id)
            aliases_in_use = {
                alias: owner
                for alias, owner in self._backend_aliases.items()
                if owner != backend_id
            }
            if backend_id.casefold() in aliases_in_use or any(
                alias in self.backends and alias != backend_id
                or alias in aliases_in_use
                for alias in registration.aliases
            ):
                logger.warning("Rejected registration with a duplicate alias")
                return False
            backend_info = BackendInfo(
                backend_id=backend_id,
                backend_type=registration.backend_type,
                name=registration.name,
                instance=registration.instance,
                endpoint=registration.endpoint,
                capabilities=registration.capabilities,
                metadata=dict(registration.metadata),
                status=registration.status or BackendStatus.HEALTHY,
                aliases=registration.aliases,
                configured=registration.configured,
                authorized=registration.authorized,
                reachable=registration.reachable,
                live=registration.live,
                ready=registration.ready,
                healthy=registration.healthy,
                routable=registration.routable,
                provider=registration.provider,
                models=registration.models,
                deployments=registration.deployments,
                bindings=registration.bindings,
            )
            try:
                self._project_backend_catalog_records(backend_info)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Rejected malformed backend registration %s: %s",
                    backend_id,
                    exc,
                )
                return False
            if previous is not None:
                if backend_id in self.backends_by_type.get(
                    previous.backend_type, []
                ):
                    self.backends_by_type[previous.backend_type].remove(backend_id)
                    if not self.backends_by_type[previous.backend_type]:
                        del self.backends_by_type[previous.backend_type]
                for task in previous.capabilities.supported_tasks:
                    if backend_id in self.task_routing.get(task, []):
                        self.task_routing[task].remove(backend_id)
                        if not self.task_routing[task]:
                            del self.task_routing[task]
                for alias in previous.aliases:
                    self._backend_aliases.pop(alias, None)
            self.backends[backend_id] = backend_info
            if backend_id not in self.backends_by_type[registration.backend_type]:
                self.backends_by_type[registration.backend_type].append(backend_id)
            for alias in registration.aliases:
                self._backend_aliases[alias] = backend_id
            
            # Update task routing
            for task in backend_info.capabilities.supported_tasks:
                if backend_id not in self.task_routing[task]:
                    self.task_routing[task].append(backend_id)
            
            logger.info(
                "Registered backend: %s (%s) - Type: %s",
                backend_id,
                registration.name,
                registration.backend_type.value,
            )
            self._publish_catalog_locked()
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
                    if not self.task_routing[task]:
                        del self.task_routing[task]
            for alias in backend_info.aliases:
                self._backend_aliases.pop(alias, None)
            
            # Remove from registry
            del self.backends[backend_id]
            self._publish_catalog_locked()
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
                for alias in backend_info.aliases:
                    self._backend_aliases.pop(alias, None)

                del self.backends[backend_id]
                removed.append(backend_id)

            if removed:
                self._publish_catalog_locked()
                self._save_registry_state()

        if removed:
            logger.info("Pruned stale backends: %s", ", ".join(removed))
        return removed
    
    def get_backend(self, backend_id: str) -> Optional[BackendInfo]:
        """Get information about a specific backend"""
        requested = str(backend_id).strip()
        with self._lock:
            direct = self.backends.get(requested)
            if direct is not None:
                return direct
            owner = self._backend_aliases.get(requested.casefold())
            return self.backends.get(owner) if owner is not None else None

    def get_backend_by_deployment(
        self, deployment_id: str
    ) -> Optional[BackendInfo]:
        """Look up the executable backend owning a typed deployment."""

        with self._lock:
            return next(
                (
                    backend
                    for backend in self.backends.values()
                    if any(
                        item.deployment_id == deployment_id
                        for item in backend.deployments
                    )
                ),
                None,
            )

    def get_provider_descriptor(self, provider: str) -> ProviderDescriptor:
        """Resolve a provider name, stable identity, or alias."""

        requested = str(provider or "").strip().casefold()
        canonical = self._resolve_provider_name(requested)
        spec = self._provider_registration(canonical)
        if spec is not None:
            return spec.descriptor  # type: ignore[return-value]
        with self._lock:
            matches = {
                backend.provider.provider_id: backend.provider
                for backend in self.backends.values()
                if backend.provider is not None
                and requested
                in {
                    backend.provider.name,
                    backend.provider.provider_id,
                    *backend.provider.aliases,
                }
            }
        if len(matches) == 1:
            return next(iter(matches.values()))
        if len(matches) > 1:
            raise ValueError(f"ambiguous provider alias: {provider}")
        raise KeyError(f"unknown provider: {provider}")

    def update_backend_endpoint(
        self,
        backend_id: str,
        endpoint: Optional[str],
        *,
        status: BackendStatus = BackendStatus.INITIALIZING,
    ) -> bool:
        """Replace endpoint identity and clear prior liveness observations."""

        with self._lock:
            backend = self.get_backend(backend_id)
            if backend is None:
                return False
            previous = (
                backend.endpoint,
                backend.status,
                backend.reachable,
                backend.live,
                backend.ready,
                backend.healthy,
                backend.routable,
                backend.deployments,
                backend.bindings,
            )
            backend.endpoint = endpoint
            backend.status = self._coerce_backend_status(status)
            backend.reachable = None
            backend.live = None
            backend.ready = None
            backend.healthy = None
            backend.routable = None
            backend.deployments = ()
            backend.bindings = ()
            try:
                self._project_backend_catalog_records(backend)
            except (TypeError, ValueError) as exc:
                (
                    backend.endpoint,
                    backend.status,
                    backend.reachable,
                    backend.live,
                    backend.ready,
                    backend.healthy,
                    backend.routable,
                    backend.deployments,
                    backend.bindings,
                ) = previous
                raise ValueError(
                    "endpoint cannot be represented as a catalog deployment"
                ) from exc
            self._publish_catalog_locked()
            self._save_registry_state()
            return True

    set_backend_endpoint = update_backend_endpoint

    def update_backend_liveness(
        self,
        backend_id: str,
        *,
        status: Optional[BackendStatus] = None,
        reachable: Optional[bool] = None,
        live: Optional[bool] = None,
        ready: Optional[bool] = None,
        healthy: Optional[bool] = None,
        routable: Optional[bool] = None,
    ) -> bool:
        """Publish explicitly observed endpoint facts without deriving peers."""

        values = {
            "reachable": reachable,
            "live": live,
            "ready": ready,
            "healthy": healthy,
            "routable": routable,
        }
        if any(
            value is not None and not isinstance(value, bool)
            for value in values.values()
        ):
            raise TypeError("liveness observations must be boolean or None")
        with self._lock:
            backend = self.get_backend(backend_id)
            if backend is None:
                return False
            if status is not None:
                backend.status = self._coerce_backend_status(status)
            for field_name, value in values.items():
                setattr(backend, field_name, value)
            backend.last_seen = time.time()
            self._project_backend_catalog_records(backend)
            self._publish_catalog_locked()
            self._save_registry_state()
            return True

    def update_backend_status(
        self,
        backend_id: str,
        status: BackendStatus,
        *,
        observed: bool = False,
    ) -> bool:
        """Update runtime status without inventing endpoint liveness facts."""

        with self._lock:
            backend = self.get_backend(backend_id)
            if backend is None:
                return False
            backend.status = self._coerce_backend_status(status)
            backend.last_seen = time.time()
            if observed:
                backend.healthy = status == BackendStatus.HEALTHY
            self._project_backend_catalog_records(backend)
            self._publish_catalog_locked()
            self._save_registry_state()
            return True
    
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
            backends = [
                self.backends[backend_id]
                for backend_id in sorted(self.backends)
            ]
            
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
        required_protocols: Optional[List[str]] = None,
        *,
        provider: Optional[str] = None,
        deployment_id: Optional[str] = None,
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
        normalized_preferred_types: Optional[List[BackendType]] = None
        if preferred_types:
            normalized_preferred_types = []
            seen_types: Set[BackendType] = set()
            for item in preferred_types:
                backend_type_value: Optional[BackendType] = None
                if isinstance(item, BackendType):
                    backend_type_value = item
                else:
                    try:
                        backend_type_value = BackendType(str(item).strip().lower())
                    except Exception:
                        backend_type_value = None
                if backend_type_value is None or backend_type_value in seen_types:
                    continue
                seen_types.add(backend_type_value)
                normalized_preferred_types.append(backend_type_value)

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
                if normalized_preferred_types and candidate.backend_type in normalized_preferred_types:
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
                    if self._backend_supports_model(b, model)
                ]

            if provider:
                requested_provider = self._resolve_provider_name(
                    str(provider).strip().casefold()
                )
                candidates = [
                    b for b in candidates
                    if b.provider is not None
                    and requested_provider
                    in {
                        b.provider.name,
                        b.provider.provider_id,
                        *b.provider.aliases,
                    }
                ]

            if deployment_id:
                candidates = [
                    b for b in candidates
                    if any(
                        item.deployment_id == deployment_id
                        for item in b.deployments
                    )
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
            if normalized_preferred_types:
                type_priority = {t: i for i, t in enumerate(normalized_preferred_types)}
                candidates.sort(key=lambda b: type_priority.get(b.backend_type, len(normalized_preferred_types)))
            
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

    @staticmethod
    def _backend_supports_model(
        backend: BackendInfo, requested: str
    ) -> bool:
        if not backend.capabilities.supported_models and not backend.models:
            return True
        if requested in backend.capabilities.supported_models:
            return True
        normalized = str(requested).strip().casefold()
        return any(
            normalized in {model.name, model.model_id, *model.aliases}
            for model in backend.models
        )
    
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
                "supported_tasks": sorted(self.task_routing),
                "catalog_revision": self.catalog_revision,
                "backends": [
                    {
                        "id": b.backend_id,
                        "backend_id": b.backend_id,
                        "name": b.name,
                        "type": b.backend_type.value,
                        "status": b.status.value,
                        "endpoint": b.endpoint,
                        "aliases": list(b.aliases),
                        "tasks": sorted(b.capabilities.supported_tasks),
                        "protocols": sorted(b.capabilities.protocols),
                        "hardware_types": sorted(b.capabilities.hardware_types),
                        "provider_id": (
                            b.provider.provider_id if b.provider else None
                        ),
                        "deployment_ids": [
                            item.deployment_id for item in b.deployments
                        ],
                        "liveness": {
                            "configured": b.configured,
                            "authorized": b.authorized,
                            "reachable": b.reachable,
                            "live": b.live,
                            "ready": b.ready,
                            "healthy": b.healthy,
                            "routable": b.routable,
                        },
                        "placement_node": (
                            b.metadata.get("placement_node")
                            or b.metadata.get("node_name")
                            or b.metadata.get("peer_id")
                            or b.metadata.get("worker_id")
                        ),
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
                    for b in (
                        self.backends[backend_id]
                        for backend_id in sorted(self.backends)
                    )
                ],
                "timestamp": time.time()
            }
    
    def _update_backend_status(
        self,
        backend_id: str,
        status: BackendStatus,
        *,
        observed_health: Optional[bool] = None,
    ) -> None:
        """Internal status update used by health monitoring."""

        with self._lock:
            backend = self.backends.get(backend_id)
            if backend is None:
                return
            backend.status = status
            backend.last_seen = time.time()
            if observed_health is not None:
                backend.healthy = observed_health
                backend.live = observed_health
            self._project_backend_catalog_records(backend)
            self._publish_catalog_locked()
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
                self._update_backend_status(
                    backend_id,
                    BackendStatus.UNHEALTHY,
                    observed_health=False,
                )
    
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
                
                if isinstance(result, Mapping):
                    observed = {
                        name: result.get(name)
                        for name in (
                            "reachable", "live", "ready", "healthy", "routable"
                        )
                    }
                    if any(
                        value is not None and not isinstance(value, bool)
                        for value in observed.values()
                    ):
                        raise TypeError(
                            "health check observations must be boolean or None"
                        )
                    successful = (
                        observed["healthy"]
                        if observed["healthy"] is not None
                        else observed["live"]
                    )
                    if successful is None:
                        successful = bool(result)
                    self.update_backend_liveness(
                        backend_id,
                        status=(
                            BackendStatus.HEALTHY
                            if successful
                            else BackendStatus.UNHEALTHY
                        ),
                        **observed,
                    )
                    return bool(successful)
                successful = bool(result)
                self._update_backend_status(
                    backend_id,
                    (
                        BackendStatus.HEALTHY
                        if successful
                        else BackendStatus.UNHEALTHY
                    ),
                    observed_health=successful,
                )
                return successful
            except Exception as e:
                logger.error(f"Health check error for {backend_id}: {e}")
                self._update_backend_status(
                    backend_id,
                    BackendStatus.UNHEALTHY,
                    observed_health=False,
                )
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

    # ------------------------------------------------------------------
    # API provider configuration helpers
    # ------------------------------------------------------------------

    #: Named provider construction records. Legacy seven-tuples are accepted
    #: only by :meth:`_adapt_provider_registration`.
    _PROVIDER_REGISTRY: Dict[str, ProviderRegistration] = {
        "xai": _provider_spec(
            "xai", "ipfs_accelerate_py.api_backends.xai", "xai",
            "XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY",
            "https://api.x.ai/v1", "xAI Grok",
            {"text-generation", "embeddings", "vision"},
            aliases=("grok", "xai_grok"),
        ),
        "meta_ai": _provider_spec(
            "meta_ai", "ipfs_accelerate_py.api_backends.meta_ai", "meta_ai",
            "META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY",
            "https://api.llamameta.net/v1", "Meta AI (Llama / Spark)",
            {"text-generation", "embeddings", "vision"},
            aliases=("meta", "meta_llama", "meta_spark", "spark"),
        ),
        "openai": _provider_spec(
            "openai", "ipfs_accelerate_py.api_backends.openai_api", "openai_api",
            "OPENAI_API_KEY", "ipfs_accelerate_py_OPENAI_API_KEY",
            "https://api.openai.com/v1", "OpenAI",
            {"text-generation", "embeddings", "vision", "audio"},
            aliases=("openai_api",),
        ),
        "claude": _provider_spec(
            "claude", "ipfs_accelerate_py.api_backends.claude", "claude",
            "ANTHROPIC_API_KEY", "ipfs_accelerate_py_ANTHROPIC_API_KEY",
            "https://api.anthropic.com", "Anthropic Claude",
            {"text-generation", "vision"},
            aliases=("anthropic",),
        ),
        "gemini": _provider_spec(
            "gemini", "ipfs_accelerate_py.api_backends.gemini", "gemini",
            "GEMINI_API_KEY", "ipfs_accelerate_py_GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com", "Google Gemini",
            {"text-generation", "embeddings", "vision", "audio"},
        ),
        "groq": _provider_spec(
            "groq", "ipfs_accelerate_py.api_backends.groq", "groq",
            "GROQ_API_KEY", "ipfs_accelerate_py_GROQ_API_KEY",
            "https://api.groq.com/openai/v1", "Groq",
            {"text-generation", "audio"},
        ),
        "hf_tei": _provider_spec(
            "hf_tei", "ipfs_accelerate_py.api_backends.hf_tei", "hf_tei",
            "HF_API_KEY", "ipfs_accelerate_py_HF_API_KEY", None,
            "HuggingFace TEI", {"embeddings"},
        ),
        "hf_tgi": _provider_spec(
            "hf_tgi", "ipfs_accelerate_py.api_backends.hf_tgi", "hf_tgi",
            "HF_API_KEY", "ipfs_accelerate_py_HF_API_KEY", None,
            "HuggingFace TGI", {"text-generation"},
        ),
        "ollama": _provider_spec(
            "ollama", "ipfs_accelerate_py.api_backends.ollama", "ollama",
            None, None, "http://localhost:11434", "Ollama",
            {"text-generation", "embeddings", "vision"},
        ),
        "vllm": _provider_spec(
            "vllm", "ipfs_accelerate_py.api_backends.vllm", "vllm",
            None, None, "http://localhost:8000/v1", "vLLM",
            {"text-generation", "embeddings"},
        ),
    }

    # Alias → canonical name
    _PROVIDER_ALIASES: Dict[str, str] = {
        "grok": "xai",
        "xai_grok": "xai",
        "meta": "meta_ai",
        "spark": "meta_ai",
        "meta_spark": "meta_ai",
        "meta_llama": "meta_ai",
        "anthropic": "claude",
        "openai_api": "openai",
    }

    @staticmethod
    def _adapt_provider_registration(
        name: str, value: Any
    ) -> ProviderRegistration:
        """The sole adapter for deprecated seven-position provider tuples."""

        if isinstance(value, ProviderRegistration):
            if value.name != name:
                raise ValueError(
                    "provider registration name must match its registry key"
                )
            return value
        if isinstance(value, tuple):
            warnings.warn(
                "Tuple-shaped provider registrations are deprecated; "
                "use ProviderRegistration",
                DeprecationWarning,
                stacklevel=3,
            )
            if len(value) != 7:
                raise ValueError(
                    "legacy provider registration must contain 7 fields"
                )
            return ProviderRegistration(
                name=name,
                backend_module_path=value[0],
                backend_class_name=value[1],
                env_key_primary=value[2],
                env_key_secondary=value[3],
                default_base_url=value[4],
                display_name=value[5],
                supported_tasks=frozenset(value[6]),
            )
        raise TypeError(
            "provider registration must be ProviderRegistration or a legacy tuple"
        )

    def _resolve_provider_name(self, provider: str) -> str:
        """Normalise a provider alias to the canonical name."""
        requested = str(provider).strip().casefold()
        canonical = self._PROVIDER_ALIASES.get(requested, requested)
        if canonical in self._provider_registry:
            return canonical
        matches = [
            name
            for name, spec in self._provider_registry.items()
            if requested in spec.descriptor.aliases
        ]
        return matches[0] if len(matches) == 1 else canonical

    def configure_provider(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Instantiate and register an external API inference provider.

        This method looks up the correct backend class for *provider*, creates
        an instance with the given credentials, and registers it with the
        manager so that :py:meth:`execute_task` can route requests to it.

        Alias resolution is applied so callers can pass e.g. ``"grok"`` and
        the xAI backend will be selected.

        Args:
            provider:  Provider name or alias (e.g. ``"xai"``, ``"grok"``,
                       ``"meta_ai"``, ``"openai"``, ``"claude"``).
            api_key:   API key.  When *None* the relevant environment variable
                       is checked automatically.
            base_url:  Optional override for the provider's API base URL.
            **kwargs:  Additional keyword arguments forwarded to the backend
                       constructor via its ``metadata`` dict.

        Returns:
            A status dict with keys ``provider``, ``configured`` (bool), and
            ``backend_id``.
        """
        import importlib

        canonical = self._resolve_provider_name(provider)
        spec = self._provider_registration(canonical)
        if spec is None:
            logger.warning("configure_provider: unknown provider '%s'", provider)
            return {"provider": provider, "configured": False,
                    "error": f"Unknown provider '{provider}'"}

        mod_path = spec.backend_module_path
        cls_name = spec.backend_class_name
        env_primary = spec.env_key_primary
        env_secondary = spec.env_key_secondary
        default_base_url = spec.default_base_url
        display_name = spec.display_name
        supported_tasks = spec.supported_tasks

        # Resolve API key from env if not supplied
        resolved_key = api_key
        if not resolved_key and env_primary:
            resolved_key = (
                os.environ.get(env_primary)
                or os.environ.get(env_secondary or "")
                or ""
            )

        # Resolve base URL
        resolved_url = base_url or default_base_url

        # Import and instantiate the backend
        try:
            module = importlib.import_module(mod_path)
            cls = getattr(module, cls_name)
            metadata: Dict[str, Any] = {"api_key": resolved_key}
            if resolved_url:
                metadata["api_base"] = resolved_url
            metadata.update(kwargs)
            instance = cls(resources={}, metadata=metadata)
        except Exception as exc:
            logger.warning(
                "configure_provider: failed to instantiate '%s': %s",
                canonical, exc,
            )
            return {"provider": provider, "configured": False, "error": str(exc)}

        backend_id = f"api_{canonical}"
        caps = BackendCapabilities(
            supported_tasks=supported_tasks,
            supports_streaming=True,
            protocols={"http"},
        )
        ok = self.register_backend(
            backend_id=backend_id,
            backend_type=BackendType.API,
            name=display_name,
            instance=instance,
            capabilities=caps,
            endpoint=resolved_url,
            metadata={"provider": canonical, "api_key_set": bool(resolved_key)},
        )
        logger.info(
            "configure_provider: registered '%s' (backend_id=%s, key_set=%s)",
            canonical, backend_id, bool(resolved_key),
        )
        return {"provider": canonical, "configured": ok, "backend_id": backend_id}

    def auto_discover_api_providers(self) -> List[str]:
        """Scan well-known environment variables and register any API provider
        whose key is present.

        Returns:
            List of canonical provider names that were registered.
        """
        registered: List[str] = []
        for canonical, spec in self._provider_registry.items():
            env_primary = spec.env_key_primary
            env_secondary = spec.env_key_secondary
            if env_primary is None:
                continue
            key = os.environ.get(env_primary) or os.environ.get(env_secondary or "")
            if key:
                result = self.configure_provider(canonical, api_key=key)
                if result.get("configured"):
                    registered.append(canonical)
        if registered:
            logger.info("auto_discover_api_providers: registered %s", registered)
        return registered


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
