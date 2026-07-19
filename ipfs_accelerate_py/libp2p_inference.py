"""
MCP++ P2P facade for distributed inference.

Historically this module exposed custom libp2p stream protocols for discovery,
status, and inference. Those protocols parsed unauthenticated plaintext JSON and
the request path was never completed. The public Python API is kept for callers
such as ``UnifiedInferenceService``, but network execution now goes through the
canonical MCP++ TaskQueue tool transport.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Kept as a compatibility signal for callers that introspect this module. It
# refers only to the removed raw stream implementation, not MCP++ p2p support.
HAVE_LIBP2P = False
HAVE_MCPPLUSPLUS_P2P = True


class PeerCapability(Enum):
    """Capabilities that a peer can offer."""

    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"
    IMAGE_GENERATION = "image-generation"
    IMAGE_CLASSIFICATION = "image-classification"
    AUDIO_TRANSCRIPTION = "audio-transcription"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


@dataclass
class PeerInfo:
    """Information about a P2P peer."""

    peer_id: str
    addresses: List[str] = field(default_factory=list)
    capabilities: Set[PeerCapability] = field(default_factory=set)
    models: Set[str] = field(default_factory=set)
    last_seen: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None
    active_requests: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """P2P inference request."""

    request_id: str
    task: str
    model: str
    inputs: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """P2P inference response."""

    request_id: str
    result: Any
    peer_id: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


def _extract_peer_id_from_multiaddr(multiaddr: str) -> str:
    """Best-effort extraction of a peer id from a multiaddr string."""
    parts = [part for part in str(multiaddr or "").split("/") if part]
    for index, part in enumerate(parts):
        if part == "p2p" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _normalize_capability(value: Any) -> Optional[PeerCapability]:
    """Convert a raw capability value into a known PeerCapability."""
    raw = value.value if isinstance(value, PeerCapability) else str(value or "").strip()
    if not raw:
        return None
    try:
        return PeerCapability(raw)
    except ValueError:
        return None


def _peer_from_record(record: Dict[str, Any]) -> Optional[PeerInfo]:
    """Normalize MCP++ peer records into the historical PeerInfo dataclass."""
    peer_id = str(record.get("peer_id") or record.get("id") or "").strip()
    multiaddr = str(record.get("multiaddr") or record.get("remote_multiaddr") or "").strip()
    if not peer_id and multiaddr:
        peer_id = _extract_peer_id_from_multiaddr(multiaddr)
    if not peer_id:
        return None

    raw_capabilities = record.get("capabilities") or record.get("tasks") or []
    if isinstance(raw_capabilities, dict):
        raw_capabilities = (
            raw_capabilities.get("supported_tasks")
            or raw_capabilities.get("tasks")
            or raw_capabilities.get("capabilities")
            or []
        )
    if isinstance(raw_capabilities, str):
        raw_capabilities = [raw_capabilities]

    capabilities = {
        capability
        for capability in (_normalize_capability(item) for item in raw_capabilities)
        if capability is not None
    }

    raw_models = record.get("models") or record.get("supported_models") or []
    if isinstance(raw_models, str):
        raw_models = [raw_models]

    addresses = list(record.get("addresses") or [])
    if multiaddr and multiaddr not in addresses:
        addresses.append(multiaddr)

    metadata = dict(record)
    if multiaddr:
        metadata["multiaddr"] = multiaddr

    return PeerInfo(
        peer_id=peer_id,
        addresses=[str(addr) for addr in addresses if str(addr).strip()],
        capabilities=capabilities,
        models={str(model) for model in raw_models if str(model).strip()},
        metadata=metadata,
    )


class LibP2PInferenceNode:
    """
    Compatibility node for distributed inference over MCP++ p2p tools.

    The class name is retained to avoid breaking older code, but it no longer
    starts custom libp2p stream handlers. Remote inference is routed through
    ``p2p_taskqueue_call_tool`` against the standard ``inference_run`` tool.
    """

    REMOTE_INFERENCE_TOOL = "inference_run"
    TRANSPORT = "mcpplusplus-p2p"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.host = None
        self.peer_id: Optional[str] = None

        self.peers: Dict[str, PeerInfo] = {}
        self._peers_lock = asyncio.Lock()

        self.bootstrap_peers = list(self.config.get("bootstrap_peers", []) or [])
        self.local_capabilities: Set[PeerCapability] = set()
        self.local_models: Set[str] = set()

        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.request_timeout = float(self.config.get("request_timeout", 30.0))
        self.discovery_interval = float(self.config.get("discovery_interval", 60.0))
        self.enable_mdns = bool(self.config.get("enable_mdns", True))

        self._discovery_task: Optional[asyncio.Task[Any]] = None
        self._heartbeat_task: Optional[asyncio.Task[Any]] = None
        self.is_running = False

        for record in self.config.get("remote_peers", []) or self.config.get("peers", []) or []:
            if isinstance(record, dict):
                peer = _peer_from_record(record)
                if peer is not None:
                    self.peers[peer.peer_id] = peer

        for multiaddr in self.bootstrap_peers:
            peer_id = _extract_peer_id_from_multiaddr(str(multiaddr))
            if peer_id:
                self.peers.setdefault(
                    peer_id,
                    PeerInfo(
                        peer_id=peer_id,
                        addresses=[str(multiaddr)],
                        metadata={"multiaddr": str(multiaddr), "bootstrap": True},
                    ),
                )

    async def start(self) -> None:
        """Start the MCP++ p2p inference facade."""
        self.peer_id = str(self.config.get("peer_id") or f"mcpplusplus-{uuid.uuid4().hex[:12]}")
        self.is_running = True
        await self._refresh_peer_registry()
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("MCP++ p2p inference facade started with peer ID: %s", self.peer_id)

    async def stop(self) -> None:
        """Stop background discovery tasks."""
        self.is_running = False
        for task in (self._discovery_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._discovery_task = None
        self._heartbeat_task = None
        logger.info("MCP++ p2p inference facade stopped")

    def register_capability(self, capability: PeerCapability | str) -> None:
        """Register a capability this node can provide locally."""
        normalized = _normalize_capability(capability)
        if normalized is not None:
            self.local_capabilities.add(normalized)
            logger.info("Registered capability: %s", normalized.value)

    def register_model(self, model_id: str) -> None:
        """Register a model available on this node."""
        if str(model_id).strip():
            self.local_models.add(str(model_id).strip())
            logger.info("Registered model: %s", model_id)

    async def discover_peers(self) -> List[PeerInfo]:
        """Discover available inference peers via MCP++ peer tooling."""
        await self._refresh_peer_registry()
        async with self._peers_lock:
            return list(self.peers.values())

    async def find_peers_for_task(
        self,
        task: str,
        model: Optional[str] = None,
        max_peers: int = 5,
    ) -> List[PeerInfo]:
        """Find suitable peers for a given inference task."""
        capability = _normalize_capability(task)
        if capability is None:
            logger.warning("Unknown task type: %s", task)
            return []

        await self._refresh_peer_registry()
        async with self._peers_lock:
            suitable_peers = [
                peer
                for peer in self.peers.values()
                if not peer.capabilities or capability in peer.capabilities
            ]

            if model:
                suitable_peers = [
                    peer for peer in suitable_peers if not peer.models or str(model) in peer.models
                ]

            suitable_peers.sort(key=lambda peer: (peer.active_requests, -peer.successful_requests))
            return suitable_peers[: max(1, int(max_peers))]

    async def submit_inference_request(
        self,
        task: str,
        model: str,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> InferenceResponse:
        """Submit an inference request through MCP++ p2p remote tool calls."""
        request_id = f"{self.peer_id or 'mcpplusplus'}_{int(time.time() * 1000)}"
        request = InferenceRequest(
            request_id=request_id,
            task=str(task),
            model=str(model),
            inputs=inputs,
            parameters=parameters or {},
        )

        peers = await self.find_peers_for_task(task, model)
        if not peers:
            return InferenceResponse(
                request_id=request_id,
                result=None,
                peer_id="",
                latency_ms=0.0,
                success=False,
                error=f"No MCP++ peers available for task: {task}",
            )

        request_timeout = float(timeout or self.request_timeout)
        for peer in peers:
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    self._send_inference_request(peer, request),
                    timeout=request_timeout,
                )
                latency_ms = (time.time() - start_time) * 1000.0

                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        tracked = self.peers[peer.peer_id]
                        tracked.successful_requests += 1
                        tracked.total_requests += 1
                        tracked.latency_ms = latency_ms

                return InferenceResponse(
                    request_id=request_id,
                    result=result,
                    peer_id=peer.peer_id,
                    latency_ms=latency_ms,
                    success=True,
                )
            except asyncio.TimeoutError:
                await self._record_peer_failure(peer.peer_id)
                logger.warning("MCP++ inference request to peer %s timed out", peer.peer_id)
            except Exception as exc:
                await self._record_peer_failure(peer.peer_id)
                logger.error("MCP++ inference request to peer %s failed: %s", peer.peer_id, exc)

        return InferenceResponse(
            request_id=request_id,
            result=None,
            peer_id="",
            latency_ms=0.0,
            success=False,
            error="All available MCP++ peers failed to process request",
        )

    async def run_inference(self, **kwargs: Any) -> Dict[str, Any]:
        """Backend-manager compatible inference entry point."""
        model = str(kwargs.get("model") or kwargs.get("model_id") or "")
        task = str(kwargs.get("task") or "text-generation")
        inputs = kwargs.get("inputs", kwargs.get("input_data", kwargs.get("prompt", kwargs.get("data"))))
        parameters = {
            key: value
            for key, value in kwargs.items()
            if key not in {"model", "model_id", "task", "inputs", "input_data", "prompt", "data"}
        }
        response = await self.submit_inference_request(
            task=task,
            model=model,
            inputs=inputs,
            parameters=parameters,
        )
        return {
            "success": response.success,
            "result": response.result,
            "peer_id": response.peer_id,
            "latency_ms": response.latency_ms,
            "error": response.error,
            "transport": self.TRANSPORT,
        }

    async def _send_inference_request(self, peer: PeerInfo, request: InferenceRequest) -> Any:
        """Call a remote MCP++ inference tool on a peer."""
        from ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools import (
            p2p_taskqueue_call_tool,
        )

        remote_multiaddr = str(peer.metadata.get("multiaddr") or (peer.addresses[0] if peer.addresses else ""))
        if not remote_multiaddr and not peer.peer_id:
            raise RuntimeError("peer must provide a peer_id or multiaddr")

        args = {
            "model": request.model,
            "input_data": request.inputs,
            "task": request.task,
        }
        args.update(request.parameters)

        payload = await p2p_taskqueue_call_tool(
            tool_name=str(self.config.get("remote_inference_tool") or self.REMOTE_INFERENCE_TOOL),
            args=args,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=peer.peer_id,
            timeout_s=self.request_timeout,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("remote MCP++ inference returned a non-object payload")
        if payload.get("ok") is False or payload.get("success") is False or payload.get("status") == "error":
            raise RuntimeError(str(payload.get("error") or "remote MCP++ inference failed"))
        if "result" in payload:
            return payload["result"]
        if "output" in payload:
            return payload["output"]
        return payload

    async def _refresh_peer_registry(self) -> None:
        """Merge MCP++ discovered peers into the local registry."""
        try:
            from ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools import list_peers

            payload = await list_peers(
                include_capabilities=True,
                discover=bool(self.config.get("discover", True)),
                discovery_timeout_s=float(self.config.get("discovery_timeout_s", 1.5)),
                limit=int(self.config.get("peer_limit", 50)),
            )
        except Exception as exc:
            logger.debug("MCP++ peer discovery unavailable: %s", exc)
            return

        if not isinstance(payload, dict) or payload.get("ok") is False:
            return

        discovered: List[PeerInfo] = []
        for row in payload.get("peers", []) or []:
            if isinstance(row, dict):
                peer = _peer_from_record(row)
                if peer is not None:
                    discovered.append(peer)

        if not discovered:
            return

        async with self._peers_lock:
            for peer in discovered:
                existing = self.peers.get(peer.peer_id)
                if existing is None:
                    self.peers[peer.peer_id] = peer
                    continue
                existing.addresses = peer.addresses or existing.addresses
                existing.capabilities = peer.capabilities or existing.capabilities
                existing.models = peer.models or existing.models
                existing.metadata.update(peer.metadata)
                existing.last_seen = time.time()

    async def _record_peer_failure(self, peer_id: str) -> None:
        async with self._peers_lock:
            if peer_id in self.peers:
                self.peers[peer_id].failed_requests += 1
                self.peers[peer_id].total_requests += 1

    async def _discovery_loop(self) -> None:
        while self.is_running:
            try:
                await self._refresh_peer_registry()
            except Exception as exc:
                logger.debug("MCP++ discovery loop failed: %s", exc)
            await asyncio.sleep(max(1.0, self.discovery_interval))

    async def _heartbeat_loop(self) -> None:
        while self.is_running:
            await asyncio.sleep(30.0)


_global_p2p_node: Optional[LibP2PInferenceNode] = None


async def get_p2p_node(config: Optional[Dict[str, Any]] = None) -> LibP2PInferenceNode:
    """Get the global MCP++ p2p inference node instance."""
    global _global_p2p_node
    if _global_p2p_node is None:
        _global_p2p_node = LibP2PInferenceNode(config)
        await _global_p2p_node.start()
    return _global_p2p_node


async def shutdown_p2p_node() -> None:
    """Shutdown the global MCP++ p2p inference node."""
    global _global_p2p_node
    if _global_p2p_node:
        await _global_p2p_node.stop()
        _global_p2p_node = None


__all__ = [
    "HAVE_LIBP2P",
    "HAVE_MCPPLUSPLUS_P2P",
    "PeerCapability",
    "PeerInfo",
    "InferenceRequest",
    "InferenceResponse",
    "LibP2PInferenceNode",
    "get_p2p_node",
    "shutdown_p2p_node",
]
