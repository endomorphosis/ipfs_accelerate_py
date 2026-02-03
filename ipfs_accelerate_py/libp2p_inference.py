"""
libp2p Integration for Distributed Inference

Provides peer-to-peer inference capabilities using libp2p:
- Distributed model serving across P2P network
- Peer discovery and connection management
- Request routing to available peers
- Load balancing across the network
- Fault tolerance and fallback mechanisms
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import libp2p components
try:
    from libp2p import new_host
    from libp2p.peer.peerinfo import info_from_p2p_addr
    from libp2p.typing import TProtocol
    HAVE_LIBP2P = True
except ImportError:
    logger.warning("libp2p not available - P2P inference features disabled")
    HAVE_LIBP2P = False
    new_host = None
    info_from_p2p_addr = None
    TProtocol = str

# Try to import compatibility layer
try:
    from ..github_cli.libp2p_compat import patch_libp2p_compatibility
    patch_libp2p_compatibility()
except ImportError:
    logger.debug("libp2p compatibility layer not available")


class PeerCapability(Enum):
    """Capabilities that a peer can offer"""
    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"
    IMAGE_GENERATION = "image-generation"
    IMAGE_CLASSIFICATION = "image-classification"
    AUDIO_TRANSCRIPTION = "audio-transcription"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


@dataclass
class PeerInfo:
    """Information about a P2P peer"""
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
    """P2P inference request"""
    request_id: str
    task: str
    model: str
    inputs: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """P2P inference response"""
    request_id: str
    result: Any
    peer_id: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class LibP2PInferenceNode:
    """
    libp2p node for distributed inference
    
    Provides:
    - Peer discovery and registration
    - Request routing and load balancing
    - Fault tolerance and retries
    - Performance monitoring
    """
    
    # Protocol identifiers
    DISCOVERY_PROTOCOL = "/ipfs-accelerate/discovery/1.0.0"
    INFERENCE_PROTOCOL = "/ipfs-accelerate/inference/1.0.0"
    STATUS_PROTOCOL = "/ipfs-accelerate/status/1.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # libp2p host
        self.host = None
        self.peer_id = None
        
        # Peer registry
        self.peers: Dict[str, PeerInfo] = {}
        self._peers_lock = asyncio.Lock()
        
        # Bootstrap peers
        self.bootstrap_peers = self.config.get('bootstrap_peers', [])
        
        # Local capabilities
        self.local_capabilities: Set[PeerCapability] = set()
        self.local_models: Set[str] = set()
        
        # Request tracking
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.request_timeout = self.config.get('request_timeout', 30.0)
        
        # Discovery configuration
        self.enable_mdns = self.config.get('enable_mdns', True)
        self.discovery_interval = self.config.get('discovery_interval', 60.0)
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        self.is_running = False
    
    async def start(self):
        """Start the P2P node"""
        if not HAVE_LIBP2P:
            logger.error("Cannot start P2P node: libp2p not available")
            raise RuntimeError("libp2p not available")
        
        logger.info("Starting libp2p inference node...")
        
        try:
            # Create libp2p host
            self.host = await new_host()
            self.peer_id = self.host.get_id().pretty()
            
            logger.info(f"P2P node started with peer ID: {self.peer_id}")
            
            # Register protocol handlers
            self.host.set_stream_handler(self.DISCOVERY_PROTOCOL, self._handle_discovery)
            self.host.set_stream_handler(self.INFERENCE_PROTOCOL, self._handle_inference)
            self.host.set_stream_handler(self.STATUS_PROTOCOL, self._handle_status)
            
            # Connect to bootstrap peers
            await self._connect_bootstrap_peers()
            
            # Start background tasks
            self._discovery_task = asyncio.create_task(self._discovery_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.is_running = True
            logger.info("P2P node fully initialized")
        
        except Exception as e:
            logger.error(f"Failed to start P2P node: {e}")
            raise
    
    async def stop(self):
        """Stop the P2P node"""
        logger.info("Stopping P2P node...")
        
        self.is_running = False
        
        # Cancel background tasks
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close host
        if self.host:
            await self.host.close()
        
        logger.info("P2P node stopped")
    
    def register_capability(self, capability: PeerCapability):
        """Register a capability this node can provide"""
        self.local_capabilities.add(capability)
        logger.info(f"Registered capability: {capability.value}")
    
    def register_model(self, model_id: str):
        """Register a model available on this node"""
        self.local_models.add(model_id)
        logger.info(f"Registered model: {model_id}")
    
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover available peers in the network"""
        async with self._peers_lock:
            return list(self.peers.values())
    
    async def find_peers_for_task(
        self,
        task: str,
        model: Optional[str] = None,
        max_peers: int = 5
    ) -> List[PeerInfo]:
        """
        Find suitable peers for a given task
        
        Args:
            task: The inference task type
            model: Optional specific model requirement
            max_peers: Maximum number of peers to return
            
        Returns:
            List of suitable peers, sorted by suitability
        """
        try:
            capability = PeerCapability(task)
        except ValueError:
            logger.warning(f"Unknown task type: {task}")
            return []
        
        async with self._peers_lock:
            # Filter peers with required capability
            suitable_peers = [
                p for p in self.peers.values()
                if capability in p.capabilities
            ]
            
            # Filter by model if specified
            if model:
                suitable_peers = [
                    p for p in suitable_peers
                    if not p.models or model in p.models
                ]
            
            # Sort by load (prefer peers with fewer active requests)
            suitable_peers.sort(key=lambda p: (p.active_requests, -p.successful_requests))
            
            return suitable_peers[:max_peers]
    
    async def submit_inference_request(
        self,
        task: str,
        model: str,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> InferenceResponse:
        """
        Submit an inference request to the P2P network
        
        Args:
            task: Inference task type
            model: Model to use
            inputs: Input data
            parameters: Optional inference parameters
            timeout: Request timeout in seconds
            
        Returns:
            InferenceResponse with the result
        """
        request_id = f"{self.peer_id}_{int(time.time() * 1000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            task=task,
            model=model,
            inputs=inputs,
            parameters=parameters or {}
        )
        
        # Find suitable peers
        peers = await self.find_peers_for_task(task, model)
        
        if not peers:
            return InferenceResponse(
                request_id=request_id,
                result=None,
                peer_id="",
                latency_ms=0.0,
                success=False,
                error=f"No peers available for task: {task}"
            )
        
        # Try peers in order until one succeeds
        timeout = timeout or self.request_timeout
        
        for peer in peers:
            try:
                logger.info(f"Sending request {request_id} to peer {peer.peer_id}")
                
                start_time = time.time()
                result = await asyncio.wait_for(
                    self._send_inference_request(peer, request),
                    timeout=timeout
                )
                latency_ms = (time.time() - start_time) * 1000
                
                # Update peer stats
                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        self.peers[peer.peer_id].successful_requests += 1
                        self.peers[peer.peer_id].total_requests += 1
                        self.peers[peer.peer_id].latency_ms = latency_ms
                
                return InferenceResponse(
                    request_id=request_id,
                    result=result,
                    peer_id=peer.peer_id,
                    latency_ms=latency_ms,
                    success=True
                )
            
            except asyncio.TimeoutError:
                logger.warning(f"Request to peer {peer.peer_id} timed out")
                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        self.peers[peer.peer_id].failed_requests += 1
                        self.peers[peer.peer_id].total_requests += 1
                continue
            
            except Exception as e:
                logger.error(f"Error sending request to peer {peer.peer_id}: {e}")
                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        self.peers[peer.peer_id].failed_requests += 1
                        self.peers[peer.peer_id].total_requests += 1
                continue
        
        # All peers failed
        return InferenceResponse(
            request_id=request_id,
            result=None,
            peer_id="",
            latency_ms=0.0,
            success=False,
            error="All available peers failed to process request"
        )
    
    async def _send_inference_request(
        self,
        peer: PeerInfo,
        request: InferenceRequest
    ) -> Any:
        """Send an inference request to a specific peer"""
        # TODO: Implement actual libp2p stream communication
        # For now, this is a placeholder
        raise NotImplementedError("Actual libp2p communication not yet implemented")
    
    async def _connect_bootstrap_peers(self):
        """Connect to bootstrap peers"""
        for peer_addr in self.bootstrap_peers:
            try:
                logger.info(f"Connecting to bootstrap peer: {peer_addr}")
                peer_info = info_from_p2p_addr(peer_addr)
                await self.host.connect(peer_info)
                logger.info(f"Connected to bootstrap peer: {peer_info.peer_id.pretty()}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer_addr}: {e}")
    
    async def _discovery_loop(self):
        """Periodic peer discovery"""
        while self.is_running:
            try:
                await self._run_discovery()
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
            
            await asyncio.sleep(self.discovery_interval)
    
    async def _run_discovery(self):
        """Run peer discovery"""
        # TODO: Implement actual peer discovery
        # This would involve:
        # 1. Broadcasting discovery messages
        # 2. Listening for peer announcements
        # 3. Updating peer registry
        pass
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat to maintain connections"""
        while self.is_running:
            try:
                await self._send_heartbeats()
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
            
            await asyncio.sleep(30.0)  # Heartbeat every 30 seconds
    
    async def _send_heartbeats(self):
        """Send heartbeats to known peers"""
        # TODO: Implement heartbeat mechanism
        pass
    
    async def _handle_discovery(self, stream):
        """Handle incoming discovery protocol messages"""
        try:
            # Read discovery message
            data = await stream.read()
            message = json.loads(data.decode())
            
            # Extract peer information
            peer_id = message.get('peer_id')
            capabilities = [PeerCapability(c) for c in message.get('capabilities', [])]
            models = set(message.get('models', []))
            
            # Update peer registry
            async with self._peers_lock:
                if peer_id not in self.peers:
                    self.peers[peer_id] = PeerInfo(
                        peer_id=peer_id,
                        capabilities=set(capabilities),
                        models=models
                    )
                    logger.info(f"Discovered new peer: {peer_id}")
                else:
                    self.peers[peer_id].capabilities = set(capabilities)
                    self.peers[peer_id].models = models
                    self.peers[peer_id].last_seen = time.time()
            
            # Send response with our capabilities
            response = {
                'peer_id': self.peer_id,
                'capabilities': [c.value for c in self.local_capabilities],
                'models': list(self.local_models)
            }
            await stream.write(json.dumps(response).encode())
        
        except Exception as e:
            logger.error(f"Error handling discovery message: {e}")
        finally:
            await stream.close()
    
    async def _handle_inference(self, stream):
        """Handle incoming inference protocol messages"""
        try:
            # Read inference request
            data = await stream.read()
            message = json.loads(data.decode())
            
            # Extract request details
            request_id = message.get('request_id')
            task = message.get('task')
            model = message.get('model')
            inputs = message.get('inputs')
            parameters = message.get('parameters', {})
            
            logger.info(f"Received inference request {request_id} for task {task}")
            
            # TODO: Execute inference locally
            # For now, send error response
            response = {
                'request_id': request_id,
                'success': False,
                'error': 'Local inference not yet implemented'
            }
            
            await stream.write(json.dumps(response).encode())
        
        except Exception as e:
            logger.error(f"Error handling inference request: {e}")
            response = {
                'success': False,
                'error': str(e)
            }
            await stream.write(json.dumps(response).encode())
        finally:
            await stream.close()
    
    async def _handle_status(self, stream):
        """Handle incoming status protocol messages"""
        try:
            # Send status information
            status = {
                'peer_id': self.peer_id,
                'capabilities': [c.value for c in self.local_capabilities],
                'models': list(self.local_models),
                'active_requests': len(self.pending_requests),
                'uptime': time.time() - (self.config.get('start_time', time.time()))
            }
            
            await stream.write(json.dumps(status).encode())
        
        except Exception as e:
            logger.error(f"Error handling status request: {e}")
        finally:
            await stream.close()


# Global P2P node instance
_global_p2p_node: Optional[LibP2PInferenceNode] = None


async def get_p2p_node(config: Optional[Dict[str, Any]] = None) -> LibP2PInferenceNode:
    """Get the global P2P node instance"""
    global _global_p2p_node
    if _global_p2p_node is None:
        _global_p2p_node = LibP2PInferenceNode(config)
        await _global_p2p_node.start()
    return _global_p2p_node


async def shutdown_p2p_node():
    """Shutdown the global P2P node"""
    global _global_p2p_node
    if _global_p2p_node:
        await _global_p2p_node.stop()
        _global_p2p_node = None
