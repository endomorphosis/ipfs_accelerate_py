"""
Unified Inference Service Integration

This module integrates all inference components:
- Backend manager for routing and load balancing
- HF model server for local inference
- WebSocket for real-time communication
- libp2p for distributed P2P inference
- MCP server for tool integration

Provides a single entry point for configuring and starting all inference services.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import backend manager
try:
    from .inference_backend_manager import (
        InferenceBackendManager,
        BackendType,
        BackendCapabilities,
        get_backend_manager
    )
    HAVE_BACKEND_MANAGER = True
except ImportError:
    logger.warning("Backend manager not available")
    HAVE_BACKEND_MANAGER = False
    BackendType = None
    BackendCapabilities = None

# Import libp2p
try:
    from .libp2p_inference import LibP2PInferenceNode, PeerCapability
    HAVE_LIBP2P = True
except ImportError:
    logger.warning("libp2p inference not available")
    HAVE_LIBP2P = False
    LibP2PInferenceNode = None
    PeerCapability = None

# Import WebSocket
try:
    from .hf_model_server.websocket_handler import get_connection_manager
    HAVE_WEBSOCKET = True
except ImportError:
    logger.warning("WebSocket handler not available")
    HAVE_WEBSOCKET = False

# Import HF server
try:
    from .hf_model_server.server import HFModelServer
    from .hf_model_server.config import ServerConfig
    HAVE_HF_SERVER = True
except ImportError:
    logger.warning("HF model server not available")
    HAVE_HF_SERVER = False
    HFModelServer = None
    ServerConfig = None


@dataclass
class InferenceServiceConfig:
    """Configuration for the unified inference service"""
    
    # Backend manager configuration
    enable_backend_manager: bool = True
    backend_health_checks: bool = True
    backend_health_check_interval: int = 60
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, best_performance
    
    # HF model server configuration
    enable_hf_server: bool = True
    hf_server_host: str = "0.0.0.0"
    hf_server_port: int = 8000
    hf_auto_discover_skills: bool = True
    hf_enable_hardware_detection: bool = True
    
    # WebSocket configuration
    enable_websocket: bool = True
    websocket_path: str = "/ws"
    
    # libp2p configuration
    enable_libp2p: bool = True
    libp2p_bootstrap_peers: List[str] = field(default_factory=list)
    libp2p_enable_mdns: bool = True
    libp2p_discovery_interval: int = 60
    
    # API backend configuration
    enable_api_backends: bool = True
    api_backends: List[str] = field(default_factory=lambda: [
        "hf_tgi", "hf_tei", "ollama", "openai_api"
    ])
    
    # CLI backend configuration
    enable_cli_backends: bool = True
    cli_backends: List[str] = field(default_factory=lambda: [
        "claude_cli", "openai_cli"
    ])
    
    # General settings
    log_level: str = "INFO"


class UnifiedInferenceService:
    """
    Unified inference service that coordinates all components
    
    This class:
    - Initializes and manages the backend manager
    - Starts the HF model server
    - Enables WebSocket communication
    - Connects to the P2P network
    - Registers all available backends
    """
    
    def __init__(self, config: Optional[InferenceServiceConfig] = None):
        self.config = config or InferenceServiceConfig()
        
        # Component instances
        self.backend_manager: Optional[InferenceBackendManager] = None
        self.hf_server: Optional[HFModelServer] = None
        self.p2p_node: Optional[LibP2PInferenceNode] = None
        self.connection_manager = None
        
        # Running state
        self.is_running = False
        self._server_task: Optional[asyncio.Task] = None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def start(self):
        """Start all inference services"""
        logger.info("Starting Unified Inference Service...")
        
        # Initialize backend manager
        if self.config.enable_backend_manager and HAVE_BACKEND_MANAGER:
            logger.info("Initializing backend manager...")
            self.backend_manager = get_backend_manager({
                'enable_health_checks': self.config.backend_health_checks,
                'health_check_interval': self.config.backend_health_check_interval,
                'load_balancing': self.config.load_balancing_strategy
            })
        
        # Start libp2p node
        if self.config.enable_libp2p and HAVE_LIBP2P:
            logger.info("Starting libp2p node...")
            self.p2p_node = LibP2PInferenceNode({
                'bootstrap_peers': self.config.libp2p_bootstrap_peers,
                'enable_mdns': self.config.libp2p_enable_mdns,
                'discovery_interval': self.config.libp2p_discovery_interval,
                'start_time': asyncio.get_event_loop().time()
            })
            await self.p2p_node.start()
            
            # Register P2P node as a backend
            if self.backend_manager:
                self.backend_manager.register_backend(
                    backend_id="libp2p_node",
                    backend_type=BackendType.P2P,
                    name="libp2p Distributed Network",
                    instance=self.p2p_node,
                    capabilities=BackendCapabilities(
                        supported_tasks={"text-generation", "text-embedding", "image-generation"},
                        supports_streaming=False,
                        supports_batching=False,
                        protocols={"libp2p"}
                    )
                )
        
        # Register API backends
        if self.config.enable_api_backends and self.backend_manager:
            await self._register_api_backends()
        
        # Register CLI backends
        if self.config.enable_cli_backends and self.backend_manager:
            await self._register_cli_backends()
        
        # Start HF model server
        if self.config.enable_hf_server and HAVE_HF_SERVER:
            logger.info("Starting HF model server...")
            
            server_config = ServerConfig()
            server_config.host = self.config.hf_server_host
            server_config.port = self.config.hf_server_port
            server_config.auto_discover = self.config.hf_auto_discover_skills
            server_config.enable_hardware_detection = self.config.hf_enable_hardware_detection
            
            self.hf_server = HFModelServer(server_config)
            
            # Register HF server as a backend
            if self.backend_manager:
                self.backend_manager.register_backend(
                    backend_id="hf_server_local",
                    backend_type=BackendType.GPU,
                    name="Local HuggingFace Server",
                    instance=self.hf_server,
                    endpoint=f"http://{self.config.hf_server_host}:{self.config.hf_server_port}",
                    capabilities=BackendCapabilities(
                        supported_tasks={"text-generation", "text-embedding", "image-classification"},
                        supports_streaming=True,
                        supports_batching=True,
                        hardware_types={"cuda", "cpu", "mps"},
                        protocols={"http", "websocket"}
                    )
                )
        
        # Setup WebSocket
        if self.config.enable_websocket and HAVE_WEBSOCKET:
            logger.info("WebSocket endpoint available at /ws/{client_id}")
            self.connection_manager = get_connection_manager()
        
        # Start health monitoring
        if self.backend_manager and self.config.backend_health_checks:
            self.backend_manager.start_health_monitoring()
        
        self.is_running = True
        logger.info("Unified Inference Service started successfully")
        
        # Print status
        self.print_status()
    
    async def stop(self):
        """Stop all inference services"""
        logger.info("Stopping Unified Inference Service...")
        
        self.is_running = False
        
        # Stop health monitoring
        if self.backend_manager:
            self.backend_manager.stop_health_monitoring()
        
        # Stop libp2p node
        if self.p2p_node:
            await self.p2p_node.stop()
        
        # Stop server task
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Unified Inference Service stopped")
    
    async def _register_api_backends(self):
        """Register available API backends"""
        logger.info("Registering API backends...")
        
        # Try to import and register each API backend
        for backend_name in self.config.api_backends:
            try:
                # Dynamically import backend
                module = __import__(
                    f'ipfs_accelerate_py.api_backends.{backend_name}',
                    fromlist=[backend_name]
                )
                backend_class = getattr(module, backend_name)
                
                # Create instance
                backend_instance = backend_class()
                
                # Determine capabilities based on backend type
                capabilities = self._get_backend_capabilities(backend_name)
                
                # Register with backend manager
                self.backend_manager.register_backend(
                    backend_id=f"api_{backend_name}",
                    backend_type=BackendType.API,
                    name=f"{backend_name.upper()} API",
                    instance=backend_instance,
                    capabilities=capabilities,
                    metadata={"backend_type": "api", "backend_name": backend_name}
                )
                
                logger.info(f"Registered API backend: {backend_name}")
            
            except Exception as e:
                logger.warning(f"Could not register API backend {backend_name}: {e}")
    
    async def _register_cli_backends(self):
        """Register available CLI backends"""
        logger.info("Registering CLI backends...")
        
        cli_backend_info = {
            "claude_cli": {
                "name": "Claude CLI",
                "tasks": {"text-generation", "code-generation"}
            },
            "openai_cli": {
                "name": "OpenAI CLI",
                "tasks": {"text-generation", "code-generation"}
            }
        }
        
        for backend_name in self.config.cli_backends:
            if backend_name not in cli_backend_info:
                continue
            
            try:
                info = cli_backend_info[backend_name]
                
                # Register with backend manager (without actual instance for now)
                self.backend_manager.register_backend(
                    backend_id=f"cli_{backend_name}",
                    backend_type=BackendType.CLI,
                    name=info["name"],
                    instance=None,  # CLI backends are invoked differently
                    capabilities=BackendCapabilities(
                        supported_tasks=info["tasks"],
                        supports_streaming=False,
                        supports_batching=False,
                        protocols={"cli"}
                    ),
                    metadata={"backend_type": "cli", "backend_name": backend_name}
                )
                
                logger.info(f"Registered CLI backend: {backend_name}")
            
            except Exception as e:
                logger.warning(f"Could not register CLI backend {backend_name}: {e}")
    
    def _get_backend_capabilities(self, backend_name: str) -> BackendCapabilities:
        """Determine capabilities for a backend"""
        
        # Default capabilities
        capabilities = BackendCapabilities(
            supported_tasks={"text-generation"},
            supports_streaming=False,
            supports_batching=False,
            protocols={"http"}
        )
        
        # Backend-specific capabilities
        if backend_name in ["hf_tgi", "ollama", "vllm"]:
            capabilities.supported_tasks = {"text-generation"}
            capabilities.supports_streaming = True
            capabilities.supports_batching = True
        
        elif backend_name in ["hf_tei"]:
            capabilities.supported_tasks = {"text-embedding"}
            capabilities.supports_batching = True
        
        elif backend_name in ["openai_api"]:
            capabilities.supported_tasks = {
                "text-generation", "text-embedding", "image-generation"
            }
            capabilities.supports_streaming = True
        
        return capabilities
    
    def print_status(self):
        """Print status of all components"""
        print("\n" + "="*70)
        print("UNIFIED INFERENCE SERVICE STATUS")
        print("="*70)
        
        if self.backend_manager:
            status = self.backend_manager.get_backend_status_report()
            print(f"\n📊 Backend Manager:")
            print(f"  Total Backends: {status['total_backends']}")
            print(f"  Backends by Type: {status['backends_by_type']}")
            print(f"  Supported Tasks: {', '.join(status['supported_tasks'])}")
            
            print(f"\n📋 Registered Backends:")
            for backend in status['backends']:
                health = "✅" if backend['status'] == 'healthy' else "⚠️"
                print(f"  {health} {backend['name']} ({backend['type']})")
                print(f"     Tasks: {', '.join(backend['tasks'])}")
                print(f"     Protocols: {', '.join(backend['protocols'])}")
                if backend['endpoint']:
                    print(f"     Endpoint: {backend['endpoint']}")
        
        if self.hf_server:
            print(f"\n🚀 HF Model Server:")
            print(f"  Address: http://{self.config.hf_server_host}:{self.config.hf_server_port}")
            print(f"  OpenAI API: /v1/completions, /v1/chat/completions, /v1/embeddings")
        
        if self.config.enable_websocket:
            print(f"\n🔌 WebSocket:")
            print(f"  Endpoint: ws://{self.config.hf_server_host}:{self.config.hf_server_port}/ws/{{client_id}}")
        
        if self.p2p_node:
            print(f"\n🌐 libp2p Node:")
            print(f"  Peer ID: {self.p2p_node.peer_id}")
            print(f"  Capabilities: {', '.join(c.value for c in self.p2p_node.local_capabilities)}")
        
        print("\n" + "="*70 + "\n")
    
    def get_backend_manager(self) -> Optional[InferenceBackendManager]:
        """Get the backend manager instance"""
        return self.backend_manager
    
    def get_hf_server(self):
        """Get the HF server instance"""
        return self.hf_server
    
    def get_p2p_node(self):
        """Get the P2P node instance"""
        return self.p2p_node


async def start_unified_service(config: Optional[InferenceServiceConfig] = None):
    """
    Start the unified inference service
    
    This is a convenience function that creates and starts the service.
    
    Args:
        config: Optional configuration
        
    Returns:
        The running service instance
    """
    service = UnifiedInferenceService(config)
    await service.start()
    return service


# Example usage
if __name__ == "__main__":
    async def main():
        # Create configuration
        config = InferenceServiceConfig(
            enable_backend_manager=True,
            enable_hf_server=True,
            enable_websocket=True,
            enable_libp2p=False,  # Disable by default for testing
            enable_api_backends=True,
            enable_cli_backends=True
        )
        
        # Start service
        service = await start_unified_service(config)
        
        try:
            # Run until interrupted
            if service.hf_server:
                # Run the FastAPI server
                import uvicorn
                uvicorn.run(
                    service.hf_server.app,
                    host=config.hf_server_host,
                    port=config.hf_server_port
                )
            else:
                # Just keep running
                while service.is_running:
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await service.stop()
    
    asyncio.run(main())
