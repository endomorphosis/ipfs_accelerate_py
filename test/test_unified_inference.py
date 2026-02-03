"""
Integration tests for unified inference backend system

Tests the core functionality of:
- Backend registration and discovery
- Health monitoring
- Load balancing
- WebSocket communication (mocked)
- Backend manager integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestBackendManager:
    """Test the inference backend manager"""
    
    def test_backend_manager_initialization(self):
        """Test that backend manager initializes correctly"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )
        
        manager = InferenceBackendManager({
            'enable_health_checks': True,
            'health_check_interval': 30,
            'load_balancing': 'round_robin'
        })
        
        assert manager is not None
        assert manager.backends == {}
        assert manager.health_check_enabled is True
    
    def test_backend_registration(self):
        """Test backend registration"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )
        
        manager = InferenceBackendManager()
        
        # Create mock backend instance
        mock_backend = Mock()
        
        # Register backend
        success = manager.register_backend(
            backend_id="test_gpu_backend",
            backend_type=BackendType.GPU,
            name="Test GPU Backend",
            instance=mock_backend,
            capabilities=BackendCapabilities(
                supported_tasks={"text-generation"},
                supports_streaming=True,
                hardware_types={"cuda"}
            ),
            endpoint="http://localhost:8000"
        )
        
        assert success is True
        assert "test_gpu_backend" in manager.backends
        assert manager.backends["test_gpu_backend"].name == "Test GPU Backend"
    
    def test_backend_listing(self):
        """Test listing backends with filters"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )
        
        manager = InferenceBackendManager()
        
        # Register multiple backends
        for i, backend_type in enumerate([BackendType.GPU, BackendType.API, BackendType.CLI]):
            manager.register_backend(
                backend_id=f"backend_{i}",
                backend_type=backend_type,
                name=f"Backend {i}",
                instance=Mock(),
                capabilities=BackendCapabilities(
                    supported_tasks={"text-generation"}
                )
            )
        
        # List all backends
        all_backends = manager.list_backends()
        assert len(all_backends) == 3
        
        # Filter by type
        gpu_backends = manager.list_backends(backend_type=BackendType.GPU)
        assert len(gpu_backends) == 1
        assert gpu_backends[0].backend_type == BackendType.GPU
        
        # Filter by task
        text_gen_backends = manager.list_backends(task="text-generation")
        assert len(text_gen_backends) == 3
    
    def test_backend_selection_round_robin(self):
        """Test round-robin backend selection"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities, BackendStatus
        )
        
        manager = InferenceBackendManager({
            'load_balancing': 'round_robin'
        })
        
        # Register multiple backends for same task
        for i in range(3):
            manager.register_backend(
                backend_id=f"backend_{i}",
                backend_type=BackendType.API,
                name=f"Backend {i}",
                instance=Mock(),
                capabilities=BackendCapabilities(
                    supported_tasks={"text-generation"}
                )
            )
            # Mark as healthy
            manager.backends[f"backend_{i}"].status = BackendStatus.HEALTHY
        
        # Select backends multiple times
        selections = []
        for _ in range(6):
            backend = manager.select_backend_for_task("text-generation")
            if backend:
                selections.append(backend.backend_id)
        
        # Should cycle through backends
        assert len(selections) == 6
        assert selections[0] != selections[1] or len(set(selections[:3])) > 1
    
    def test_metrics_recording(self):
        """Test recording request metrics"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType
        )
        
        manager = InferenceBackendManager()
        
        manager.register_backend(
            backend_id="test_backend",
            backend_type=BackendType.API,
            name="Test Backend",
            instance=Mock()
        )
        
        # Record some requests
        manager.record_request("test_backend", success=True, latency_ms=100.0)
        manager.record_request("test_backend", success=True, latency_ms=200.0)
        manager.record_request("test_backend", success=False, latency_ms=50.0)
        
        backend = manager.get_backend("test_backend")
        assert backend.metrics.total_requests == 3
        assert backend.metrics.successful_requests == 2
        assert backend.metrics.failed_requests == 1
        assert backend.metrics.average_latency_ms > 0
    
    def test_status_report(self):
        """Test comprehensive status reporting"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )
        
        manager = InferenceBackendManager()
        
        # Register backends
        manager.register_backend(
            backend_id="backend_1",
            backend_type=BackendType.GPU,
            name="GPU Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(
                supported_tasks={"text-generation", "text-embedding"}
            )
        )
        
        # Get status report
        status = manager.get_backend_status_report()
        
        assert "total_backends" in status
        assert status["total_backends"] == 1
        assert "backends_by_type" in status
        assert "backends" in status
        assert len(status["backends"]) == 1


class TestWebSocketHandler:
    """Test WebSocket handler functionality"""
    
    def test_connection_manager_initialization(self):
        """Test connection manager initialization"""
        try:
            from ipfs_accelerate_py.hf_model_server.websocket_handler import ConnectionManager
        except ImportError:
            pytest.skip("WebSocket handler requires fastapi - optional dependency")
            return
        
        manager = ConnectionManager()
        
        assert manager.active_connections == {}
        assert "inference" in manager.subscriptions
        assert "status" in manager.subscriptions
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test connection/disconnection lifecycle"""
        try:
            from ipfs_accelerate_py.hf_model_server.websocket_handler import ConnectionManager
        except ImportError:
            pytest.skip("WebSocket handler requires fastapi - optional dependency")
            return
        
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        
        # Connect
        await manager.connect("client_1", mock_websocket)
        assert "client_1" in manager.active_connections
        
        # Subscribe
        manager.subscribe("client_1", "inference")
        assert "client_1" in manager.subscriptions["inference"]
        
        # Disconnect
        manager.disconnect("client_1")
        assert "client_1" not in manager.active_connections
        assert "client_1" not in manager.subscriptions["inference"]


class TestLibP2PInference:
    """Test libp2p inference functionality"""
    
    def test_peer_info_creation(self):
        """Test PeerInfo dataclass"""
        from ipfs_accelerate_py.libp2p_inference import PeerInfo, PeerCapability
        
        peer = PeerInfo(
            peer_id="QmTest123",
            capabilities={PeerCapability.TEXT_GENERATION},
            models={"gpt2", "bert"}
        )
        
        assert peer.peer_id == "QmTest123"
        assert PeerCapability.TEXT_GENERATION in peer.capabilities
        assert "gpt2" in peer.models
    
    def test_inference_request_creation(self):
        """Test InferenceRequest dataclass"""
        from ipfs_accelerate_py.libp2p_inference import InferenceRequest
        
        request = InferenceRequest(
            request_id="req_123",
            task="text-generation",
            model="gpt2",
            inputs="Hello, world!",
            parameters={"max_length": 50}
        )
        
        assert request.request_id == "req_123"
        assert request.task == "text-generation"
        assert request.model == "gpt2"


class TestUnifiedInferenceService:
    """Test unified inference service"""
    
    def test_service_config_creation(self):
        """Test service configuration"""
        from ipfs_accelerate_py.unified_inference_service import InferenceServiceConfig
        
        config = InferenceServiceConfig(
            enable_backend_manager=True,
            enable_hf_server=True,
            enable_websocket=True,
            enable_libp2p=False
        )
        
        assert config.enable_backend_manager is True
        assert config.enable_hf_server is True
        assert config.enable_websocket is True
        assert config.enable_libp2p is False
    
    def test_service_initialization(self):
        """Test service initialization"""
        from ipfs_accelerate_py.unified_inference_service import (
            UnifiedInferenceService, InferenceServiceConfig
        )
        
        config = InferenceServiceConfig(
            enable_backend_manager=True,
            enable_hf_server=False,  # Don't start server in test
            enable_websocket=False,
            enable_libp2p=False
        )
        
        service = UnifiedInferenceService(config)
        
        assert service is not None
        assert service.config.enable_backend_manager is True


class TestMCPTools:
    """Test MCP tool integration"""
    
    def test_mcp_tools_import(self):
        """Test that MCP tools can be imported"""
        try:
            from ipfs_accelerate_py.mcp.tools.backend_management import (
                list_inference_backends,
                get_backend_status,
                select_backend_for_inference
            )
            assert callable(list_inference_backends)
            assert callable(get_backend_status)
            assert callable(select_backend_for_inference)
        except ImportError as e:
            pytest.skip(f"MCP tools not available: {e}")
    
    def test_list_backends_no_manager(self):
        """Test listing backends when manager not available"""
        from ipfs_accelerate_py.mcp.tools.backend_management import list_inference_backends
        
        # Should handle case where manager doesn't exist yet
        result = list_inference_backends()
        assert isinstance(result, dict)


def test_all_modules_importable():
    """Test that all modules can be imported"""
    modules = [
        "ipfs_accelerate_py.inference_backend_manager",
        "ipfs_accelerate_py.libp2p_inference",
        "ipfs_accelerate_py.unified_inference_service",
        "ipfs_accelerate_py.mcp.tools.backend_management"
    ]
    
    # Optional modules that may require additional dependencies
    optional_modules = [
        "ipfs_accelerate_py.hf_model_server.websocket_handler",  # Requires fastapi
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
    
    # Try optional modules but don't fail if they're not available
    for module_name in optional_modules:
        try:
            __import__(module_name)
        except ImportError:
            pass  # Optional modules are allowed to fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
