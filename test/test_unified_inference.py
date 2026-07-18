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
            'load_balancing': 'round_robin',
            'persist_registry': False,
        })
        
        assert manager is not None
        assert manager.backends == {}
        assert manager.health_check_enabled is True
    
    def test_backend_registration(self):
        """Test backend registration"""
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )
        
        manager = InferenceBackendManager({'persist_registry': False})
        
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
        
        manager = InferenceBackendManager({'persist_registry': False})
        
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
            'load_balancing': 'round_robin',
            'persist_registry': False,
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
        
        manager = InferenceBackendManager({'persist_registry': False})
        
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
        
        manager = InferenceBackendManager({'persist_registry': False})
        
        # Register backends
        manager.register_backend(
            backend_id="backend_1",
            backend_type=BackendType.GPU,
            name="GPU Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(
                supported_tasks={"text-generation", "text-embedding"},
                hardware_types={"cuda"},
                protocols={"http"},
            ),
            metadata={"placement_node": "node-gpu-1"},
        )
        
        # Get status report
        status = manager.get_backend_status_report()
        
        assert "total_backends" in status
        assert status["total_backends"] == 1
        assert "backends_by_type" in status
        assert "backends" in status
        assert len(status["backends"]) == 1
        assert status["backends"][0]["hardware_types"] == ["cuda"]
        assert status["backends"][0]["protocols"] == ["http"]
        assert status["backends"][0]["placement_node"] == "node-gpu-1"

    def test_backend_registry_state_persists_and_reloads(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities, BackendStatus
        )

        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        state_path = Path(temp_dir) / "backend_registry.json"

        manager = InferenceBackendManager({"registry_state_path": str(state_path)})
        manager.register_backend(
            backend_id="persistent_backend",
            backend_type=BackendType.API,
            name="Persistent Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(
                supported_tasks={"text-generation"},
                protocols={"http"},
                hardware_types={"cpu"},
            ),
            endpoint="http://localhost:9100",
            metadata={"region": "local"},
        )
        manager._update_backend_status("persistent_backend", BackendStatus.DEGRADED)
        selected = manager.select_backend_for_task("text-generation")
        assert selected is None

        # Mark healthy again and confirm the state file captures selection metadata.
        manager._update_backend_status("persistent_backend", BackendStatus.HEALTHY)
        selected = manager.select_backend_for_task("text-generation")
        assert selected is not None
        assert selected.backend_id == "persistent_backend"
        assert selected.selection_count == 1
        assert selected.last_selection_reason is not None
        assert "supports_task:text-generation" in selected.last_selection_reason

        manager2 = InferenceBackendManager({"registry_state_path": str(state_path)})
        reloaded = manager2.get_backend("persistent_backend")
        assert reloaded is not None
        assert reloaded.endpoint == "http://localhost:9100"
        assert reloaded.status == BackendStatus.HEALTHY
        assert reloaded.capabilities.protocols == {"http"}
        assert reloaded.capabilities.supported_tasks == {"text-generation"}
        assert reloaded.selection_count == 1
        assert reloaded.last_selected_task == "text-generation"
        assert reloaded.last_selection_reason == selected.last_selection_reason
        assert reloaded.metadata["region"] == "local"

        report = manager2.get_backend_status_report()
        backend_entry = next(item for item in report["backends"] if item["id"] == "persistent_backend")
        assert backend_entry["last_selection_reason"] == selected.last_selection_reason

    def test_prune_stale_backends_removes_only_stale_inactive_entries(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities, BackendStatus
        )

        manager = InferenceBackendManager({"persist_registry": False})
        healthy = manager.register_backend(
            backend_id="healthy_backend",
            backend_type=BackendType.API,
            name="Healthy Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}),
        )
        stale = manager.register_backend(
            backend_id="stale_backend",
            backend_type=BackendType.API,
            name="Stale Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}),
        )

        assert healthy is True
        assert stale is True

        manager.backends["healthy_backend"].status = BackendStatus.HEALTHY
        manager.backends["stale_backend"].status = BackendStatus.OFFLINE
        manager.backends["healthy_backend"].last_seen = 1_000_000.0
        manager.backends["stale_backend"].last_seen = 1.0

        removed = manager.prune_stale_backends(max_age_s=60.0)

        assert removed == ["stale_backend"]
        assert "healthy_backend" in manager.backends
        assert "stale_backend" not in manager.backends
        assert manager.get_backend("healthy_backend") is not None

    def test_backend_manager_finalize_inference_result_uses_result_recorder(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )

        recorded_calls = []

        def fake_result_recorder(**kwargs):
            recorded_calls.append(kwargs)
            result = dict(kwargs["result"])
            result["output_cid"] = "cid-out"
            result["provenance_cid"] = "cid-prov"
            return result

        manager = InferenceBackendManager({"result_recorder": fake_result_recorder, "persist_registry": False})
        manager.register_backend(
            backend_id="api_backend_1",
            backend_type=BackendType.API,
            name="API Backend 1",
            instance=Mock(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}, hardware_types={"cpu"}),
            endpoint="http://localhost:9000",
            metadata={"placement_node": "node-a"},
        )

        finalized = manager.finalize_inference_result(
            backend_id="api_backend_1",
            task="text-generation",
            model="demo-model",
            inputs=["hello"],
            result={"outputs": ["ok"], "processing_time": 0.1, "device": "cpu"},
        )

        assert finalized["backend_id"] == "api_backend_1"
        assert finalized["backend_type"] == "api"
        assert finalized["endpoint"] == "http://localhost:9000"
        assert finalized["protocol"] == "http"
        assert finalized["protocols"] == ["http"]
        assert finalized["hardware_type"] == "cpu"
        assert finalized["hardware_types"] == ["cpu"]
        assert finalized["placement_node"] == "node-a"
        assert finalized["output_cid"] == "cid-out"
        assert finalized["provenance_cid"] == "cid-prov"
        assert len(recorded_calls) == 1
        assert recorded_calls[0]["backend_id"] == "api_backend_1"

    @pytest.mark.asyncio
    async def test_backend_manager_execute_task_runs_backend_and_finalizes_result(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )

        class FakeBackend:
            device = "cpu"

            def generate(self, model=None, prompt=None, **kwargs):
                return {
                    "outputs": [f"generated:{prompt}"],
                    "model": model,
                }

        def fake_result_recorder(**kwargs):
            result = dict(kwargs["result"])
            result["output_cid"] = "cid-generated"
            result["provenance_cid"] = "cid-prov"
            return result

        manager = InferenceBackendManager({"result_recorder": fake_result_recorder, "persist_registry": False})
        manager.register_backend(
            backend_id="api_backend_2",
            backend_type=BackendType.API,
            name="API Backend 2",
            instance=FakeBackend(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}, hardware_types={"cpu"}),
            endpoint="http://localhost:9001",
            metadata={"placement_node": "node-b"},
        )

        result = await manager.execute_task(
            task="text-generation",
            model="demo-model",
            inputs=["hello world"],
        )

        assert result["backend_id"] == "api_backend_2"
        assert result["backend_type"] == "api"
        assert result["protocol"] == "http"
        assert result["hardware_type"] == "cpu"
        assert result["placement_node"] == "node-b"
        assert result["output_cid"] == "cid-generated"
        assert result["provenance_cid"] == "cid-prov"
        assert result["outputs"] == ["generated:hello world"]

        backend = manager.get_backend("api_backend_2")
        assert backend is not None
        assert backend.metrics.total_requests == 1
        assert backend.metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_backend_manager_execute_embedding_task_runs_backend_and_finalizes_result(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager, BackendType, BackendCapabilities
        )

        class FakeEmbeddingBackend:
            device = "cpu"

            def run_inference(self, model_id=None, text=None, texts=None, **kwargs):
                payload = texts if texts is not None else [text]
                return {
                    "embeddings": [[float(i), float(i) + 0.5] for i, _ in enumerate(payload)],
                    "model": model_id,
                    "task": "text-embedding",
                }

        def fake_result_recorder(**kwargs):
            result = dict(kwargs["result"])
            result["output_cid"] = "cid-embed"
            result["provenance_cid"] = "cid-embed-prov"
            return result

        manager = InferenceBackendManager({"result_recorder": fake_result_recorder, "persist_registry": False})
        manager.register_backend(
            backend_id="api_backend_embed",
            backend_type=BackendType.API,
            name="API Embed Backend",
            instance=FakeEmbeddingBackend(),
            capabilities=BackendCapabilities(supported_tasks={"text-embedding"}),
            endpoint="http://localhost:9002",
        )

        result = await manager.execute_task(
            task="text-embedding",
            model="embed-model",
            inputs=["hello", "world"],
        )

        assert result["backend_id"] == "api_backend_embed"
        assert result["backend_type"] == "api"
        assert result["output_cid"] == "cid-embed"
        assert result["provenance_cid"] == "cid-embed-prov"
        assert result["embeddings"] == [[0.0, 0.5], [1.0, 1.5]]

        backend = manager.get_backend("api_backend_embed")
        assert backend is not None
        assert backend.metrics.total_requests == 1
        assert backend.metrics.successful_requests == 1


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


class TestUnifiedInferencePersistence:
    def test_record_inference_result_persists_and_indexes(self):
        from ipfs_accelerate_py.unified_inference_service import UnifiedInferenceService

        class FakeStorage:
            def __init__(self):
                self.items = []
                self.using_fallback = True

            def store(self, data, filename=None, pin=False):
                cid = f"cid-{len(self.items) + 1}"
                self.items.append({"cid": cid, "data": data, "filename": filename, "pin": pin})
                return cid

        class FakeDatasetsManager:
            def __init__(self):
                self.events = []
                self.provenance = []

            def log_event(self, event_type, data, level="INFO", category="GENERAL"):
                self.events.append(
                    {
                        "event_type": event_type,
                        "data": data,
                        "level": level,
                        "category": category,
                    }
                )
                return True

            def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
                self.provenance.append(
                    {
                        "operation": operation,
                        "data": data,
                        "record_type": record_type,
                    }
                )
                return "prov-cid-1"

        service = UnifiedInferenceService()
        fake_storage = FakeStorage()
        fake_datasets = FakeDatasetsManager()
        service._storage_client = fake_storage
        service._datasets_manager = fake_datasets

        result = service.record_inference_result(
            model="demo-model",
            inputs=["hello world"],
            result={
                "model": "demo-model",
                "outputs": ["ok"],
                "processing_time": 0.25,
                "device": "cpu",
            },
            backend_id="hf_server_local",
            backend_type="gpu",
            endpoint="http://127.0.0.1:8000",
            device="cpu",
        )

        assert result["input_cid"] == "cid-1"
        assert result["output_cid"] == "cid-2"
        assert result["provenance_cid"] == "prov-cid-1"
        assert result["audit_logged"] is True
        assert result["storage"]["success"] is True

        assert fake_storage.items[0]["filename"] == "demo-model_input.json"
        assert fake_storage.items[1]["filename"] == "demo-model_output.json"
        assert fake_datasets.events[0]["event_type"] == "inference_completed"
        assert fake_datasets.provenance[0]["data"]["output_cid"] == "cid-2"

    @pytest.mark.asyncio
    async def test_service_start_wires_backend_manager_result_recorder(self):
        from ipfs_accelerate_py.unified_inference_service import UnifiedInferenceService, InferenceServiceConfig

        service = UnifiedInferenceService(
            InferenceServiceConfig(
                enable_backend_manager=True,
                enable_hf_server=False,
                enable_websocket=False,
                enable_libp2p=False,
                enable_api_backends=False,
                enable_cli_backends=False,
            )
        )

        await service.start()
        try:
            assert service.backend_manager is not None
            assert service.backend_manager._result_recorder == service.record_inference_result
        finally:
            await service.stop()

    def test_record_inference_result_accepts_backend_finalize_metadata(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager,
            BackendType,
            BackendCapabilities,
        )
        from ipfs_accelerate_py.unified_inference_service import UnifiedInferenceService

        class FakeBackend:
            device = "cpu"

            def generate(self, model=None, prompt=None, **kwargs):
                return {"outputs": [f"ok:{prompt}"], "model": model}

        service = UnifiedInferenceService()
        manager = InferenceBackendManager(
            {
                "result_recorder": service.record_inference_result,
                "persist_registry": False,
            }
        )

        manager.register_backend(
            backend_id="api_backend_meta",
            backend_type=BackendType.API,
            name="API Meta Backend",
            instance=FakeBackend(),
            capabilities=BackendCapabilities(
                supported_tasks={"text-generation"},
                protocols={"http"},
                hardware_types={"cpu"},
            ),
            endpoint="http://localhost:9999",
            metadata={"placement_node": "node-meta"},
        )

        finalized = manager.finalize_inference_result(
            backend_id="api_backend_meta",
            task="text-generation",
            model="demo-model",
            inputs=["hello"],
            result={"outputs": ["ok"], "processing_time": 0.01, "device": "cpu"},
        )

        assert finalized["backend_id"] == "api_backend_meta"
        assert finalized["protocol"] == "http"
        assert finalized["hardware_type"] == "cpu"
        assert finalized["placement_node"] == "node-meta"

    def test_backend_selection_allows_string_preferred_types(self):
        from ipfs_accelerate_py.inference_backend_manager import (
            InferenceBackendManager,
            BackendType,
            BackendCapabilities,
            BackendStatus,
        )

        manager = InferenceBackendManager(
            {
                "load_balancing": "round_robin",
                "persist_registry": False,
            }
        )

        manager.register_backend(
            backend_id="backend_api",
            backend_type=BackendType.API,
            name="API Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}),
        )
        manager.register_backend(
            backend_id="backend_gpu",
            backend_type=BackendType.GPU,
            name="GPU Backend",
            instance=Mock(),
            capabilities=BackendCapabilities(supported_tasks={"text-generation"}),
        )

        manager.backends["backend_api"].status = BackendStatus.HEALTHY
        manager.backends["backend_gpu"].status = BackendStatus.HEALTHY

        selected = manager.select_backend_for_task(
            task="text-generation",
            preferred_types=["gpu"],
        )

        assert selected is not None
        assert selected.backend_id == "backend_gpu"


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
            from ipfs_accelerate_py.mcp_server.tools.backend_management import (
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
        from ipfs_accelerate_py.mcp_server.tools.backend_management import list_inference_backends
        
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
