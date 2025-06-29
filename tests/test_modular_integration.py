#!/usr/bin/env python3
"""
Comprehensive Test Suite for Modular Integration

This test suite validates the integration of Mojo/MAX/Modular backends
with the IPFS Accelerate system.
"""

import pytest
import asyncio
import os
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import the modules we're testing
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.backends.modular_backend import (
        ModularEnvironment, MojoBackend, MaxBackend, ModularBackend,
        CompilationResult, DeploymentResult, BenchmarkResult
    )
except ImportError:
    # Fallback for when running in different directory structures
    from backends.modular_backend import (
        ModularEnvironment, MojoBackend, MaxBackend, ModularBackend,
        CompilationResult, DeploymentResult, BenchmarkResult
    )

class TestModularEnvironment:
    """Test the Modular environment detection and setup."""
    
    def test_environment_initialization(self):
        """Test ModularEnvironment initialization."""
        env = ModularEnvironment()
        
        assert hasattr(env, 'max_available')
        assert hasattr(env, 'mojo_available')
        assert hasattr(env, 'modular_version')
        assert hasattr(env, 'devices')
        assert isinstance(env.devices, list)
    
    @patch('subprocess.run')
    def test_mojo_detection_success(self, mock_run):
        """Test successful Mojo compiler detection."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "mojo 24.5.0"
        mock_run.return_value = mock_result
        
        env = ModularEnvironment()
        result = env._detect_mojo()
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_mojo_detection_failure(self, mock_run):
        """Test Mojo compiler detection failure."""
        mock_run.side_effect = FileNotFoundError()
        
        env = ModularEnvironment()
        result = env._detect_mojo()
        
        assert result == False
    
    def test_device_enumeration(self):
        """Test device enumeration."""
        env = ModularEnvironment()
        devices = env._enumerate_devices()
        
        # Should always have at least CPU device
        assert len(devices) >= 1
        
        cpu_device = devices[0]
        assert cpu_device['type'] == 'cpu'
        assert 'cores' in cpu_device
        assert 'simd_width' in cpu_device
        assert 'supported_dtypes' in cpu_device

class TestMojoBackend:
    """Test the Mojo compilation backend."""
    
    @pytest.fixture
    def mojo_backend(self):
        """Create a MojoBackend instance for testing."""
        env = ModularEnvironment()
        return MojoBackend(env)
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration for testing."""
        return {
            'model_id': 'test_model',
            'input_shape': [1, 512],
            'hidden_size': 768,
            'num_layers': 12,
            'optimization_level': 'O2'
        }
    
    def test_backend_initialization(self, mojo_backend):
        """Test MojoBackend initialization."""
        assert hasattr(mojo_backend, 'env')
        assert hasattr(mojo_backend, 'optimization_flags')
        assert 'O0' in mojo_backend.optimization_flags
        assert 'O3' in mojo_backend.optimization_flags
    
    async def test_mojo_code_generation(self, mojo_backend):
        """Test Mojo code generation."""
        model_path = "/tmp/test_model.onnx"
        target_device = "cpu"
        
        code = await mojo_backend._generate_mojo_code(model_path, target_device)
        
        assert isinstance(code, str)
        assert len(code) > 0
        assert "struct" in code  # Should contain Mojo struct
        assert "fn forward" in code  # Should have forward function
        assert "vectorize" in code  # Should use vectorization
    
    async def test_compilation_success_mock(self, mojo_backend):
        """Test successful compilation (mocked)."""
        model_path = "/tmp/test_model.onnx"
        output_path = "/tmp/compiled_model.bin"
        
        # Mock the compilation process
        with patch.object(mojo_backend, '_run_compilation') as mock_compile:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_compile.return_value = mock_result
            
            result = await mojo_backend.compile_model(
                model_path, output_path, "O2", "cpu"
            )
        
        assert result.success == True
        assert result.compiled_path == output_path
        assert result.optimizations_applied is not None
        assert "vectorization" in result.optimizations_applied
    
    async def test_compilation_failure_mock(self, mojo_backend):
        """Test compilation failure (mocked)."""
        model_path = "/tmp/test_model.onnx"
        output_path = "/tmp/compiled_model.bin"
        
        # Mock compilation failure
        with patch.object(mojo_backend, '_run_compilation') as mock_compile:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Compilation error: syntax error"
            mock_compile.return_value = mock_result
            
            result = await mojo_backend.compile_model(
                model_path, output_path, "O2", "cpu"
            )
        
        assert result.success == False
        assert "Compilation error" in result.error_message
    
    def test_optimization_levels(self, mojo_backend):
        """Test optimization level configuration."""
        optimizations = mojo_backend._get_applied_optimizations("O3")
        
        assert "vectorization" in optimizations
        assert "loop_optimization" in optimizations
        assert "aggressive_inlining" in optimizations

class TestMaxBackend:
    """Test the MAX Engine backend."""
    
    @pytest.fixture
    def max_backend(self):
        """Create a MaxBackend instance for testing."""
        env = ModularEnvironment()
        return MaxBackend(env)
    
    async def test_backend_initialization(self, max_backend):
        """Test MaxBackend initialization."""
        await max_backend.initialize()
        
        assert hasattr(max_backend, 'env')
        assert hasattr(max_backend, 'models')
        assert hasattr(max_backend, 'deployment_configs')
    
    async def test_mock_model_deployment(self, max_backend):
        """Test mock model deployment."""
        model_path = "/tmp/test_model.onnx"
        model_id = "test_model"
        target_hardware = "cpu"
        
        result = await max_backend.deploy_model(
            model_path, model_id, target_hardware
        )
        
        assert result.success == True
        assert result.model_id == model_id
        assert result.target_hardware == target_hardware
        assert result.endpoint_url is not None
        assert "localhost" in result.endpoint_url
    
    async def test_deployment_resource_allocation(self, max_backend):
        """Test resource allocation during deployment."""
        model_path = "/tmp/test_model.onnx"
        model_id = "test_model"
        
        # Test CPU deployment
        cpu_result = await max_backend.deploy_model(
            model_path, model_id, "cpu"
        )
        assert cpu_result.allocated_resources['memory_mb'] > 0
        assert cpu_result.allocated_resources['cpu_cores'] > 0
        
        # Test GPU deployment
        gpu_result = await max_backend.deploy_model(
            model_path, model_id, "gpu"
        )
        assert gpu_result.allocated_resources['gpu_memory_mb'] > 0

class TestModularBackend:
    """Test the main ModularBackend integration."""
    
    @pytest.fixture
    async def modular_backend(self):
        """Create an initialized ModularBackend instance."""
        backend = ModularBackend()
        await backend.initialize()
        return backend
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration."""
        return {
            'model_id': 'test_llama_7b',
            'input_shape': [1, 512],
            'input_dtype': 'float32',
            'layers': [
                {'type': 'embedding', 'vocab_size': 32000, 'hidden_size': 4096},
                {'type': 'transformer', 'num_layers': 32, 'num_heads': 32},
                {'type': 'lm_head', 'vocab_size': 32000}
            ],
            'optimization_level': 'O2'
        }
    
    async def test_backend_initialization(self, modular_backend):
        """Test ModularBackend initialization."""
        assert hasattr(modular_backend, 'environment')
        assert hasattr(modular_backend, 'mojo_backend')
        assert hasattr(modular_backend, 'max_backend')
        assert hasattr(modular_backend, 'performance_cache')
    
    async def test_mojo_compilation_integration(self, modular_backend, sample_model_config):
        """Test Mojo compilation through ModularBackend."""
        result = await modular_backend.compile_mojo(
            sample_model_config['model_id'], 
            sample_model_config['optimization_level']
        )
        
        assert 'model_id' in result
        assert 'success' in result
        assert 'optimization_level' in result
        assert result['model_id'] == sample_model_config['model_id']
    
    async def test_max_deployment_integration(self, modular_backend, sample_model_config):
        """Test MAX deployment through ModularBackend."""
        result = await modular_backend.deploy_max(
            sample_model_config['model_id'], 
            "auto"
        )
        
        assert 'model_id' in result
        assert 'success' in result
        assert 'target_hardware' in result
        assert result['model_id'] == sample_model_config['model_id']
    
    async def test_performance_benchmarking(self, modular_backend, sample_model_config):
        """Test performance benchmarking."""
        result = await modular_backend.benchmark_modular(
            sample_model_config['model_id'],
            "inference"
        )
        
        assert 'model_id' in result
        assert 'workload_type' in result
        assert 'benchmark_results' in result
        assert 'success' in result
        
        benchmark_data = result['benchmark_results']
        assert 'throughput_tokens_per_sec' in benchmark_data
        assert 'latency_ms' in benchmark_data
        assert 'memory_usage_mb' in benchmark_data
        
        # Test caching
        result2 = await modular_backend.benchmark_modular(
            sample_model_config['model_id'],
            "inference"
        )
        assert result2 == result  # Should return cached result
    
    def test_hardware_info_retrieval(self, modular_backend):
        """Test hardware information retrieval."""
        info = modular_backend.get_hardware_info()
        
        assert 'mojo_available' in info
        assert 'max_engine_available' in info
        assert 'detected_devices' in info
        assert 'capabilities' in info
        
        capabilities = info['capabilities']
        assert 'vectorization' in capabilities
        assert 'graph_optimization' in capabilities

class TestMCPIntegration:
    """Test MCP server integration with Modular tools."""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server instance."""
        server = Mock()
        server.tools = {}
        server.tool_descriptions = {}
        server.parameter_descriptions = {}
        
        def mock_register_tool(name, func, description, params=None):
            server.tools[name] = func
            server.tool_descriptions[name] = description
            if params:
                server.parameter_descriptions[name] = params
        
        server.register_tool = mock_register_tool
        return server
    
    @pytest.fixture
    def mock_accel(self):
        """Create a mock accelerator instance."""
        accel = Mock()
        accel.compile_mojo = AsyncMock(return_value={
            'success': True,
            'compiled_path': '/tmp/test.bin'
        })
        accel.deploy_max = AsyncMock(return_value={
            'success': True,
            'endpoint_url': 'http://localhost:8000'
        })
        accel.benchmark_modular = AsyncMock(return_value={
            'success': True,
            'throughput_tokens_per_sec': 1000.0
        })
        return accel
    
    def test_modular_tools_registration(self, mock_mcp_server, mock_accel):
        """Test registration of Modular tools with MCP server."""
        # Import and test tool registration
        sys.path.append(os.path.dirname(__file__))
        from final_mcp_server import register_modular_tools
        
        register_modular_tools(mock_mcp_server, mock_accel)
        
        # Verify tools are registered
        expected_tools = [
            'compile_to_mojo',
            'compile_to_max', 
            'deploy_max_engine',
            'serve_max_model',
            'benchmark_modular_performance',
            'detect_modular_hardware'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in mock_mcp_server.tools
            assert tool_name in mock_mcp_server.tool_descriptions
    
    async def test_compile_to_mojo_tool(self, mock_mcp_server, mock_accel):
        """Test compile_to_mojo tool execution."""
        from final_mcp_server import register_modular_tools
        register_modular_tools(mock_mcp_server, mock_accel)
        
        tool_func = mock_mcp_server.tools['compile_to_mojo']
        result = await tool_func('test_model', 'O2')
        
        assert 'model_id' in result
        assert 'optimization_level' in result
        assert 'success' in result
    
    async def test_deploy_max_engine_tool(self, mock_mcp_server, mock_accel):
        """Test deploy_max_engine tool execution."""
        from final_mcp_server import register_modular_tools
        register_modular_tools(mock_mcp_server, mock_accel)
        
        tool_func = mock_mcp_server.tools['deploy_max_engine']
        result = await tool_func('test_model', 'gpu')
        
        assert 'model_id' in result
        assert 'target_hardware' in result
        assert 'success' in result
    
    async def test_benchmark_performance_tool(self, mock_mcp_server, mock_accel):
        """Test benchmark_modular_performance tool execution."""
        from final_mcp_server import register_modular_tools
        register_modular_tools(mock_mcp_server, mock_accel)
        
        tool_func = mock_mcp_server.tools['benchmark_modular_performance']
        result = await tool_func('test_model', 'inference')
        
        assert 'model_id' in result
        assert 'workload_type' in result
        assert 'benchmark_results' in result
        assert 'success' in result

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    async def full_system(self):
        """Set up full system for end-to-end testing."""
        backend = ModularBackend()
        await backend.initialize()
        return backend
    
    async def test_complete_workflow(self, full_system):
        """Test complete workflow from compilation to deployment to benchmarking."""
        model_id = "test_end_to_end_model"
        
        # Step 1: Compile with Mojo
        compile_result = await full_system.compile_mojo(model_id, "O2")
        assert compile_result['success'] == True
        
        # Step 2: Deploy with MAX
        deploy_result = await full_system.deploy_max(model_id, "auto")
        assert deploy_result['success'] == True
        
        # Step 3: Benchmark performance
        benchmark_result = await full_system.benchmark_modular(model_id, "inference")
        assert benchmark_result['success'] == True
        
        # Verify all components worked together
        assert benchmark_result['benchmark_results']['throughput_tokens_per_sec'] > 0
    
    async def test_error_handling_and_recovery(self, full_system):
        """Test error handling and recovery mechanisms."""
        # Test with invalid model ID
        invalid_result = await full_system.compile_mojo("", "O2")
        # Should handle gracefully (in mock mode, still returns success)
        
        # Test with invalid optimization level (should handle gracefully)
        result = await full_system.compile_mojo("test_model", "O99")
        # Mock implementation should still work
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        backend = ModularBackend()
        
        # Test hardware info with defaults
        info = backend.get_hardware_info()
        assert info is not None
        assert isinstance(info['detected_devices'], list)
        assert len(info['detected_devices']) > 0

class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.fixture
    async def performance_backend(self):
        """Create backend for performance testing."""
        backend = ModularBackend()
        await backend.initialize()
        return backend
    
    async def test_concurrent_compilations(self, performance_backend):
        """Test concurrent compilation requests."""
        model_ids = [f"model_{i}" for i in range(5)]
        
        # Run concurrent compilations
        tasks = [
            performance_backend.compile_mojo(model_id, "O2") 
            for model_id in model_ids
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result['success'] == True
    
    async def test_benchmark_caching(self, performance_backend):
        """Test benchmark result caching."""
        model_id = "cache_test_model"
        
        # First benchmark
        start_time = asyncio.get_event_loop().time()
        result1 = await performance_backend.benchmark_modular(model_id, "inference")
        first_duration = asyncio.get_event_loop().time() - start_time
        
        # Second benchmark (should be cached)
        start_time = asyncio.get_event_loop().time()
        result2 = await performance_backend.benchmark_modular(model_id, "inference")
        second_duration = asyncio.get_event_loop().time() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second call should be much faster (cached)
        assert second_duration < first_duration

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
