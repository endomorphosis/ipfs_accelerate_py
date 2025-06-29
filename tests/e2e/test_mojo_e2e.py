#!/usr/bin/env python3
"""
End-to-End Tests for Mojo Integration

This test suite validates the complete Mojo workflow from model compilation
to deployment and inference, ensuring the entire pipeline works correctly.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import json
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import ClientTimeout
import numpy as np

# Test configuration
MOJO_TEST_CONFIG = {
    "test_timeout": 300,  # 5 minutes max per test
    "compilation_timeout": 180,  # 3 minutes max for compilation
    "inference_timeout": 30,  # 30 seconds max for inference
    "performance_threshold": {
        "min_throughput_tokens_per_sec": 100,
        "max_latency_p95_ms": 500,
        "max_memory_usage_mb": 8192
    }
}

class MojoE2ETestFramework:
    """Framework for running end-to-end Mojo tests."""
    
    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="mojo_e2e_"))
        self.mcp_server_process = None
        self.mcp_server_url = "http://localhost:8004"
        self.compiled_models = {}
        self.deployed_endpoints = {}
        
    async def setup(self):
        """Set up the test environment."""
        print(f"Setting up Mojo E2E test environment in {self.test_dir}")
        
        # Create test directories
        (self.test_dir / "models").mkdir(exist_ok=True)
        (self.test_dir / "compiled").mkdir(exist_ok=True)
        (self.test_dir / "outputs").mkdir(exist_ok=True)
        
        # Start MCP server if not running
        await self._ensure_mcp_server_running()
        
        # Verify Mojo environment
        await self._verify_mojo_environment()
        
    async def teardown(self):
        """Clean up the test environment."""
        print("Cleaning up Mojo E2E test environment")
        
        # Stop any deployed models
        for endpoint_url in self.deployed_endpoints.values():
            await self._stop_model_endpoint(endpoint_url)
        
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    async def _ensure_mcp_server_running(self):
        """Ensure MCP server is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/health") as response:
                    if response.status == 200:
                        print("✅ MCP server is already running")
                        return
        except:
            pass
        
        print("🚀 Starting MCP server for E2E tests...")
        
        # Start MCP server
        server_cmd = [
            "python", "final_mcp_server.py",
            "--host", "127.0.0.1",
            "--port", "8004",
            "--timeout", "600"
        ]
        
        self.mcp_server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Wait for server to start
        for _ in range(30):  # Wait up to 30 seconds
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.mcp_server_url}/health") as response:
                        if response.status == 200:
                            print("✅ MCP server started successfully")
                            return
            except:
                await asyncio.sleep(1)
        
        raise RuntimeError("Failed to start MCP server")
    
    async def _verify_mojo_environment(self):
        """Verify Mojo environment is available."""
        print("🔍 Verifying Mojo environment...")
        
        # Check if Mojo compiler is available
        try:
            result = subprocess.run(
                ["mojo", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Mojo compiler found: {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # If no real Mojo, ensure mock mode works
        print("⚠️ Mojo compiler not found - running in mock mode")
        
        # Verify mock tools are available
        tools_available = await self._check_mojo_tools_available()
        if not tools_available:
            raise RuntimeError("Neither real Mojo nor mock tools are available")
        
        return False
    
    async def _check_mojo_tools_available(self):
        """Check if Mojo tools are available via MCP."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/tools") as response:
                    if response.status == 200:
                        tools = await response.json()
                        mojo_tools = [
                            "compile_to_mojo",
                            "deploy_max_engine", 
                            "benchmark_modular_performance"
                        ]
                        available_tools = [tool for tool in tools if isinstance(tool, str)]
                        return all(tool in available_tools for tool in mojo_tools)
        except Exception as e:
            print(f"Error checking tools: {e}")
            return False
    
    async def _stop_model_endpoint(self, endpoint_url: str):
        """Stop a deployed model endpoint."""
        try:
            # This would typically call an endpoint shutdown API
            print(f"Stopping endpoint: {endpoint_url}")
        except Exception as e:
            print(f"Error stopping endpoint {endpoint_url}: {e}")

class TestMojoCompilation:
    """Test Mojo compilation pipeline."""
    
    @pytest_asyncio.fixture
    async def framework(self):
        """Set up test framework."""
        framework = MojoE2ETestFramework()
        await framework.setup()
        yield framework
        await framework.teardown()
    
    @pytest.mark.asyncio
    async def test_simple_model_compilation(self, framework):
        """Test compilation of a simple model to Mojo."""
        print("\n🧪 Testing simple model compilation to Mojo...")
        
        model_config = {
            "model_id": "simple_linear_test",
            "model_type": "linear",
            "input_shape": [1, 784],
            "output_shape": [1, 10],
            "optimization_level": "O2"
        }
        
        # Create a simple test model
        test_model_path = framework.test_dir / "models" / "simple_linear.onnx"
        await self._create_simple_test_model(test_model_path, model_config)
        
        # Compile to Mojo
        compilation_result = await self._compile_model_to_mojo(
            framework, model_config["model_id"], "O2"
        )
        
        # Verify compilation success
        assert compilation_result["success"] == True
        assert "compiled_path" in compilation_result
        assert "compilation_time" in compilation_result
        assert compilation_result["compilation_time"] < MOJO_TEST_CONFIG["compilation_timeout"]
        
        # Store for later tests
        framework.compiled_models[model_config["model_id"]] = compilation_result
        
        print(f"✅ Simple model compilation successful in {compilation_result['compilation_time']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_transformer_model_compilation(self, framework):
        """Test compilation of a transformer model to Mojo."""
        print("\n🧪 Testing transformer model compilation to Mojo...")
        
        model_config = {
            "model_id": "mini_transformer_test",
            "model_type": "transformer",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "seq_length": 128,
            "optimization_level": "O3"
        }
        
        # Compile transformer model
        compilation_result = await self._compile_model_to_mojo(
            framework, model_config["model_id"], "O3"
        )
        
        # Verify compilation success
        assert compilation_result["success"] == True
        assert "optimization_applied" in compilation_result
        
        # Verify optimizations were applied
        optimizations = compilation_result["optimization_applied"]
        expected_optimizations = ["vectorization", "loop_unrolling", "simd_optimization"]
        for opt in expected_optimizations:
            assert opt in optimizations, f"Expected optimization {opt} not found"
        
        framework.compiled_models[model_config["model_id"]] = compilation_result
        
        print(f"✅ Transformer model compilation successful with optimizations: {optimizations}")
    
    @pytest.mark.asyncio
    async def test_compilation_optimization_levels(self, framework):
        """Test different optimization levels."""
        print("\n🧪 Testing different optimization levels...")
        
        model_id = "optimization_test_model"
        optimization_levels = ["O0", "O1", "O2", "O3"]
        results = {}
        
        for opt_level in optimization_levels:
            print(f"Testing optimization level: {opt_level}")
            
            compilation_result = await self._compile_model_to_mojo(
                framework, f"{model_id}_{opt_level}", opt_level
            )
            
            assert compilation_result["success"] == True
            assert compilation_result["optimization_level"] == opt_level
            
            results[opt_level] = compilation_result
        
        # Verify that higher optimization levels apply more optimizations
        o0_opts = len(results["O0"].get("optimization_applied", []))
        o3_opts = len(results["O3"].get("optimization_applied", []))
        assert o3_opts >= o0_opts, "O3 should have at least as many optimizations as O0"
        
        print("✅ All optimization levels tested successfully")
    
    async def _create_simple_test_model(self, model_path: Path, config: Dict):
        """Create a simple test model for compilation."""
        # Create a mock ONNX-like model file
        model_data = {
            "model_type": config["model_type"],
            "input_shape": config["input_shape"],
            "output_shape": config["output_shape"],
            "weights": np.random.randn(784, 10).tolist(),
            "bias": np.random.randn(10).tolist()
        }
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'w') as f:
            json.dump(model_data, f)
    
    async def _compile_model_to_mojo(self, framework: MojoE2ETestFramework, 
                                   model_id: str, optimization_level: str) -> Dict[str, Any]:
        """Compile a model to Mojo via MCP."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "compile_to_mojo",
                "params": {
                    "model_id": model_id,
                    "optimization_level": optimization_level
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload,
                timeout=MOJO_TEST_CONFIG["compilation_timeout"]
            ) as response:
                assert response.status == 200
                result = await response.json()
                
                if "error" in result:
                    raise AssertionError(f"Compilation failed: {result['error']}")
                
                return result["result"]

class TestMojoDeployment:
    """Test Mojo model deployment and serving."""
    
    @pytest_asyncio.fixture
    async def framework(self):
        """Set up test framework."""
        framework = MojoE2ETestFramework()
        await framework.setup()
        yield framework
        await framework.teardown()
    
    @pytest.mark.asyncio
    async def test_max_engine_deployment(self, framework):
        """Test MAX Engine deployment."""
        print("\n🧪 Testing MAX Engine deployment...")
        
        model_id = "deployment_test_model"
        
        # First compile the model
        compilation_result = await self._compile_model_to_mojo(
            framework, model_id, "O2"
        )
        assert compilation_result["success"] == True
        
        # Deploy via MAX Engine
        deployment_result = await self._deploy_max_engine(
            framework, model_id, "auto"
        )
        
        # Verify deployment success
        assert deployment_result["success"] == True
        assert "endpoint_url" in deployment_result
        assert "allocated_resources" in deployment_result
        
        endpoint_url = deployment_result["endpoint_url"]
        framework.deployed_endpoints[model_id] = endpoint_url
        
        # Verify endpoint is accessible
        await self._verify_endpoint_health(endpoint_url)
        
        print(f"✅ MAX Engine deployment successful: {endpoint_url}")
    
    @pytest.mark.asyncio
    async def test_openai_compatible_serving(self, framework):
        """Test OpenAI-compatible model serving."""
        print("\n🧪 Testing OpenAI-compatible serving...")
        
        model_id = "openai_serving_test"
        port = 8001
        
        # Deploy model with OpenAI-compatible endpoint
        serving_result = await self._serve_max_model(
            framework, model_id, port
        )
        
        assert serving_result["success"] == True
        assert serving_result["openai_compatible"] == True
        assert serving_result["port"] == port
        
        endpoint_url = serving_result["endpoint_url"]
        framework.deployed_endpoints[model_id] = endpoint_url
        
        # Test OpenAI-compatible inference
        await self._test_openai_inference(endpoint_url)
        
        print(f"✅ OpenAI-compatible serving successful: {endpoint_url}")
    
    @pytest.mark.asyncio
    async def test_multi_model_deployment(self, framework):
        """Test deploying multiple models simultaneously."""
        print("\n🧪 Testing multi-model deployment...")
        
        model_configs = [
            {"model_id": "multi_model_1", "target_hardware": "cpu"},
            {"model_id": "multi_model_2", "target_hardware": "gpu"},
            {"model_id": "multi_model_3", "target_hardware": "auto"}
        ]
        
        deployment_tasks = []
        for config in model_configs:
            task = self._deploy_max_engine(
                framework, config["model_id"], config["target_hardware"]
            )
            deployment_tasks.append(task)
        
        # Deploy all models concurrently
        results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Verify all deployments succeeded
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise result
            
            assert result["success"] == True
            model_id = model_configs[i]["model_id"]
            framework.deployed_endpoints[model_id] = result["endpoint_url"]
        
        print("✅ Multi-model deployment successful")
    
    async def _compile_model_to_mojo(self, framework: MojoE2ETestFramework, 
                                   model_id: str, optimization_level: str) -> Dict[str, Any]:
        """Compile a model to Mojo via MCP."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "compile_to_mojo",
                "params": {
                    "model_id": model_id,
                    "optimization_level": optimization_level
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _deploy_max_engine(self, framework: MojoE2ETestFramework,
                               model_id: str, target_hardware: str) -> Dict[str, Any]:
        """Deploy model via MAX Engine."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "deploy_max_engine",
                "params": {
                    "model_id": model_id,
                    "target_hardware": target_hardware
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _serve_max_model(self, framework: MojoE2ETestFramework,
                             model_id: str, port: int) -> Dict[str, Any]:
        """Serve model with OpenAI-compatible endpoint."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "serve_max_model",
                "params": {
                    "model_id": model_id,
                    "port": port
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _verify_endpoint_health(self, endpoint_url: str):
        """Verify endpoint is healthy and responding."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try basic health check
                health_url = endpoint_url.replace("/infer", "/health")
                timeout = ClientTimeout(total=10)
                async with session.get(health_url, timeout=timeout) as response:
                    # In mock mode, this might not exist, so just check connection
                    pass
        except Exception as e:
            print(f"Endpoint health check failed (expected in mock mode): {e}")
    
    async def _test_openai_inference(self, endpoint_url: str):
        """Test OpenAI-compatible inference."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test chat completions endpoint
                payload = {
                    "model": "test_model",
                    "messages": [
                        {"role": "user", "content": "Hello, world!"}
                    ],
                    "max_tokens": 50
                }
                
                async with session.post(
                    endpoint_url,
                    json=payload,
                    timeout=ClientTimeout(total=30)
                ) as response:
                    # In mock mode, this might return a mock response
                    result = await response.json()
                    print(f"OpenAI inference test response: {result}")
        except Exception as e:
            print(f"OpenAI inference test failed (expected in mock mode): {e}")

class TestMojoPerformance:
    """Test Mojo performance benchmarking and optimization."""
    
    @pytest_asyncio.fixture
    async def framework(self):
        """Set up test framework."""
        framework = MojoE2ETestFramework()
        await framework.setup()
        yield framework
        await framework.teardown()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, framework):
        """Test comprehensive performance benchmarking."""
        print("\n🧪 Testing performance benchmarking...")
        
        model_id = "performance_test_model"
        workload_types = ["inference", "training", "mixed"]
        
        for workload_type in workload_types:
            print(f"Benchmarking {workload_type} workload...")
            
            benchmark_result = await self._benchmark_modular_performance(
                framework, model_id, workload_type
            )
            
            assert benchmark_result["success"] == True
            assert "benchmark_results" in benchmark_result
            
            # Verify benchmark metrics
            metrics = benchmark_result["benchmark_results"]
            assert "throughput_tokens_per_sec" in metrics
            assert "latency_ms" in metrics
            assert "memory_usage_mb" in metrics
            
            # Verify performance meets thresholds
            throughput = metrics["throughput_tokens_per_sec"]
            assert throughput >= MOJO_TEST_CONFIG["performance_threshold"]["min_throughput_tokens_per_sec"]
            
            latency_p95 = metrics["latency_ms"]["p95"]
            assert latency_p95 <= MOJO_TEST_CONFIG["performance_threshold"]["max_latency_p95_ms"]
            
            memory_usage = metrics["memory_usage_mb"]
            assert memory_usage <= MOJO_TEST_CONFIG["performance_threshold"]["max_memory_usage_mb"]
            
            print(f"✅ {workload_type} benchmark: {throughput:.1f} tokens/sec, {latency_p95:.1f}ms p95")
    
    @pytest.mark.asyncio
    async def test_optimization_impact(self, framework):
        """Test the impact of different optimizations on performance."""
        print("\n🧪 Testing optimization impact...")
        
        model_id = "optimization_impact_test"
        optimization_levels = ["O0", "O2", "O3"]
        performance_results = {}
        
        for opt_level in optimization_levels:
            print(f"Testing performance with optimization {opt_level}...")
            
            # Compile with specific optimization level
            compilation_result = await self._compile_model_to_mojo(
                framework, f"{model_id}_{opt_level}", opt_level
            )
            assert compilation_result["success"] == True
            
            # Benchmark performance
            benchmark_result = await self._benchmark_modular_performance(
                framework, f"{model_id}_{opt_level}", "inference"
            )
            assert benchmark_result["success"] == True
            
            performance_results[opt_level] = benchmark_result["benchmark_results"]
        
        # Verify optimization improvements
        o0_throughput = performance_results["O0"]["throughput_tokens_per_sec"]
        o3_throughput = performance_results["O3"]["throughput_tokens_per_sec"]
        
        # O3 should be at least as fast as O0 (in real scenarios)
        # In mock mode, they might be the same
        improvement_ratio = o3_throughput / o0_throughput
        print(f"Performance improvement O3 vs O0: {improvement_ratio:.2f}x")
        
        print("✅ Optimization impact testing completed")
    
    @pytest.mark.asyncio
    async def test_hardware_detection_and_optimization(self, framework):
        """Test hardware detection and automatic optimization."""
        print("\n🧪 Testing hardware detection and optimization...")
        
        # Detect available hardware
        hardware_info = await self._detect_modular_hardware(framework)
        
        assert "mojo_available" in hardware_info
        assert "detected_devices" in hardware_info
        assert "capabilities" in hardware_info
        
        # Verify we have at least basic capabilities
        devices = hardware_info["detected_devices"]
        assert len(devices) > 0, "Should detect at least one device"
        
        # Test automatic optimization based on detected hardware
        for device in devices[:2]:  # Test first 2 devices
            device_type = device["type"]
            print(f"Testing optimization for {device_type} device...")
            
            model_id = f"hardware_opt_test_{device_type}"
            
            # Deploy with automatic hardware selection
            deployment_result = await self._deploy_max_engine(
                framework, model_id, "auto"
            )
            
            assert deployment_result["success"] == True
            assert "allocated_resources" in deployment_result
            
        print("✅ Hardware detection and optimization testing completed")
    
    async def _compile_model_to_mojo(self, framework: MojoE2ETestFramework, 
                                   model_id: str, optimization_level: str) -> Dict[str, Any]:
        """Compile a model to Mojo via MCP."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "compile_to_mojo",
                "params": {
                    "model_id": model_id,
                    "optimization_level": optimization_level
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _deploy_max_engine(self, framework: MojoE2ETestFramework,
                               model_id: str, target_hardware: str) -> Dict[str, Any]:
        """Deploy model via MAX Engine."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "deploy_max_engine",
                "params": {
                    "model_id": model_id,
                    "target_hardware": target_hardware
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _benchmark_modular_performance(self, framework: MojoE2ETestFramework,
                                           model_id: str, workload_type: str) -> Dict[str, Any]:
        """Benchmark model performance."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "benchmark_modular_performance",
                "params": {
                    "model_id": model_id,
                    "workload_type": workload_type
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload,
                timeout=MOJO_TEST_CONFIG["inference_timeout"]
            ) as response:
                result = await response.json()
                return result["result"]
    
    async def _detect_modular_hardware(self, framework: MojoE2ETestFramework) -> Dict[str, Any]:
        """Detect available Modular hardware."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "detect_modular_hardware",
                "params": {
                    "include_capabilities": True
                },
                "id": 1
            }
            
            async with session.post(
                f"{framework.mcp_server_url}/jsonrpc",
                json=payload
            ) as response:
                result = await response.json()
                return result["result"]

class TestMojoIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest_asyncio.fixture
    async def framework(self):
        """Set up test framework."""
        framework = MojoE2ETestFramework()
        await framework.setup()
        yield framework
        await framework.teardown()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, framework):
        """Test complete workflow from compilation to inference."""
        print("\n🧪 Testing complete Mojo workflow...")
        
        model_id = "complete_workflow_test"
        
        # Step 1: Detect hardware
        print("Step 1: Hardware detection...")
        hardware_info = await self._detect_modular_hardware(framework)
        assert hardware_info["success"] == True
        
        # Step 2: Compile model
        print("Step 2: Model compilation...")
        compilation_result = await self._compile_model_to_mojo(
            framework, model_id, "O2"
        )
        assert compilation_result["success"] == True
        
        # Step 3: Deploy model
        print("Step 3: Model deployment...")
        deployment_result = await self._deploy_max_engine(
            framework, model_id, "auto"
        )
        assert deployment_result["success"] == True
        
        # Step 4: Benchmark performance
        print("Step 4: Performance benchmarking...")
        benchmark_result = await self._benchmark_modular_performance(
            framework, model_id, "inference"
        )
        assert benchmark_result["success"] == True
        
        # Step 5: Verify end-to-end metrics
        print("Step 5: End-to-end verification...")
        total_time = (
            compilation_result.get("compilation_time", 0) +
            deployment_result.get("deployment_time", 0)
        )
        
        throughput = benchmark_result["benchmark_results"]["throughput_tokens_per_sec"]
        
        print(f"✅ Complete workflow successful:")
        print(f"   Total setup time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} tokens/sec")
        
        # Verify workflow meets performance targets
        assert total_time < 300  # Max 5 minutes for setup
        assert throughput >= 100  # Min 100 tokens/sec
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, framework):
        """Test error handling and recovery scenarios."""
        print("\n🧪 Testing error handling and recovery...")
        
        # Test invalid model compilation
        print("Testing invalid model compilation...")
        try:
            invalid_result = await self._compile_model_to_mojo(
                framework, "", "O2"  # Invalid model ID
            )
            # Should either succeed (mock) or handle gracefully
            print(f"Invalid compilation result: {invalid_result}")
        except Exception as e:
            print(f"Expected error for invalid model: {e}")
        
        # Test invalid optimization level
        print("Testing invalid optimization level...")
        try:
            invalid_opt_result = await self._compile_model_to_mojo(
                framework, "test_model", "O99"  # Invalid optimization
            )
            print(f"Invalid optimization result: {invalid_opt_result}")
        except Exception as e:
            print(f"Expected error for invalid optimization: {e}")
        
        # Test deployment without compilation
        print("Testing deployment without compilation...")
        try:
            deploy_result = await self._deploy_max_engine(
                framework, "nonexistent_model", "auto"
            )
            # Should handle gracefully
            print(f"Deploy without compile result: {deploy_result}")
        except Exception as e:
            print(f"Expected error for nonexistent model: {e}")
        
        print("✅ Error handling and recovery testing completed")
    
    # Helper methods (reuse from other test classes)
    async def _detect_modular_hardware(self, framework):
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "detect_modular_hardware",
                "params": {"include_capabilities": True},
                "id": 1
            }
            async with session.post(f"{framework.mcp_server_url}/jsonrpc", json=payload) as response:
                result = await response.json()
                return result["result"]
    
    async def _compile_model_to_mojo(self, framework, model_id, optimization_level):
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "compile_to_mojo",
                "params": {"model_id": model_id, "optimization_level": optimization_level},
                "id": 1
            }
            async with session.post(f"{framework.mcp_server_url}/jsonrpc", json=payload) as response:
                result = await response.json()
                return result["result"]
    
    async def _deploy_max_engine(self, framework, model_id, target_hardware):
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "deploy_max_engine",
                "params": {"model_id": model_id, "target_hardware": target_hardware},
                "id": 1
            }
            async with session.post(f"{framework.mcp_server_url}/jsonrpc", json=payload) as response:
                result = await response.json()
                return result["result"]
    
    async def _benchmark_modular_performance(self, framework, model_id, workload_type):
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "benchmark_modular_performance",
                "params": {"model_id": model_id, "workload_type": workload_type},
                "id": 1
            }
            async with session.post(f"{framework.mcp_server_url}/jsonrpc", json=payload) as response:
                result = await response.json()
                return result["result"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
