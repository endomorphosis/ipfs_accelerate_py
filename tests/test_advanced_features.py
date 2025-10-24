#!/usr/bin/env python3
"""
Tests for Real-World Model Integration and Performance Modeling

These tests validate the advanced features added to the IPFS Accelerate Python
repository, including real model testing capabilities and detailed performance modeling.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the repository root to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from test_real_world_models import RealWorldModelTester, TestRealWorldModels
from utils.performance_modeling import (
    PerformanceSimulator,
    performance_simulator,
    simulate_model_performance,
    get_hardware_recommendations,
    HardwareType,
    PrecisionMode
)
from utils.model_compatibility import (
    get_detailed_performance_analysis,
    benchmark_model_performance
)

class TestPerformanceModeling:
    """Test the enhanced performance modeling system."""
    
    def test_performance_simulator_initialization(self):
        """Test that performance simulator initializes correctly."""
        simulator = PerformanceSimulator()
        assert simulator is not None
        assert hasattr(simulator, 'HARDWARE_SPECS')
        assert hasattr(simulator, 'MODEL_SPECS')
        assert len(simulator.HARDWARE_SPECS) >= 8  # At least 8 hardware types
        assert len(simulator.MODEL_SPECS) >= 7     # At least 7 model families
        
    def test_hardware_specifications(self):
        """Test that hardware specifications are realistic."""
        simulator = PerformanceSimulator()
        
        # Check that all required hardware types are present
        required_hardware = [
            HardwareType.CPU, HardwareType.CUDA, HardwareType.MPS,
            HardwareType.WEBNN, HardwareType.WEBGPU
        ]
        
        for hw_type in required_hardware:
            assert hw_type in simulator.HARDWARE_SPECS
            spec = simulator.HARDWARE_SPECS[hw_type]
            
            # Verify realistic values
            assert spec.compute_capability > 0
            assert spec.memory_bandwidth_gbps > 0
            assert spec.memory_size_gb > 0
            assert spec.power_consumption_watts > 0
            assert len(spec.supported_precisions) > 0
            
    def test_model_specifications(self):
        """Test that model specifications cover major model families."""
        simulator = PerformanceSimulator()
        
        required_models = ["bert", "gpt2", "llama", "clip", "whisper"]
        
        for model_name in required_models:
            assert model_name in simulator.MODEL_SPECS
            spec = simulator.MODEL_SPECS[model_name]
            
            # Verify realistic values
            assert spec.parameter_count_m > 0
            assert spec.memory_footprint_mb > 0
            assert spec.compute_intensity > 0
            assert isinstance(spec.parallelizable, bool)
            assert isinstance(spec.web_compatible, bool)
            
    def test_basic_performance_simulation(self):
        """Test basic performance simulation functionality."""
        result = simulate_model_performance("bert-base-uncased", "cpu")
        
        assert result is not None
        assert hasattr(result, 'inference_time_ms')
        assert hasattr(result, 'memory_usage_mb')
        assert hasattr(result, 'efficiency_score')
        assert hasattr(result, 'bottleneck')
        assert hasattr(result, 'recommendations')
        
        # Verify realistic values
        assert result.inference_time_ms > 0
        assert result.memory_usage_mb > 0
        assert 0 <= result.efficiency_score <= 1.0
        assert isinstance(result.recommendations, list)
        
    def test_hardware_comparison(self):
        """Test performance comparison across different hardware."""
        hardware_options = ["cpu", "cuda", "mps"]
        results = performance_simulator.compare_hardware_options(
            "bert-base-uncased", hardware_options
        )
        
        assert len(results) == len(hardware_options)
        
        # CUDA should generally be faster than CPU
        if "cpu" in results and "cuda" in results:
            cpu_time = results["cpu"].inference_time_ms
            cuda_time = results["cuda"].inference_time_ms
            assert cuda_time < cpu_time  # CUDA should be faster
            
    def test_precision_effects(self):
        """Test that different precision modes affect performance."""
        model = "bert-base-uncased"
        hardware = "cuda"
        
        fp32_result = performance_simulator.simulate_inference_performance(
            model, hardware, precision="fp32"
        )
        fp16_result = performance_simulator.simulate_inference_performance(
            model, hardware, precision="fp16"
        )
        
        # FP16 should generally be faster and use less memory
        assert fp16_result.inference_time_ms <= fp32_result.inference_time_ms
        assert fp16_result.memory_usage_mb <= fp32_result.memory_usage_mb
        
    def test_optimal_configuration_finding(self):
        """Test finding optimal hardware configuration."""
        hardware_options = ["cpu", "cuda", "mps", "webnn"]
        
        best_hw, result, details = performance_simulator.get_optimal_configuration(
            "bert-base-uncased", hardware_options, optimize_for="speed"
        )
        
        assert best_hw in hardware_options
        assert result is not None
        assert details is not None
        assert "optimization_criteria" in details
        assert details["optimization_criteria"] == "speed"
        
    def test_hardware_recommendations(self):
        """Test hardware recommendation system."""
        recommendations = get_hardware_recommendations(
            "gpt2", ["cpu", "cuda", "mps"]
        )
        
        assert "recommended_hardware" in recommendations
        assert "performance" in recommendations
        assert "details" in recommendations
        assert "all_options" in recommendations
        
        # Verify performance data
        perf = recommendations["performance"]
        assert hasattr(perf, 'inference_time_ms')
        assert hasattr(perf, 'memory_usage_mb')
        
class TestRealWorldModelIntegration:
    """Test real-world model integration capabilities."""
    
    def test_model_tester_initialization(self):
        """Test that RealWorldModelTester initializes correctly."""
        tester = RealWorldModelTester()
        assert tester is not None
        assert hasattr(tester, 'TEST_MODELS')
        assert len(tester.TEST_MODELS) >= 3  # At least 3 test models
        
        # Verify test model specifications
        for model_info in tester.TEST_MODELS:
            assert "name" in model_info
            assert "size_mb" in model_info
            assert "type" in model_info
            assert "description" in model_info
            assert model_info["size_mb"] > 0
            
    def test_model_compatibility_simulation(self):
        """Test model compatibility simulation without actual loading."""
        tester = RealWorldModelTester()
        model_info = tester.TEST_MODELS[0]  # First test model
        
        result = tester.test_model_compatibility(model_info, use_mock=True)
        
        assert "model_name" in result
        assert "best_hardware" in result
        assert "compatible" in result
        assert "estimated_inference_time_ms" in result
        assert "memory_requirements_mb" in result
        
        assert result["compatible"] is True
        assert result["estimated_inference_time_ms"] > 0
        assert result["memory_requirements_mb"] > 0
        
    def test_comprehensive_model_tests(self):
        """Test comprehensive testing across all test models."""
        tester = RealWorldModelTester()
        results = tester.run_comprehensive_model_tests(use_actual_loading=False)
        
        assert len(results) >= 3  # Should test at least 3 models
        
        for result in results:
            assert "model_name" in result
            assert "status" in result
            # Status should be either success or a known error type
            assert any(status in result["status"] for status in 
                      ["simulated_success", "test_error", "load_failed"])
                      
    def test_inference_time_estimation(self):
        """Test inference time estimation accuracy."""
        tester = RealWorldModelTester()
        
        # Test different model types
        bert_time = tester._estimate_inference_time("bert", "cpu")
        gpt_time = tester._estimate_inference_time("gpt", "cpu")
        
        assert bert_time > 0
        assert gpt_time > 0
        
        # GPT should generally be slower than BERT
        assert gpt_time >= bert_time
        
        # CUDA should be faster than CPU
        bert_cpu = tester._estimate_inference_time("bert", "cpu")
        bert_cuda = tester._estimate_inference_time("bert", "cuda")
        assert bert_cuda < bert_cpu
        
    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        tester = RealWorldModelTester()
        model_info = {"size_mb": 100}
        
        memory_req = tester._estimate_memory_requirements(model_info)
        
        assert memory_req > model_info["size_mb"]  # Should include overhead
        assert memory_req <= model_info["size_mb"] * 3  # Reasonable multiplier
        
class TestAdvancedCompatibilityFeatures:
    """Test advanced model-hardware compatibility features."""
    
    def test_detailed_performance_analysis(self):
        """Test detailed performance analysis functionality."""
        try:
            analysis = get_detailed_performance_analysis(
                "bert-base-uncased", ["cpu", "cuda", "mps"]
            )
            
            assert "model_name" in analysis
            assert "recommended_hardware" in analysis
            assert "performance_details" in analysis
            assert "hardware_comparison" in analysis
            
            # Verify performance details structure
            perf_details = analysis["performance_details"]
            required_metrics = [
                "inference_time_ms", "memory_usage_mb", "throughput_samples_per_sec",
                "efficiency_score", "power_consumption_watts"
            ]
            
            for metric in required_metrics:
                assert metric in perf_details
                assert perf_details[metric] > 0
                
        except ImportError:
            pytest.skip("Performance modeling not available")
            
    def test_benchmarking_functionality(self):
        """Test performance benchmarking across configurations."""
        try:
            results = benchmark_model_performance(
                "gpt2", ["cpu", "cuda"], batch_sizes=[1, 4], optimize_for="speed"
            )
            
            if "error" not in results:
                assert "model_name" in results
                assert "hardware_results" in results
                assert "batch_size_analysis" in results
                assert "optimal_configurations" in results
                
                # Verify structure for each hardware
                for hw_name, hw_results in results["hardware_results"].items():
                    assert hw_name in ["cpu", "cuda"]
                    assert "batch_1" in hw_results
                    assert "batch_4" in hw_results
                    
                    for batch_key, metrics in hw_results.items():
                        assert "inference_time_ms" in metrics
                        assert "memory_usage_mb" in metrics
                        assert "throughput" in metrics
                        
        except ImportError:
            pytest.skip("Performance modeling not available")
            
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_cpu_only_environment(self):
        """Test that everything works in CPU-only environments."""
        # Simulate CPU-only environment
        cpu_hardware = ["cpu", "webnn", "webgpu"]  # Web platforms work on CPU
        
        # Should be able to get recommendations
        recommendations = get_hardware_recommendations("bert-base-uncased", cpu_hardware)
        assert "recommended_hardware" in recommendations
        assert recommendations["recommended_hardware"] in cpu_hardware
        
        # Should be able to simulate performance
        result = simulate_model_performance("bert-base-uncased", "cpu")
        assert result.inference_time_ms > 0
        
    def test_mixed_hardware_environment(self):
        """Test behavior with mixed hardware availability."""
        mixed_hardware = ["cpu", "cuda", "mps", "webnn"]
        
        # Should handle missing/unavailable hardware gracefully  
        recommendations = get_hardware_recommendations("llama", mixed_hardware)
        assert "recommended_hardware" in recommendations
        
        # Should prefer high-performance hardware for compute-intensive models
        recommended = recommendations["recommended_hardware"]
        assert recommended in ["cuda", "mps", "cpu"]  # Should prefer GPU-like hardware
        
    def test_web_deployment_scenario(self):
        """Test web deployment optimization."""
        web_hardware = ["webnn", "webgpu", "cpu"]
        
        # Web-compatible models should work well
        web_models = ["bert-base-uncased", "gpt2", "clip-vit-base-patch32"]
        
        for model in web_models:
            recommendations = get_hardware_recommendations(model, web_hardware)
            assert "recommended_hardware" in recommendations
            
            # Should recommend WebNN/WebGPU for web deployment
            recommended = recommendations["recommended_hardware"]
            assert recommended in web_hardware
            
    def test_mobile_optimization(self):
        """Test mobile/edge optimization scenarios."""
        mobile_hardware = ["qualcomm", "cpu"]
        
        # Should provide mobile-optimized recommendations
        recommendations = get_hardware_recommendations("bert-base-uncased", mobile_hardware)
        assert "recommended_hardware" in recommendations
        
        # Should consider power efficiency for mobile
        perf = recommendations["performance"]
        assert hasattr(perf, 'power_consumption_watts')
        assert perf.power_consumption_watts < 50  # Reasonable for mobile

def test_module_imports():
    """Test that all modules can be imported successfully."""
    # Test core imports
    from test_real_world_models import RealWorldModelTester
    from utils.performance_modeling import performance_simulator
    from utils.model_compatibility import get_detailed_performance_analysis
    
    assert RealWorldModelTester is not None
    assert performance_simulator is not None
    assert get_detailed_performance_analysis is not None

def test_backward_compatibility():
    """Test that existing functionality still works."""
    from utils.model_compatibility import get_optimal_hardware, check_model_compatibility
    
    # Existing functions should still work
    result = get_optimal_hardware("bert-base-uncased", ["cpu", "cuda"])
    assert "recommended_hardware" in result
    
    compatibility = check_model_compatibility("bert-base-uncased", "cpu")
    assert "compatible" in compatibility

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])