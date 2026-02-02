#!/usr/bin/env python3
"""
Pytest-compatible test template for HuggingFace models.

This template provides a standardized structure for testing HuggingFace models
with proper pytest functions, assertions, and hardware compatibility.

Usage:
    Replace DINOV2, dinov2-base, feature-extraction with actual values.
"""

import pytest
import torch
from typing import Dict, Any
from unittest.mock import MagicMock

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.test_utils import ModelTestUtils, HardwareTestUtils, PerformanceTestUtils

# Try to import required packages
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    pytest.skip("transformers not available", allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    pytest.skip("torch not available", allow_module_level=True)


# Model configuration
MODEL_ID = "dinov2-base"  # e.g., "bert-base-uncased"
MODEL_NAME = "DINOV2"  # e.g., "bert"
TASK_TYPE = "feature-extraction"  # e.g., "text_embedding"


# Fixtures

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer (cached for module)."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


@pytest.fixture
def sample_inputs(model_and_tokenizer):
    """Create sample inputs for testing."""
    _, tokenizer = model_and_tokenizer
    return ModelTestUtils.create_sample_text_inputs(tokenizer)


# Basic Tests

@pytest.mark.model_test
@pytest.mark.model
@pytest.mark.text
class TestModelLoading:
    """Test model loading and initialization."""
    
    def test_model_loads(self, model_and_tokenizer):
        """Test that model loads successfully."""
        model, tokenizer = model_and_tokenizer
        ModelTestUtils.assert_model_loaded(model, MODEL_NAME)
        ModelTestUtils.assert_tokenizer_loaded(tokenizer, MODEL_NAME)
    
    def test_model_config(self, model_and_tokenizer):
        """Test model configuration."""
        model, _ = model_and_tokenizer
        assert hasattr(model, 'config'), "Model has no config"
        assert model.config is not None, "Model config is None"
    
    def test_model_parameters(self, model_and_tokenizer):
        """Test model has parameters."""
        model, _ = model_and_tokenizer
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"
        assert all(isinstance(p, torch.Tensor) for p in params), "Not all parameters are tensors"


@pytest.mark.model_test
@pytest.mark.model
@pytest.mark.text
class TestInference:
    """Test model inference."""
    
    def test_forward_pass(self, model_and_tokenizer, sample_inputs):
        """Test basic forward pass."""
        model, _ = model_and_tokenizer
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None, "Model output is None"
        assert hasattr(outputs, 'last_hidden_state'), "Output missing last_hidden_state"
        ModelTestUtils.assert_tensor_valid(outputs.last_hidden_state, "last_hidden_state")
    
    def test_output_shape(self, model_and_tokenizer, sample_inputs):
        """Test output shape is correct."""
        model, _ = model_and_tokenizer
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        batch_size = sample_inputs['input_ids'].shape[0]
        seq_length = sample_inputs['input_ids'].shape[1]
        hidden_size = model.config.hidden_size
        
        expected_shape = (batch_size, seq_length, hidden_size)
        assert outputs.last_hidden_state.shape == expected_shape, \
            f"Wrong output shape: {outputs.last_hidden_state.shape} != {expected_shape}"
    
    def test_deterministic_output(self, model_and_tokenizer, sample_inputs):
        """Test that output is deterministic (same input -> same output)."""
        model, _ = model_and_tokenizer
        
        with torch.no_grad():
            output1 = model(**sample_inputs)
            output2 = model(**sample_inputs)
        
        assert ModelTestUtils.compare_outputs(
            output1.last_hidden_state,
            output2.last_hidden_state
        ), "Outputs are not deterministic"
    
    def test_batch_inference(self, model_and_tokenizer):
        """Test inference with different batch sizes."""
        model, tokenizer = model_and_tokenizer
        
        for batch_size in [1, 2, 4]:
            inputs = ModelTestUtils.create_sample_text_inputs(
                tokenizer,
                texts=["test"] * batch_size
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            assert outputs.last_hidden_state.shape[0] == batch_size, \
                f"Wrong batch size in output: {outputs.last_hidden_state.shape[0]} != {batch_size}"


# Hardware Tests

@pytest.mark.hardware
@pytest.mark.cpu
class TestCPU:
    """Test model on CPU."""
    
    def test_cpu_inference(self, model_and_tokenizer, sample_inputs):
        """Test inference on CPU."""
        model, _ = model_and_tokenizer
        model = model.to("cpu")
        inputs = {k: v.to("cpu") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "cpu")


@pytest.mark.hardware
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Test model on CUDA."""
    
    def test_cuda_inference(self, model_and_tokenizer, sample_inputs):
        """Test inference on CUDA."""
        model, _ = model_and_tokenizer
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "cuda")
    
    def test_cuda_fp16(self, model_and_tokenizer, sample_inputs):
        """Test inference with FP16 on CUDA."""
        model, _ = model_and_tokenizer
        model = model.to("cuda").half()
        inputs = {k: v.to("cuda") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.last_hidden_state.dtype == torch.float16, \
            f"Output not FP16: {outputs.last_hidden_state.dtype}"
        ModelTestUtils.assert_tensor_valid(outputs.last_hidden_state, "FP16 output")


@pytest.mark.hardware
@pytest.mark.mps
@pytest.mark.skipif(
    not (hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()),
    reason="MPS not available"
)
class TestMPS:
    """Test model on Apple MPS."""
    
    def test_mps_inference(self, model_and_tokenizer, sample_inputs):
        """Test inference on MPS."""
        model, _ = model_and_tokenizer
        model = model.to("mps")
        inputs = {k: v.to("mps") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "mps")


# Performance Tests

@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformance:
    """Test model performance."""
    
    def test_inference_speed(self, model_and_tokenizer, sample_inputs):
        """Test and report inference speed."""
        model, _ = model_and_tokenizer
        
        timing_stats = ModelTestUtils.measure_inference_time(
            model, sample_inputs,
            warmup_runs=3,
            test_runs=10
        )
        
        # Print performance report
        report = PerformanceTestUtils.create_performance_report(
            MODEL_NAME, timing_stats
        )
        print(report)
        
        # Assert reasonable performance (adjust threshold as needed)
        assert timing_stats['mean'] < 1.0, \
            f"Inference too slow: {timing_stats['mean']:.4f}s"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage(self, model_and_tokenizer, sample_inputs):
        """Test and report memory usage."""
        model, _ = model_and_tokenizer
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in sample_inputs.items()}
        
        memory_stats = ModelTestUtils.measure_memory_usage(model, inputs, device="cuda")
        
        # Print memory report
        report = PerformanceTestUtils.create_performance_report(
            MODEL_NAME, {'mean': 0}, memory_stats
        )
        print(report)
        
        # Assert reasonable memory usage (adjust threshold as needed)
        assert memory_stats['peak_mb'] < 2000, \
            f"Memory usage too high: {memory_stats['peak_mb']:.2f}MB"


# Error Handling Tests

@pytest.mark.model_test
@pytest.mark.model
class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_input_raises_error(self, model_and_tokenizer):
        """Test that invalid input raises appropriate error."""
        model, _ = model_and_tokenizer
        
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            model(input_ids=None)
    
    def test_empty_input_handling(self, model_and_tokenizer, sample_inputs):
        """Test handling of empty inputs."""
        model, tokenizer = model_and_tokenizer
        
        # Create empty input
        empty_inputs = tokenizer("", return_tensors="pt", padding=True)
        
        # Should not crash
        with torch.no_grad():
            outputs = model(**empty_inputs)
        
        assert outputs is not None, "Model returned None for empty input"


# Integration Tests

@pytest.mark.integration
class TestIntegration:
    """Integration tests."""
    
    def test_pipeline_api(self):
        """Test model works with transformers pipeline API."""
        if not HAS_TRANSFORMERS:
            pytest.skip("transformers not available")
        
        pipeline = transformers.pipeline(TASK_TYPE, model=MODEL_ID)
        result = pipeline("This is a test")
        
        assert result is not None, "Pipeline returned None"
    
    def test_save_and_load(self, model_and_tokenizer, tmp_path):
        """Test model can be saved and loaded."""
        model, tokenizer = model_and_tokenizer
        
        # Save
        save_dir = tmp_path / "model"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Load
        loaded_model = transformers.AutoModel.from_pretrained(save_dir)
        loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(save_dir)
        
        # Verify loaded model works
        inputs = ModelTestUtils.create_sample_text_inputs(loaded_tokenizer)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        
        ModelTestUtils.assert_tensor_valid(outputs.last_hidden_state, "loaded model output")
