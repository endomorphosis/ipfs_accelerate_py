#!/usr/bin/env python3
"""
Improved pytest-compatible test for BERT model.

This demonstrates the improved testing approach with proper pytest structure,
assertions, hardware testing, and performance benchmarks.
"""

import pytest
import torch
from typing import Dict, Any

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
    HAS_TRANSFORMERS = False
    pytest.skip("transformers not available", allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("torch not available", allow_module_level=True)


# Model configuration
MODEL_ID = "bert-base-uncased"
MODEL_NAME = "BERT"


# Fixtures

@pytest.fixture(scope="module")
def bert_model_and_tokenizer():
    """Load BERT model and tokenizer (cached for module)."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer

@pytest.fixture
def pytest_config(request):
    """Get pytest configuration."""
    return request.config

@pytest.fixture
def sample_inputs(bert_model_and_tokenizer):
    """Create sample inputs for testing."""
    _, tokenizer = bert_model_and_tokenizer
    return ModelTestUtils.create_sample_text_inputs(
        tokenizer,
        texts=[
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you today?",
        ]
    )


# Basic Tests

@pytest.mark.model_test
@pytest.mark.model
@pytest.mark.text
class TestBERTLoading:
    """Test BERT model loading and initialization."""
    
    def test_model_loads(self, bert_model_and_tokenizer):
        """Test that BERT model loads successfully."""
        model, tokenizer = bert_model_and_tokenizer
        ModelTestUtils.assert_model_loaded(model, MODEL_NAME)
        ModelTestUtils.assert_tokenizer_loaded(tokenizer, MODEL_NAME)
    
    def test_model_config(self, bert_model_and_tokenizer):
        """Test BERT model configuration."""
        model, _ = bert_model_and_tokenizer
        config = model.config
        
        assert config.hidden_size == 768, f"Wrong hidden size: {config.hidden_size}"
        assert config.num_hidden_layers == 12, f"Wrong num layers: {config.num_hidden_layers}"
        assert config.num_attention_heads == 12, f"Wrong num attention heads: {config.num_attention_heads}"
        assert config.vocab_size == 30522, f"Wrong vocab size: {config.vocab_size}"


@pytest.mark.model_test
@pytest.mark.model
@pytest.mark.text
class TestBERTInference:
    """Test BERT model inference."""
    
    def test_forward_pass(self, bert_model_and_tokenizer, sample_inputs):
        """Test BERT forward pass."""
        model, _ = bert_model_and_tokenizer
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None, "Model output is None"
        assert hasattr(outputs, 'last_hidden_state'), "Output missing last_hidden_state"
        ModelTestUtils.assert_tensor_valid(outputs.last_hidden_state, "last_hidden_state")
    
    def test_output_shape(self, bert_model_and_tokenizer, sample_inputs):
        """Test BERT output shape is correct."""
        model, _ = bert_model_and_tokenizer
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        batch_size = sample_inputs['input_ids'].shape[0]
        seq_length = sample_inputs['input_ids'].shape[1]
        hidden_size = 768
        
        expected_shape = (batch_size, seq_length, hidden_size)
        assert outputs.last_hidden_state.shape == expected_shape, \
            f"Wrong shape: {outputs.last_hidden_state.shape} != {expected_shape}"


# Hardware Tests

@pytest.mark.model_test
@pytest.mark.hardware
@pytest.mark.cpu
class TestBERTCPU:
    """Test BERT on CPU."""
    
    def test_cpu_inference(self, bert_model_and_tokenizer, sample_inputs):
        """Test BERT inference on CPU."""
        model, _ = bert_model_and_tokenizer
        model = model.to("cpu")
        inputs = {k: v.to("cpu") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "cpu")


@pytest.mark.model_test
@pytest.mark.hardware
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBERTCUDA:
    """Test BERT on CUDA."""
    
    def test_cuda_inference(self, bert_model_and_tokenizer, sample_inputs):
        """Test BERT inference on CUDA."""
        model, _ = bert_model_and_tokenizer
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in sample_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "cuda")


# Performance Tests with Regression Detection

@pytest.mark.model_test
@pytest.mark.benchmark
@pytest.mark.slow
class TestBERTPerformance:
    """Test BERT performance with regression detection."""
    
    def test_cpu_performance_with_baseline(self, bert_model_and_tokenizer, sample_inputs, pytest_config):
        """Test CPU performance and check for regressions."""
        model, _ = bert_model_and_tokenizer
        model = model.to("cpu")
        inputs = {k: v.to("cpu") for k, v in sample_inputs.items()}
        
        # Measure performance
        timing_stats = ModelTestUtils.measure_inference_time(
            model, inputs, warmup_runs=2, test_runs=5
        )
        
        # Get config options
        update_baseline = getattr(pytest_config, 'update_baselines', False)
        tolerance = getattr(pytest_config, 'baseline_tolerance', 0.20)
        
        # Check for regressions or update baseline
        result = PerformanceTestUtils.check_performance_regression(
            model_name=MODEL_NAME,
            timing_stats=timing_stats,
            device="cpu",
            tolerance=tolerance,
            update_baseline=update_baseline
        )
        
        # Print results
        print(f"\n{MODEL_NAME} CPU Performance:")
        print(f"  Mean inference time: {timing_stats['mean']*1000:.2f}ms")
        if result.get('has_baseline'):
            print(f"  {result.get('message', '')}")
        
        # Log warning if regressions detected (don't fail test)
        if result.get('regressions'):
            import warnings
            warnings.warn(f"Performance regressions detected for {MODEL_NAME}")

