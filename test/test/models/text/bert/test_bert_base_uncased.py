"""
Test file for bert-base-uncased model.

This file contains tests for the bert-base-uncased model,
including model loading, inference, and CUDA support.
Generated from ModelTestTemplate.
"""

import os
import pytest
import logging
from typing import Dict, List, Any, Optional

# Import common utilities
from common.hardware_detection import detect_hardware, skip_if_no_cuda
from common.model_helpers import load_model, get_sample_inputs_for_model

# BERT-specific imports
try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    pass

# Import PyTorch for tensor operations
try:
    import torch
except ImportError:
    pass

# Model-specific fixtures
@pytest.fixture(scope='module')
def bert_base_uncased_model():
    """Load bert-base-uncased model for testing."""
    model = load_model("bert-base-uncased", framework="transformers")
    yield model

@pytest.fixture
def bert_base_uncased_model_cuda(cuda_available):
    """Load bert-base-uncased model on CUDA for testing."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    model = load_model("bert-base-uncased", framework="transformers", platform="cuda")
    yield model

@pytest.fixture
def bert_base_uncased_inputs(bert_base_uncased_model):
    """Generate sample inputs for bert-base-uncased model."""
    return get_sample_inputs_for_model("bert-base-uncased", batch_size=2)

class TestBertBaseUncased:
    """
    Tests for bert-base-uncased model.
    """
    
    def test_model_loading(self, bert_base_uncased_model):
        """Test bert-base-uncased model loading."""
        assert bert_base_uncased_model is not None
    
    def test_model_inference(self, bert_base_uncased_model, bert_base_uncased_inputs):
        """Test bert-base-uncased model inference."""
        try:
            outputs = bert_base_uncased_model(**bert_base_uncased_inputs)
            assert outputs is not None
        except Exception as e:
            pytest.fail(f"Model inference failed: {e}")
    
    @skip_if_no_cuda
    def test_model_cuda(self, bert_base_uncased_model_cuda, bert_base_uncased_inputs):
        """Test bert-base-uncased model on CUDA."""
        try:
            # Move inputs to CUDA
            cuda_inputs = {}
            for k, v in bert_base_uncased_inputs.items():
                if hasattr(v, 'to'):
                    cuda_inputs[k] = v.to('cuda')
                else:
                    cuda_inputs[k] = v
            
            outputs = bert_base_uncased_model_cuda(**cuda_inputs)
            assert outputs is not None
            
            if hasattr(outputs, 'last_hidden_state'):
                assert outputs.last_hidden_state.device.type == 'cuda'
        except Exception as e:
            pytest.fail(f"CUDA inference failed: {e}")
    
    def test_model_batch_size(self, bert_base_uncased_model):
        """Test bert-base-uncased model with different batch sizes."""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            inputs = get_sample_inputs_for_model("bert-base-uncased", batch_size=batch_size)
            outputs = bert_base_uncased_model(**inputs)
            assert outputs is not None
    
    def test_model_sequence_length(self, bert_base_uncased_model):
        """Test bert-base-uncased model with different sequence lengths."""
        sequence_lengths = [16, 32, 64]
        for seq_len in sequence_lengths:
            inputs = get_sample_inputs_for_model("bert-base-uncased", sequence_length=seq_len)
            outputs = bert_base_uncased_model(**inputs)
            assert outputs is not None