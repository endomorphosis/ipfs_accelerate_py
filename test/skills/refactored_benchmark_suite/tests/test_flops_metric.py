#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for FLOPs metric.

This script tests the FLOPs metric and factory with different device types
and validates hardware-aware optimizations.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch import nn
from metrics.flops import FLOPsMetric, FLOPsMetricFactory
import hardware

class SimpleModel(nn.Module):
    """Simple model for testing FLOPs estimation."""
    
    def __init__(self, has_attention=False):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
        # Add attention for testing transformer detection
        if has_attention:
            self.attention = nn.MultiheadAttention(256, num_heads=8)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        if hasattr(self, 'attention'):
            # Reshape for attention (sequence_length, batch_size, hidden_size)
            seq_len = x.size(0)
            batch_size = x.size(0)
            x = x.unsqueeze(0).expand(seq_len, batch_size, -1)
            x, _ = self.attention(x, x, x)
            x = x.mean(dim=0)  # Average over sequence dimension
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SimpleCNN(nn.Module):
    """Simple CNN for testing FLOPs estimation."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleTransformer(nn.Module):
    """Simple transformer model with configurable attention type."""
    
    def __init__(self, attention_type="mha", hidden_size=256, num_heads=8, layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attention_type = attention_type
        
        # Create a config attribute to mimic transformer models
        self.config = type("TransformerConfig", (), {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_hidden_layers": layers,
        })
        
        # Add KV heads for MQA/GQA detection
        if attention_type == "mqa":
            self.config.kv_heads = 1
        elif attention_type == "gqa":
            self.config.kv_heads = num_heads // 4
        else:
            self.config.kv_heads = num_heads
        
        # Embeddings
        self.embeddings = nn.Embedding(10000, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        # Encoder layers with attention
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            for _ in range(layers)
        ])
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input_ids):
        # Create position IDs
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Pass through encoder layers
        x = embeddings
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Output projection
        x = self.output(x)
        return x

class MultimodalModel(nn.Module):
    """Simple multimodal model that combines image and text features."""
    
    def __init__(self):
        super().__init__()
        # Image encoder (simplified CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 256)
        )
        
        # Text encoder (simplified transformer)
        self.text_model = SimpleTransformer(hidden_size=256, num_heads=4, layers=2)
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Output projection
        self.output = nn.Linear(128, 10)
    
    def forward(self, pixel_values, input_ids):
        # Image processing
        image_features = self.image_encoder(pixel_values)
        
        # Text processing
        text_features = self.text_model(input_ids)[:, 0]  # Use [CLS] token
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        fused = self.fusion(combined)
        
        # Output projection
        output = self.output(fused)
        return output

class TestFLOPsMetric(unittest.TestCase):
    """Test cases for FLOPs metric."""
    
    def test_flops_simple_model(self):
        """Test FLOPs estimation for a simple model."""
        # Create model and inputs
        model = SimpleModel()
        inputs = torch.randn(16, 128)
        
        # Create FLOPs metric
        metric = FLOPsMetric()
        metric.set_model_and_inputs(model, inputs)
        
        # Test metric
        metric.start()
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertIn("flops", metrics)
        self.assertIn("gflops", metrics)
        self.assertIn("tflops", metrics)
        self.assertIn("total_parameters", metrics)
        self.assertIn("trainable_parameters", metrics)
        
        # Check that FLOPs are positive
        self.assertGreater(metrics["flops"], 0)
        
        # Check that parameter counts are correct
        expected_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(metrics["total_parameters"], expected_params)
    
    def test_model_type_detection(self):
        """Test model type detection."""
        # Test simple model (default type)
        simple_model = SimpleModel()
        simple_inputs = torch.randn(16, 128)
        
        simple_metric = FLOPsMetric()
        simple_metric.set_model_and_inputs(simple_model, simple_inputs)
        self.assertEqual(simple_metric._detect_model_type(), "unknown")
        
        # Test transformer model (with attention)
        transformer_model = SimpleModel(has_attention=True)
        transformer_inputs = torch.randn(16, 128)
        
        transformer_metric = FLOPsMetric()
        transformer_metric.set_model_and_inputs(transformer_model, transformer_inputs)
        self.assertEqual(transformer_metric._detect_model_type(), "transformer")
        
        # Test CNN model
        cnn_model = SimpleCNN()
        cnn_inputs = torch.randn(16, 3, 32, 32)
        
        cnn_metric = FLOPsMetric()
        cnn_metric.set_model_and_inputs(cnn_model, cnn_inputs)
        self.assertEqual(cnn_metric._detect_model_type(), "cnn")
        
        # Test transformer model with config
        transformer_model = SimpleTransformer()
        transformer_inputs = torch.ones(2, 16, dtype=torch.long)
        
        transformer_metric = FLOPsMetric()
        transformer_metric.set_model_and_inputs(transformer_model, transformer_inputs)
        self.assertEqual(transformer_metric._detect_model_type(), "transformer")
        
        # Test multimodal model
        multimodal_model = MultimodalModel()
        image_input = torch.randn(2, 3, 32, 32)
        text_input = torch.ones(2, 16, dtype=torch.long)
        multimodal_inputs = {"pixel_values": image_input, "input_ids": text_input}
        
        multimodal_metric = FLOPsMetric()
        multimodal_metric.set_model_and_inputs(multimodal_model, multimodal_inputs)
        self.assertEqual(multimodal_metric._detect_model_type(), "multimodal")
    
    def test_sequence_length_detection(self):
        """Test sequence length detection from inputs."""
        # Create FLOPs metric
        metric = FLOPsMetric()
        
        # Test with attention_mask
        inputs_with_mask = {
            "input_ids": torch.ones(2, 10),
            "attention_mask": torch.ones(2, 10)
        }
        metric.inputs = inputs_with_mask
        self.assertEqual(metric._get_sequence_length(), 10)
        
        # Test with input_ids only
        inputs_with_ids = {
            "input_ids": torch.ones(2, 15)
        }
        metric.inputs = inputs_with_ids
        self.assertEqual(metric._get_sequence_length(), 15)
        
        # Test with tensor input
        tensor_input = torch.ones(2, 20)
        metric.inputs = tensor_input
        self.assertEqual(metric._get_sequence_length(), 20)
    
    def test_attention_type_detection(self):
        """Test detection of different attention types (MHA, MQA, GQA)."""
        # Test standard multi-head attention
        mha_model = SimpleTransformer(attention_type="mha")
        mha_inputs = torch.ones(2, 16, dtype=torch.long)
        
        mha_metric = FLOPsMetric()
        mha_metric.set_model_and_inputs(mha_model, mha_inputs)
        self.assertEqual(mha_metric._detect_model_type(), "transformer")
        
        # Manually compute attention FLOPs
        seq_length = 16
        hidden_size = 256
        num_heads = 8
        attention_flops = mha_metric._estimate_attention_flops(
            seq_length, hidden_size, num_heads
        )
        
        # Test multi-query attention
        mqa_model = SimpleTransformer(attention_type="mqa")
        mqa_inputs = torch.ones(2, 16, dtype=torch.long)
        
        mqa_metric = FLOPsMetric()
        mqa_metric.set_model_and_inputs(mqa_model, mqa_inputs)
        
        # Get MQA attention FLOPs
        mqa_attention_flops = mqa_metric._estimate_attention_flops(
            seq_length, hidden_size, num_heads
        )
        
        # MQA should use fewer FLOPs than MHA
        self.assertLess(mqa_attention_flops, attention_flops)
        
        # Test grouped-query attention
        gqa_model = SimpleTransformer(attention_type="gqa")
        gqa_inputs = torch.ones(2, 16, dtype=torch.long)
        
        gqa_metric = FLOPsMetric()
        gqa_metric.set_model_and_inputs(gqa_model, gqa_inputs)
        
        # Get GQA attention FLOPs
        gqa_attention_flops = gqa_metric._estimate_attention_flops(
            seq_length, hidden_size, num_heads
        )
        
        # GQA should use fewer FLOPs than MHA but more than MQA
        self.assertLess(gqa_attention_flops, attention_flops)
        self.assertGreater(gqa_attention_flops, mqa_attention_flops)
    
    def test_hardware_efficiency_factors(self):
        """Test hardware-specific efficiency factors."""
        model = SimpleModel()
        inputs = torch.randn(16, 128)
        
        # CPU metric
        cpu_metric = FLOPsMetric(device_type="cpu")
        cpu_metric.set_model_and_inputs(model, inputs)
        cpu_efficiency = cpu_metric._get_hardware_efficiency_factor()
        
        # GPU metric (if available)
        if torch.cuda.is_available():
            gpu_metric = FLOPsMetric(device_type="cuda")
            gpu_metric.set_model_and_inputs(model, inputs)
            gpu_efficiency = gpu_metric._get_hardware_efficiency_factor()
            
            # GPU should be more efficient than CPU for numeric operations
            self.assertLess(gpu_efficiency, cpu_efficiency)
    
    def test_detailed_metrics(self):
        """Test detailed metrics."""
        # Create model and inputs
        model = SimpleModel()
        inputs = torch.randn(16, 128)
        
        # Create FLOPs metric
        metric = FLOPsMetric()
        metric.set_model_and_inputs(model, inputs)
        
        # Force simple estimation to generate detailed metrics
        metric._estimate_flops_simple()
        
        # Get detailed metrics
        detailed_metrics = metric.get_detailed_metrics()
        
        # Validate detailed metrics
        self.assertIn("detailed_flops", detailed_metrics)
        self.assertIn("parameters", detailed_metrics["detailed_flops"])
        
        # Test transformer detailed metrics
        transformer_model = SimpleTransformer()
        transformer_inputs = torch.ones(2, 16, dtype=torch.long)
        
        transformer_metric = FLOPsMetric()
        transformer_metric.set_model_and_inputs(transformer_model, transformer_inputs)
        transformer_metric._estimate_flops_simple()
        
        transformer_detailed = transformer_metric.get_detailed_metrics()
        self.assertIn("detailed_flops", transformer_detailed)
        
        # Transformer should have component-specific breakdowns
        # Check for attention and feed_forward components
        self.assertIn("attention", transformer_detailed["detailed_flops"])
        self.assertIn("feed_forward", transformer_detailed["detailed_flops"])
        
        # Test CNN detailed metrics
        cnn_model = SimpleCNN()
        cnn_inputs = torch.randn(2, 3, 32, 32)
        
        cnn_metric = FLOPsMetric()
        cnn_metric.set_model_and_inputs(cnn_model, cnn_inputs)
        cnn_metric._estimate_flops_simple()
        
        cnn_detailed = cnn_metric.get_detailed_metrics()
        self.assertIn("detailed_flops", cnn_detailed)
        
        # CNN should have component-specific breakdowns
        # Check for convolution and pooling components
        self.assertIn("convolution", cnn_detailed["detailed_flops"])
        self.assertIn("pooling", cnn_detailed["detailed_flops"])
        
        # Test multimodal detailed metrics
        multimodal_model = MultimodalModel()
        image_input = torch.randn(2, 3, 32, 32)
        text_input = torch.ones(2, 16, dtype=torch.long)
        multimodal_inputs = {"pixel_values": image_input, "input_ids": text_input}
        
        multimodal_metric = FLOPsMetric()
        multimodal_metric.set_model_and_inputs(multimodal_model, multimodal_inputs)
        multimodal_metric._estimate_flops_simple()
        
        multimodal_detailed = multimodal_metric.get_detailed_metrics()
        self.assertIn("detailed_flops", multimodal_detailed)
        
        # Multimodal should have encoder-specific breakdowns
        self.assertIn("image_encoder", multimodal_detailed["detailed_flops"])
        self.assertIn("text_encoder", multimodal_detailed["detailed_flops"])
        self.assertIn("fusion", multimodal_detailed["detailed_flops"])
    
    def test_precision_detection(self):
        """Test detection of model precision (fp32, fp16, etc.)"""
        # Create model in default precision (fp32)
        fp32_model = SimpleModel()
        inputs = torch.randn(16, 128)
        
        fp32_metric = FLOPsMetric()
        fp32_metric.set_model_and_inputs(fp32_model, inputs)
        fp32_metric._estimate_flops_simple()
        
        fp32_metrics = fp32_metric.get_detailed_metrics()
        precision_info = fp32_metric._get_model_precision_info()
        
        # Should detect dominant precision
        self.assertIn("dominant_precision", precision_info)
        # For standard PyTorch models, default is float32
        self.assertEqual(precision_info["dominant_precision"], "float32")
        
        # Create model in fp16 precision if available
        if torch.cuda.is_available():
            try:
                # Convert model to CUDA and fp16
                fp16_model = SimpleModel().to("cuda").half()
                fp16_inputs = torch.randn(16, 128, device="cuda", dtype=torch.float16)
                
                fp16_metric = FLOPsMetric("cuda")
                fp16_metric.set_model_and_inputs(fp16_model, fp16_inputs)
                fp16_metric._estimate_flops_simple()
                
                fp16_precision_info = fp16_metric._get_model_precision_info()
                self.assertEqual(fp16_precision_info["dominant_precision"], "float16")
                
                # Should detect tensor core eligibility
                self.assertTrue(fp16_metric._has_tensor_core_operations())
            except RuntimeError:
                # Skip if CUDA fp16 not supported
                pass
    
    def test_factory_with_torch_device(self):
        """Test FLOPs metric factory with torch.device."""
        # Create metric with torch.device
        cpu_device = torch.device("cpu")
        metric = FLOPsMetricFactory.create(cpu_device)
        
        # Validate device type
        self.assertEqual(metric.device_type, "cpu")
        
        # Create metric with CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            metric = FLOPsMetricFactory.create(cuda_device)
            
            # Validate device type
            self.assertEqual(metric.device_type, "cuda")
    
    def test_factory_with_hardware_backend(self):
        """Test FLOPs metric factory with hardware backend device."""
        # Create metric with CPU hardware backend
        cpu_backend = hardware.get_hardware_backend("cpu")
        cpu_device = cpu_backend.initialize()
        metric = FLOPsMetricFactory.create(cpu_device)
        
        # Validate device type
        self.assertEqual(metric.device_type, "cpu")
        
        # Create metric with CUDA hardware backend if available
        if "cuda" in hardware.get_available_hardware():
            cuda_backend = hardware.get_hardware_backend("cuda")
            cuda_device = cuda_backend.initialize()
            metric = FLOPsMetricFactory.create(cuda_device)
            
            # Validate device type
            self.assertEqual(metric.device_type, "cuda")

if __name__ == "__main__":
    unittest.main()