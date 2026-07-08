"""
Tests for multimodal models hardware-aware metrics integration.

This module tests the integration between multimodal models and hardware-aware metrics,
ensuring that modern multimodal architectures are properly supported.
"""

import os
import unittest
import sys
from pathlib import Path
import torch
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.multimodal_models import MultimodalModelAdapter, apply_multimodal_hardware_optimizations
from metrics.power import PowerMetric
from metrics.bandwidth import BandwidthMetric


class TestMultimodalHardwareMetrics(unittest.TestCase):
    """Test the hardware-aware metrics with multimodal models."""

    def setUp(self):
        """Set up test environment."""
        # Skip tests if no GPU available for more complex models
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.has_cuda else "cpu")
        
        # Create mock models and processors
        self.setup_mocks()

    def setup_mocks(self):
        """Set up mock objects for testing."""
        # Mock the AutoProcessor
        self.mock_processor = MagicMock()
        self.mock_processor.tokenizer = MagicMock()
        self.mock_processor.image_processor = MagicMock()
        
        # Mock various model types
        self.mock_llava_model = MagicMock()
        self.mock_blip2_model = MagicMock()
        self.mock_imagebind_model = MagicMock()
        self.mock_clip_model = MagicMock()
        
        # Mock for hardware metrics
        self.mock_power_metric = MagicMock(spec=PowerMetric)
        self.mock_bandwidth_metric = MagicMock(spec=BandwidthMetric)

    @patch("models.multimodal_models.LlavaProcessor.from_pretrained")
    @patch("models.multimodal_models.LlavaForConditionalGeneration.from_pretrained")
    def test_llava_model_hardware_optimizations(self, mock_model, mock_processor):
        """Test hardware optimizations for LLaVA models."""
        # Skip test if no GPU available
        if not self.has_cuda:
            self.skipTest("CUDA not available, skipping LLaVA test")
        
        # Setup mocks
        mock_processor.return_value = self.mock_processor
        mock_model.return_value = self.mock_llava_model
        
        # Create adapter
        adapter = MultimodalModelAdapter("llava-hf/llava-1.5-7b-hf", task="image-to-text")
        
        # Test that adapter correctly identifies model type
        self.assertTrue(adapter.is_llava)
        self.assertEqual(adapter.image_size, (336, 336))
        
        # Test model loading with hardware optimizations
        with patch("models.multimodal_models.apply_multimodal_hardware_optimizations") as mock_optimize:
            mock_optimize.return_value = self.mock_llava_model
            model = adapter.load_model(self.device, use_flash_attention=True, use_torch_compile=True)
            
            # Check that hardware optimizations were applied
            mock_optimize.assert_called_once()
            mock_optimize_args = mock_optimize.call_args[0]
            self.assertEqual(mock_optimize_args[0], self.mock_llava_model)
            self.assertEqual(mock_optimize_args[1], "cuda")
            self.assertTrue(mock_optimize.call_args[1]["use_flash_attention"])
            self.assertTrue(mock_optimize.call_args[1]["use_torch_compile"])

    @patch("models.multimodal_models.Blip2Processor.from_pretrained")
    @patch("models.multimodal_models.Blip2ForConditionalGeneration.from_pretrained")
    def test_blip2_model_hardware_optimizations(self, mock_model, mock_processor):
        """Test hardware optimizations for BLIP-2 models."""
        # Setup mocks
        mock_processor.return_value = self.mock_processor
        mock_model.return_value = self.mock_blip2_model
        
        # Create adapter
        adapter = MultimodalModelAdapter("Salesforce/blip2-opt-2.7b", task="image-to-text")
        
        # Test that adapter correctly identifies model type
        self.assertTrue(adapter.is_blip2)
        self.assertEqual(adapter.image_size, (224, 224))
        
        # Test model loading with hardware optimizations
        with patch("models.multimodal_models.apply_multimodal_hardware_optimizations") as mock_optimize:
            mock_optimize.return_value = self.mock_blip2_model
            model = adapter.load_model(self.device)
            
            # Check that hardware optimizations were applied
            mock_optimize.assert_called_once()
            mock_optimize_args = mock_optimize.call_args[0]
            self.assertEqual(mock_optimize_args[0], self.mock_blip2_model)
            self.assertEqual(mock_optimize_args[1], "cpu" if not self.has_cuda else "cuda")

    @patch("models.multimodal_models.CLIPProcessor.from_pretrained")
    @patch("models.multimodal_models.CLIPModel.from_pretrained")
    def test_clip_model_with_power_metrics(self, mock_model, mock_processor):
        """Test integration of CLIP models with power metrics."""
        # Setup mocks
        mock_processor.return_value = self.mock_processor
        mock_model.return_value = self.mock_clip_model
        
        # Create adapter
        adapter = MultimodalModelAdapter("openai/clip-vit-base-patch32", task="image-to-text")
        
        # Setup power metric
        with patch("metrics.power.PowerMetric") as mock_power_class:
            mock_power_class.return_value = self.mock_power_metric
            power_metric = PowerMetric(device=self.device)
            
            # Test power measurement
            with patch.object(adapter, "prepare_inputs") as mock_prepare:
                # Setup mock inputs
                mock_inputs = {
                    "pixel_values": torch.rand(2, 3, 224, 224),
                    "input_ids": torch.randint(0, 1000, (2, 77)),
                    "attention_mask": torch.ones(2, 77, dtype=torch.long)
                }
                mock_prepare.return_value = mock_inputs
                
                # Load model
                with patch("models.multimodal_models.apply_multimodal_hardware_optimizations", return_value=self.mock_clip_model):
                    model = adapter.load_model(self.device)
                
                # Start power measurement
                power_metric.start()
                
                # Simulate model forward pass
                inputs = adapter.prepare_inputs(batch_size=2)
                if self.has_cuda:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model(**inputs)
                
                # Stop power measurement
                power_metric.stop()
                
                # Check that power metric collected data
                self.mock_power_metric.start.assert_called_once()
                self.mock_power_metric.stop.assert_called_once()

    @patch("models.multimodal_models.AutoProcessor.from_pretrained")
    @patch("models.multimodal_models.VisionTextDualEncoderModel.from_pretrained")
    def test_model_with_bandwidth_metrics(self, mock_model, mock_processor):
        """Test integration of multimodal models with bandwidth metrics."""
        # Setup mocks
        mock_processor.return_value = self.mock_processor
        mock_model.return_value = MagicMock()
        
        # Create adapter
        adapter = MultimodalModelAdapter("clip-italian/clip-italian", task="image-to-text")
        
        # Setup bandwidth metric
        with patch("metrics.bandwidth.BandwidthMetric") as mock_bw_class:
            mock_bw_class.return_value = self.mock_bandwidth_metric
            bandwidth_metric = BandwidthMetric(device=self.device)
            
            # Test bandwidth measurement
            with patch.object(adapter, "prepare_inputs") as mock_prepare:
                # Setup mock inputs
                mock_inputs = {
                    "pixel_values": torch.rand(2, 3, 224, 224),
                    "input_ids": torch.randint(0, 1000, (2, 77)),
                    "attention_mask": torch.ones(2, 77, dtype=torch.long)
                }
                mock_prepare.return_value = mock_inputs
                
                # Load model
                with patch("models.multimodal_models.apply_multimodal_hardware_optimizations", return_value=MagicMock()):
                    model = adapter.load_model(self.device)
                
                # Start bandwidth measurement
                bandwidth_metric.start()
                
                # Simulate model forward pass
                inputs = adapter.prepare_inputs(batch_size=2)
                if self.has_cuda:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model(**inputs)
                
                # Stop bandwidth measurement
                bandwidth_metric.stop()
                
                # Check that bandwidth metric collected data
                self.mock_bandwidth_metric.start.assert_called_once()
                self.mock_bandwidth_metric.stop.assert_called_once()
    
    def test_hardware_optimization_function(self):
        """Test the apply_multimodal_hardware_optimizations function."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        
        # Test CPU optimizations
        with patch("torch.set_num_threads") as mock_set_threads:
            with patch("torch.backends.mkldnn.enabled", create=True, new=False):
                # Apply optimizations
                optimized_model = apply_multimodal_hardware_optimizations(
                    model, 
                    device_type="cpu",
                    use_flash_attention=False,
                    use_torch_compile=False
                )
                
                # Check that optimizations were applied
                mock_set_threads.assert_called_once()
                
                # Check that model wasn't modified (since no CUDA or compile)
                self.assertEqual(optimized_model, model)
        
        # Test CUDA optimizations if available
        if self.has_cuda:
            with patch("torch.cuda.current_stream") as mock_stream:
                mock_stream_instance = MagicMock()
                mock_stream.return_value = mock_stream_instance
                
                # Apply optimizations
                optimized_model = apply_multimodal_hardware_optimizations(
                    model, 
                    device_type="cuda",
                    use_flash_attention=False,
                    use_torch_compile=False
                )
                
                # Check that optimizations were applied
                mock_stream.assert_called_once()
                self.assertEqual(mock_stream_instance.priority, -1)
                
                # Check that model wasn't modified (since no flash attention or compile)
                self.assertEqual(optimized_model, model)
        
        # Test torch.compile if available
        if hasattr(torch, "compile"):
            with patch("torch.compile") as mock_compile:
                mock_compile.return_value = model  # Return same model for simplicity
                
                # Apply optimizations
                optimized_model = apply_multimodal_hardware_optimizations(
                    model, 
                    device_type="cpu",
                    use_flash_attention=False,
                    use_torch_compile=True
                )
                
                # Check that optimizations were applied
                mock_compile.assert_called_once_with(model)

    def test_model_detection(self):
        """Test model type detection logic."""
        # Test LLaVA detection
        adapter = MultimodalModelAdapter("llava-hf/llava-1.5-7b-hf")
        self.assertTrue(adapter.is_llava)
        self.assertFalse(adapter.is_blip2)
        
        # Test BLIP-2 detection
        adapter = MultimodalModelAdapter("Salesforce/blip2-opt-2.7b")
        self.assertTrue(adapter.is_blip2)
        self.assertFalse(adapter.is_llava)
        
        # Test InstructBLIP detection
        adapter = MultimodalModelAdapter("Salesforce/instructblip-vicuna-7b")
        self.assertTrue(adapter.is_instructblip)
        self.assertFalse(adapter.is_blip2)
        
        # Test CLIP detection
        adapter = MultimodalModelAdapter("openai/clip-vit-base-patch32")
        self.assertTrue(adapter.is_clip)
        self.assertFalse(adapter.is_blip)
        
        # Test normal BLIP detection (not BLIP-2)
        adapter = MultimodalModelAdapter("Salesforce/blip-image-captioning-large")
        self.assertTrue(adapter.is_blip)
        self.assertFalse(adapter.is_blip2)
        
        # Test ImageBind detection
        adapter = MultimodalModelAdapter("facebook/imagebind-huge")
        self.assertTrue(adapter.is_imagebind)
        
        # Test GIT detection
        adapter = MultimodalModelAdapter("microsoft/git-base")
        self.assertTrue(adapter.is_git)
        
        # Test Pix2Struct detection
        adapter = MultimodalModelAdapter("google/pix2struct-base")
        self.assertTrue(adapter.is_pix2struct)

    def test_prepare_inputs_methods(self):
        """Test input preparation methods for different model types."""
        # Skip most tests if no GPU available to save time
        if not self.has_cuda:
            self.skipTest("CUDA not available, skipping detailed input preparation tests")
        
        # Test with different model types
        test_models = [
            ("llava-hf/llava-1.5-7b-hf", "_prepare_llava_inputs"),
            ("Salesforce/blip2-opt-2.7b", "_prepare_blip2_inputs"),
            ("facebook/imagebind-huge", "_prepare_imagebind_inputs"),
            ("Salesforce/instructblip-vicuna-7b", "_prepare_instructblip_inputs"),
            ("google/pix2struct-base", "_prepare_pix2struct_inputs"),
            ("google/videomae-base", "_prepare_video_inputs"),
            ("microsoft/layoutlm-base-uncased", "_prepare_document_inputs"),
        ]
        
        for model_id, method_name in test_models:
            # Create adapter
            adapter = MultimodalModelAdapter(model_id)
            
            # Mock the processor and method
            adapter.processor = self.mock_processor
            
            # Mock the specific preparation method
            with patch.object(adapter, method_name) as mock_prepare_method:
                mock_inputs = {
                    "pixel_values": torch.rand(2, 3, 224, 224),
                    "input_ids": torch.randint(0, 1000, (2, 77)),
                }
                mock_prepare_method.return_value = mock_inputs
                
                # Call prepare_inputs
                inputs = adapter.prepare_inputs(batch_size=2)
                
                # Check that the correct method was called
                mock_prepare_method.assert_called_once_with(2)
                
                # Check that the inputs were returned
                self.assertEqual(inputs, mock_inputs)

    def test_dummy_multimodal_model(self):
        """Test the dummy multimodal model creation."""
        # Create adapter with invalid model ID to force dummy model creation
        adapter = MultimodalModelAdapter("nonexistent-model")
        
        # Mock the loading process to force dummy model creation
        with patch("models.multimodal_models.AutoProcessor.from_pretrained", side_effect=Exception("Not found")):
            with patch("models.multimodal_models.AutoModel.from_pretrained", side_effect=Exception("Not found")):
                with patch("models.multimodal_models.apply_multimodal_hardware_optimizations", return_value=Mock()):
                    # Load model (should create dummy model)
                    model = adapter.load_model(self.device)
                    
                    # Check that a model was returned
                    self.assertIsNotNone(model)
                    
                    # Test dummy model forward pass
                    batch_size = 2
                    inputs = {
                        "pixel_values": torch.rand(batch_size, 3, 224, 224),
                        "input_ids": torch.randint(0, 1000, (batch_size, 77)),
                        "attention_mask": torch.ones(batch_size, 77, dtype=torch.long)
                    }
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Call model
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Check outputs
                    self.assertIn("logits", outputs)
                    self.assertEqual(outputs["logits"].shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()