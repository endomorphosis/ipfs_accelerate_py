"""
FLOPs (Floating Point Operations) metrics for model benchmarking.

This module provides metrics for estimating the number of floating point operations
required for model inference across different hardware platforms.
"""

import logging
from typing import Dict, Any, Optional, Union, List

import torch
import torch.nn as nn

logger = logging.getLogger("benchmark.metrics.flops")

class FLOPsMetric:
    """
    Metric for estimating FLOPs used by the model.
    
    Tracks the number of floating point operations required for a forward pass of the model.
    Supports various hardware platforms and model types.
    """
    
    def __init__(self, device_type: str = "cpu"):
        """
        Initialize the FLOPs metric.
        
        Args:
            device_type: Type of device being benchmarked
        """
        self.device_type = device_type
        self.model = None
        self.inputs = None
        self.flops_count = 0
        self.detailed_flops = {}
        self.has_estimated = False
        
        # Check for optional dependencies
        try:
            from fvcore.nn import FlopCountAnalysis
            self.flop_counter_cls = FlopCountAnalysis
            self.has_fvcore = True
        except ImportError:
            self.has_fvcore = False
            logger.warning("fvcore not available, FLOPs estimation will use fallback method")
            
        # Check if hardware-specific profilers are available
        self.hardware_profilers = self._check_hardware_profilers()
    
    def _check_hardware_profilers(self) -> Dict[str, bool]:
        """
        Check if hardware-specific profilers are available.
        
        Returns:
            Dictionary of available profilers
        """
        profilers = {
            "cuda_profiler": False,
            "xla_profiler": False,
            "openvino_profiler": False
        }
        
        # Check for CUDA profiler
        if self.device_type == "cuda":
            try:
                import torch.cuda.profiler as cuda_profiler
                profilers["cuda_profiler"] = True
            except ImportError:
                pass
        
        # Check for XLA profiler (TPU)
        if self.device_type == "xla":
            try:
                import torch_xla.debug.profiler as xla_profiler
                profilers["xla_profiler"] = True
            except ImportError:
                pass
        
        # Check for OpenVINO profiler
        if self.device_type == "openvino":
            try:
                from openvino.tools import benchmark_tool
                profilers["openvino_profiler"] = True
            except ImportError:
                pass
        
        return profilers
    
    def set_model_and_inputs(self, model: nn.Module, inputs: Any):
        """
        Set the model and inputs for FLOPs estimation.
        
        Args:
            model: PyTorch model to measure
            inputs: Model inputs
        """
        self.model = model
        self.inputs = inputs
        self.has_estimated = False
    
    def start(self):
        """Start measuring FLOPs."""
        # Nothing to do here, FLOPs are calculated at the end
        pass
    
    def stop(self):
        """Stop measuring FLOPs and calculate the total."""
        if self.model is None or self.inputs is None:
            logger.warning("Model or inputs not provided, cannot estimate FLOPs")
            return
        
        # Only estimate once
        if self.has_estimated:
            return
        
        try:
            # Try hardware-specific methods first
            if self.hardware_profilers.get("cuda_profiler") and self.device_type == "cuda":
                self._estimate_flops_cuda()
            elif self.hardware_profilers.get("xla_profiler") and self.device_type == "xla":
                self._estimate_flops_xla()
            elif self.hardware_profilers.get("openvino_profiler") and self.device_type == "openvino":
                self._estimate_flops_openvino()
            # Then try fvcore if available
            elif self.has_fvcore:
                self._estimate_flops_fvcore()
            # Fall back to simple estimation
            else:
                self._estimate_flops_simple()
            
            self.has_estimated = True
            
        except Exception as e:
            logger.error(f"Error estimating FLOPs: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Try fallback method
            if not self.has_estimated:
                try:
                    logger.warning("Trying fallback FLOPs estimation method")
                    self._estimate_flops_simple()
                    self.has_estimated = True
                except Exception as e2:
                    logger.error(f"Error in fallback FLOPs estimation: {e2}")
    
    def _estimate_flops_cuda(self):
        """Estimate FLOPs using CUDA profiler."""
        logger.info("Estimating FLOPs using CUDA profiler")
        
        try:
            import torch.cuda.profiler as cuda_profiler
            
            # Clear cache for more accurate measurements
            torch.cuda.empty_cache()
            
            # Run with profiler
            with torch.no_grad():
                cuda_profiler.start()
                
                # Run inference
                if isinstance(self.inputs, dict):
                    self.model(**self.inputs)
                else:
                    self.model(self.inputs)
                
                cuda_profiler.stop()
            
            # CUDA profiler doesn't directly provide FLOPs
            # This is just a placeholder - in a real implementation,
            # you would parse the profiler output to estimate FLOPs
            self._estimate_flops_simple()
            
        except Exception as e:
            logger.error(f"Error using CUDA profiler: {e}")
            # Fall back to simpler method
            self._estimate_flops_simple()
    
    def _estimate_flops_xla(self):
        """Estimate FLOPs using XLA profiler (for TPUs)."""
        logger.info("Estimating FLOPs using XLA profiler")
        
        try:
            import torch_xla.debug.profiler as xla_profiler
            import torch_xla.core.xla_model as xm
            
            # Run with profiler
            with torch.no_grad(), xla_profiler.trace("flops_estimation"):
                # Run inference
                if isinstance(self.inputs, dict):
                    self.model(**self.inputs)
                else:
                    self.model(self.inputs)
                
                xm.mark_step()
            
            # XLA profiler doesn't directly provide FLOPs
            # This is just a placeholder - in a real implementation,
            # you would parse the profiler output to estimate FLOPs
            self._estimate_flops_simple()
            
        except Exception as e:
            logger.error(f"Error using XLA profiler: {e}")
            # Fall back to simpler method
            self._estimate_flops_simple()
    
    def _estimate_flops_openvino(self):
        """Estimate FLOPs using OpenVINO profiler."""
        logger.info("Estimating FLOPs using OpenVINO profiler")
        
        try:
            # OpenVINO benchmark tool doesn't directly provide FLOPs
            # This is just a placeholder - in a real implementation,
            # you would use OpenVINO API to estimate operations
            self._estimate_flops_simple()
            
        except Exception as e:
            logger.error(f"Error using OpenVINO profiler: {e}")
            # Fall back to simpler method
            self._estimate_flops_simple()
    
    def _estimate_flops_fvcore(self):
        """Estimate FLOPs using fvcore."""
        logger.info("Estimating FLOPs using fvcore")
        
        try:
            # Create inputs in the right format for fvcore
            if isinstance(self.inputs, dict):
                # Try to get the first tensor for fvcore
                first_tensor = None
                for value in self.inputs.values():
                    if isinstance(value, torch.Tensor):
                        first_tensor = value
                        break
                
                if first_tensor is None:
                    logger.warning("No tensor found in inputs, falling back to simple estimation")
                    self._estimate_flops_simple()
                    return
                
                # Use the first tensor for fvcore
                flops_counter = self.flop_counter_cls(self.model, (first_tensor,))
                
                # Store the detailed FLOPs breakdown
                self.detailed_flops = flops_counter.by_module()
                
                # Get total FLOPs
                self.flops_count = flops_counter.total()
            else:
                # Direct input for fvcore
                flops_counter = self.flop_counter_cls(self.model, self.inputs)
                
                # Store the detailed FLOPs breakdown
                self.detailed_flops = flops_counter.by_module()
                
                # Get total FLOPs
                self.flops_count = flops_counter.total()
                
        except Exception as e:
            logger.error(f"Error using fvcore: {e}")
            # Fall back to simpler method
            self._estimate_flops_simple()
    
    def _estimate_flops_simple(self):
        """
        Estimate FLOPs using a simple method.
        
        This is a rough estimation based on model parameters and input shapes,
        tailored to different model architectures and hardware considerations.
        It's used as a fallback when more accurate methods are not available.
        """
        logger.info(f"Estimating FLOPs using simple method for device: {self.device_type}")
        
        # Count total parameters (all and trainable)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate input size
        input_size = self._estimate_input_size()
        
        # Get model type to improve estimation
        model_type = self._detect_model_type()
        logger.debug(f"Detected model type: {model_type}, total parameters: {total_params}")
        
        # Get batch size
        batch_size = self._get_batch_size()
        
        # Apply hardware-specific multipliers
        hardware_efficiency = self._get_hardware_efficiency_factor()
        
        # Estimate FLOPs based on model type
        if model_type == "transformer":
            # Get transformer-specific attributes
            seq_length = self._get_sequence_length()
            hidden_size, num_heads, num_layers = self._get_transformer_config()
            
            # Calculate main components of transformer FLOPs
            if seq_length > 0 and hidden_size > 0:
                # More detailed breakdown for transformers
                attention_flops = self._estimate_attention_flops(seq_length, hidden_size, num_heads)
                feed_forward_flops = self._estimate_feed_forward_flops(seq_length, hidden_size, num_layers)
                embedding_flops = self._estimate_embedding_flops(seq_length, hidden_size)
                layer_norm_flops = self._estimate_layer_norm_flops(seq_length, hidden_size, num_layers)
                
                # Hardware-specific multiplier to account for optimizations
                total_flops = (attention_flops + feed_forward_flops + embedding_flops + layer_norm_flops) * batch_size
                self.flops_count = total_flops * hardware_efficiency
                
                # Store detailed breakdown
                self.detailed_flops = {
                    "attention": attention_flops * batch_size * hardware_efficiency,
                    "feed_forward": feed_forward_flops * batch_size * hardware_efficiency,
                    "embedding": embedding_flops * batch_size * hardware_efficiency,
                    "layer_norm": layer_norm_flops * batch_size * hardware_efficiency,
                    "batch_size": batch_size,
                    "sequence_length": seq_length,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "hardware_efficiency": hardware_efficiency
                }
            else:
                # Fallback when sequence length or hidden size can't be determined
                self.flops_count = total_params * 6 * batch_size * hardware_efficiency
                self.detailed_flops = {
                    "parameters": total_params * 6 * batch_size * hardware_efficiency,
                    "batch_size": batch_size,
                    "hardware_efficiency": hardware_efficiency
                }
        
        elif model_type == "cnn":
            # For CNNs, get image dimensions if possible
            image_height, image_width, channels = self._get_image_dimensions()
            
            if image_height > 0 and image_width > 0:
                # More accurate CNN estimation
                conv_flops = self._estimate_conv_flops(image_height, image_width, channels)
                pooling_flops = self._estimate_pooling_flops(image_height, image_width, channels)
                dense_flops = self._estimate_dense_flops()
                
                # Total FLOPs with hardware efficiency
                total_flops = (conv_flops + pooling_flops + dense_flops) * batch_size
                self.flops_count = total_flops * hardware_efficiency
                
                # Store detailed breakdown
                self.detailed_flops = {
                    "convolution": conv_flops * batch_size * hardware_efficiency,
                    "pooling": pooling_flops * batch_size * hardware_efficiency,
                    "dense": dense_flops * batch_size * hardware_efficiency,
                    "batch_size": batch_size,
                    "image_height": image_height,
                    "image_width": image_width,
                    "channels": channels,
                    "hardware_efficiency": hardware_efficiency
                }
            else:
                # Fallback based on parameters
                self.flops_count = total_params * 2 * batch_size * hardware_efficiency
                self.detailed_flops = {
                    "parameters": total_params * 2 * batch_size * hardware_efficiency,
                    "batch_size": batch_size,
                    "hardware_efficiency": hardware_efficiency
                }
        
        elif model_type == "multimodal":
            # For multimodal models, estimate image and text separately
            image_flops = self._estimate_image_encoder_flops()
            text_flops = self._estimate_text_encoder_flops()
            fusion_flops = self._estimate_fusion_flops()
            
            # Total FLOPs with hardware efficiency
            total_flops = (image_flops + text_flops + fusion_flops) * batch_size
            self.flops_count = total_flops * hardware_efficiency
            
            # Store detailed breakdown
            self.detailed_flops = {
                "image_encoder": image_flops * batch_size * hardware_efficiency,
                "text_encoder": text_flops * batch_size * hardware_efficiency,
                "fusion": fusion_flops * batch_size * hardware_efficiency,
                "batch_size": batch_size,
                "hardware_efficiency": hardware_efficiency
            }
        
        elif model_type == "audio":
            # For audio models, estimate audio-specific components
            audio_encoder_flops = self._estimate_audio_encoder_flops()
            feature_extraction_flops = self._estimate_feature_extraction_flops()
            
            # Total FLOPs with hardware efficiency
            total_flops = (audio_encoder_flops + feature_extraction_flops) * batch_size
            self.flops_count = total_flops * hardware_efficiency
            
            # Store detailed breakdown
            self.detailed_flops = {
                "audio_encoder": audio_encoder_flops * batch_size * hardware_efficiency,
                "feature_extraction": feature_extraction_flops * batch_size * hardware_efficiency,
                "batch_size": batch_size,
                "hardware_efficiency": hardware_efficiency
            }
        
        else:
            # Generic estimation for unknown model types
            self.flops_count = total_params * 2 * batch_size * hardware_efficiency
            
            # Store detailed breakdown
            self.detailed_flops = {
                "parameters": total_params * 2 * batch_size * hardware_efficiency,
                "batch_size": batch_size,
                "hardware_efficiency": hardware_efficiency
            }
    
    def _estimate_input_size(self) -> int:
        """
        Estimate the total number of elements in the input.
        
        Returns:
            Total number of elements in the input
        """
        if self.inputs is None:
            return 0
        
        total_size = 0
        
        if isinstance(self.inputs, dict):
            for tensor in self.inputs.values():
                if isinstance(tensor, torch.Tensor):
                    total_size += tensor.numel()
        elif isinstance(self.inputs, (list, tuple)):
            for item in self.inputs:
                if isinstance(item, torch.Tensor):
                    total_size += item.numel()
        elif isinstance(self.inputs, torch.Tensor):
            total_size = self.inputs.numel()
        
        return total_size
    
    def _detect_model_type(self) -> str:
        """
        Detect the type of model based on its class name, attributes, and architecture.
        
        Returns:
            Model type string like "transformer", "cnn", "multimodal", etc.
        """
        if self.model is None:
            return "unknown"
        
        model_class = self.model.__class__.__name__.lower()
        
        # Check model's config attribute if available
        model_config = getattr(self.model, "config", None)
        model_arch_type = getattr(model_config, "model_type", None) if model_config else None
        
        # If model has a model_type attribute in its config, use that
        if model_arch_type:
            if model_arch_type in [
                "bert", "gpt2", "gpt_neo", "t5", "bart", "roberta", "llama", "mistral", 
                "bloom", "opt", "falcon"
            ]:
                return "transformer"
            elif model_arch_type in [
                "vit", "resnet", "convnext", "efficientnet", "mobilenet", "swin"
            ]:
                return "cnn"
            elif model_arch_type in ["clip", "blip", "flava"]:
                return "multimodal"
            elif model_arch_type in ["wav2vec2", "hubert"]:
                return "audio"
            else:
                # Return the specific type for more detailed processing
                return model_arch_type
        
        # Check for transformer-based models
        transformer_names = [
            'transformer', 'bert', 'gpt', 't5', 'bart', 'roberta', 'llama', 'bloom', 'opt',
            'falcon', 'mistral', 'mixtral', 'phi', 'gemma', 'encoder', 'decoder'
        ]
        if any(name in model_class for name in transformer_names):
            return "transformer"
        
        # Check for CNN-based models
        cnn_names = [
            'conv', 'resnet', 'vgg', 'cnn', 'efficientnet', 'vit', 'mobilenet', 'densenet',
            'inception', 'squeezenet', 'shufflenet', 'resnext', 'swin', 'convnext', 'regnet'
        ]
        if any(name in model_class for name in cnn_names):
            return "cnn"
        
        # Check for multimodal models
        multimodal_names = ['clip', 'blip', 'flava', 'align', 'flamingo', 'vilt']
        if any(name in model_class for name in multimodal_names):
            return "multimodal"
        
        # Check for audio models
        audio_names = ['wav2vec', 'hubert', 'whisper', 'encodec', 'speecht5']
        if any(name in model_class for name in audio_names):
            return "audio"
        
        # Check model attributes for specific modules
        has_attention = False
        has_conv = False
        has_multimodal = False
        has_audio = False
        
        for module in self.model.modules():
            module_class = module.__class__.__name__.lower()
            
            # Check for attention mechanisms
            if any(name in module_class for name in ['attention', 'attn', 'mha', 'multihead']):
                has_attention = True
            
            # Check for convolutional layers
            if any(name in module_class for name in ['conv', 'conv2d', 'conv1d', 'convolution']):
                has_conv = True
                
            # Check for multimodal indicators
            if any(name in module_class for name in ['visual', 'text', 'fusion', 'image_encoder']):
                has_multimodal = True
                
            # Check for audio indicators
            if any(name in module_class for name in ['audio', 'spectrogram', 'mel', 'stft']):
                has_audio = True
        
        # Decide based on module composition
        if has_multimodal:
            return "multimodal"
        if has_audio:
            return "audio"
        if has_attention and has_conv:
            return "hybrid"
        if has_attention:
            return "transformer"
        if has_conv:
            return "cnn"
        
        return "unknown"
    
    def _get_sequence_length(self) -> int:
        """
        Extract sequence length from inputs.
        
        Returns:
            Sequence length if found, 0 otherwise
        """
        if self.inputs is None:
            return 0
        
        if isinstance(self.inputs, dict):
            # Try to find attention mask or input ids
            if 'attention_mask' in self.inputs:
                mask = self.inputs['attention_mask']
                if isinstance(mask, torch.Tensor) and len(mask.shape) >= 2:
                    return mask.shape[1]
            
            if 'input_ids' in self.inputs:
                ids = self.inputs['input_ids']
                if isinstance(ids, torch.Tensor) and len(ids.shape) >= 2:
                    return ids.shape[1]
            
            # Check for any 2D+ tensors and use their second dimension
            for tensor in self.inputs.values():
                if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                    return tensor.shape[1]
        
        elif isinstance(self.inputs, torch.Tensor) and len(self.inputs.shape) >= 2:
            return self.inputs.shape[1]
        
        return 0
    
    def _get_batch_size(self) -> int:
        """
        Extract batch size from inputs.
        
        Returns:
            Batch size if found, 1 otherwise
        """
        if self.inputs is None:
            return 1
        
        # Extract batch size from inputs
        if isinstance(self.inputs, dict):
            for tensor in self.inputs.values():
                if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0:
                    return tensor.shape[0]
        elif isinstance(self.inputs, (list, tuple)):
            for item in self.inputs:
                if isinstance(item, torch.Tensor) and len(item.shape) > 0:
                    return item.shape[0]
        elif isinstance(self.inputs, torch.Tensor) and len(self.inputs.shape) > 0:
            return self.inputs.shape[0]
        
        # Default batch size if not found
        return 1
    
    def _get_hardware_efficiency_factor(self) -> float:
        """
        Get hardware-specific efficiency factor for FLOPs calculation.
        
        Different hardware platforms have different optimizations that can make
        theoretical FLOPs calculations inaccurate. This function returns a multiplier
        to adjust for these differences.
        
        Returns:
            Hardware efficiency factor (multiplier for FLOPs)
        """
        # Base efficiency is 1.0
        efficiency = 1.0
        
        # Different hardware platforms have different optimizations
        if self.device_type == "cuda":
            # CUDA hardware has tensor cores and other optimizations
            efficiency = 0.85  # GPU typically more efficient than theoretical count
            
            # Try to detect tensor core eligible operations
            if self._has_tensor_core_operations():
                efficiency = 0.65  # Tensor cores are much more efficient
        
        elif self.device_type == "mps":
            # Apple Silicon has unified memory and optimizations
            efficiency = 0.9
        
        elif self.device_type == "xla":
            # TPUs have highly optimized matrix operations
            efficiency = 0.7
        
        elif self.device_type == "cpu":
            # CPUs vary widely, but AVX/FMA can improve efficiency
            if self._has_cpu_vector_extensions():
                efficiency = 0.95
            else:
                efficiency = 1.05  # Slightly less efficient without vector extensions
        
        # Adjust based on model architecture if relevant
        model_type = self._detect_model_type()
        if model_type == "transformer" and self.device_type == "cuda":
            # Transformers are highly optimized on modern GPUs
            efficiency *= 0.9
        
        return efficiency
    
    def _has_tensor_core_operations(self) -> bool:
        """
        Check if the model has operations that can use tensor cores.
        
        Returns:
            True if tensor core operations are likely, False otherwise
        """
        # Most modern transformers and CNNs have operations eligible for tensor cores
        model_type = self._detect_model_type()
        
        # Detect if model has appropriate precision for tensor cores
        fp16_or_bf16 = False
        for param in self.model.parameters():
            if param.dtype in [torch.float16, torch.bfloat16]:
                fp16_or_bf16 = True
                break
        
        # Tensor cores are most effective with fp16/bf16 and specific model types
        return fp16_or_bf16 and model_type in ["transformer", "cnn", "multimodal"]
    
    def _has_cpu_vector_extensions(self) -> bool:
        """
        Check if CPU vector extensions (AVX, FMA, etc.) are available.
        
        Returns:
            True if vector extensions are likely available, False otherwise
        """
        # Try to detect CPU vector extensions via torch config
        if hasattr(torch, '_C') and hasattr(torch._C, '_show_config'):
            config_str = torch._C._show_config()
            return any(ext in config_str for ext in ['AVX2', 'AVX512', 'FMA', 'NEON'])
        
        # Default to True as most modern CPUs have some vector extensions
        return True
    
    def _get_transformer_config(self) -> tuple:
        """
        Extract transformer model configuration parameters.
        
        Returns:
            Tuple of (hidden_size, num_heads, num_layers)
        """
        hidden_size = 0
        num_heads = 12  # Default for many transformer models
        num_layers = 12  # Default for many transformer models
        
        # Check if model has config attribute
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Extract hidden size (different models use different attribute names)
            for attr in ['hidden_size', 'd_model', 'n_embd', 'dim', 'hidden_dim']:
                if hasattr(config, attr):
                    hidden_size = getattr(config, attr)
                    break
            
            # Extract number of heads
            for attr in ['num_attention_heads', 'n_head', 'num_heads', 'nhead', 'decoder_attention_heads']:
                if hasattr(config, attr):
                    num_heads = getattr(config, attr)
                    break
            
            # Extract number of layers
            for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'encoder_layers', 'decoder_layers']:
                if hasattr(config, attr):
                    num_layers = getattr(config, attr)
                    break
        
        # Try to infer from embedding layer if not found in config
        if hidden_size == 0:
            for module in self.model.modules():
                # Try to find embedding layer
                if any(name in module.__class__.__name__.lower() for name in ['embedding', 'embeddings']):
                    if hasattr(module, 'embedding_dim'):
                        hidden_size = module.embedding_dim
                        break
                    elif hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                        hidden_size = module.weight.shape[1]
                        break
        
        if hidden_size == 0:
            # Default to a reasonable value based on parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params > 1e9:  # >1B params
                hidden_size = 2048
            elif total_params > 1e8:  # >100M params
                hidden_size = 1024
            else:
                hidden_size = 768
        
        return hidden_size, num_heads, num_layers
    
    def _get_image_dimensions(self) -> tuple:
        """
        Extract image dimensions from inputs or model configuration.
        
        Returns:
            Tuple of (height, width, channels)
        """
        height, width, channels = 0, 0, 3  # Default RGB channels
        
        # Try to get dimensions from inputs
        if self.inputs is not None:
            if isinstance(self.inputs, dict) and any(k in self.inputs for k in ['pixel_values', 'images', 'pixel_inputs']):
                # Prioritize image tensors in dictionary
                for key in ['pixel_values', 'images', 'pixel_inputs']:
                    if key in self.inputs and isinstance(self.inputs[key], torch.Tensor):
                        tensor = self.inputs[key]
                        if len(tensor.shape) == 4:  # [batch, channels, height, width]
                            _, channels, height, width = tensor.shape
                            return height, width, channels
            
            elif isinstance(self.inputs, torch.Tensor) and len(self.inputs.shape) == 4:
                # Direct tensor input
                _, channels, height, width = self.inputs.shape
                return height, width, channels
        
        # If not found in inputs, try model config
        if hasattr(self.model, 'config'):
            config = self.model.config
            for attr in ['image_size', 'input_size']:
                if hasattr(config, attr):
                    size = getattr(config, attr)
                    if isinstance(size, (list, tuple)) and len(size) >= 2:
                        height, width = size[0], size[1]
                    elif isinstance(size, int):
                        height, width = size, size
        
        # Still not found, see if we can infer from model architecture
        if height == 0 or width == 0:
            # Common image sizes for different models
            model_class = self.model.__class__.__name__.lower()
            if 'vit' in model_class:
                height, width = 224, 224
            elif 'resnet' in model_class:
                height, width = 224, 224
            elif 'efficient' in model_class:
                height, width = 224, 224
            elif 'clip' in model_class:
                height, width = 224, 224
            elif 'convnext' in model_class:
                height, width = 224, 224
        
        return height, width, channels
    
    def _estimate_attention_flops(self, seq_length: int, hidden_size: int, num_heads: int) -> int:
        """
        Estimate FLOPs for attention mechanism with detailed breakdown.
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden size dimension
            num_heads: Number of attention heads
            
        Returns:
            Estimated FLOPs for attention mechanisms
        """
        if self.model is None:
            return 0
        
        # Check for MHA vs MQA/GQA
        multi_query_attn = False
        grouped_query_attn = False
        
        if hasattr(self.model, 'config'):
            # Check for MQA/GQA indicators
            config = self.model.config
            if hasattr(config, 'kv_heads') and config.kv_heads < num_heads:
                if config.kv_heads == 1:
                    multi_query_attn = True
                else:
                    grouped_query_attn = True
                    
            # LLaMA-2 style models often use GQA
            if any(name in str(config) for name in ['llama', 'mistral', 'falcon']):
                if not multi_query_attn and not grouped_query_attn:
                    # Check for specific attributes in modern models
                    grouped_query_attn = True
        
        # More detailed attention FLOPs calculation based on attention type
        if multi_query_attn:
            # Multi-Query Attention - used in PaLM, Falcon
            # 1. Query projections: hidden_size^2 * seq_length for all heads
            # 2. Key, Value projections: 2 * hidden_size^2 * seq_length / num_heads (shared K/V)
            # 3. QK attention: hidden_size * seq_length^2
            # 4. Softmax: seq_length^2 * num_heads
            # 5. Attention * Value: hidden_size * seq_length^2
            # 6. Output projection: hidden_size^2 * seq_length
            head_dim = hidden_size // num_heads
            
            qkv_flops = (hidden_size**2 * seq_length +  # Q projection
                         2 * (hidden_size * head_dim) * seq_length)  # K,V projections (only 1 per model)
            attention_flops = (2 * hidden_size * seq_length**2 +  # QK and Attention*V
                              seq_length**2 * num_heads)  # Softmax
            output_flops = hidden_size**2 * seq_length  # Output projection
            
            return qkv_flops + attention_flops + output_flops
        
        elif grouped_query_attn:
            # Grouped-Query Attention - used in LLaMA2, Mistral, etc.
            # Similar to MQA but with more than 1 KV head
            # Assume num_kv_heads = num_heads / 4 (common ratio)
            num_kv_heads = max(1, num_heads // 4)
            head_dim = hidden_size // num_heads
            
            qkv_flops = (hidden_size**2 * seq_length +  # Q projection
                         2 * (hidden_size * head_dim * num_kv_heads) * seq_length)  # K,V projections
            attention_flops = (2 * hidden_size * seq_length**2 +  # QK and Attention*V
                              seq_length**2 * num_heads)  # Softmax
            output_flops = hidden_size**2 * seq_length  # Output projection
            
            return qkv_flops + attention_flops + output_flops
        
        else:
            # Standard Multi-Head Attention
            # 1. Query, Key, Value projections: 3 * hidden_size^2 * seq_length
            # 2. QK attention: hidden_size * seq_length^2
            # 3. Softmax: seq_length^2 * num_heads
            # 4. Attention * Value: hidden_size * seq_length^2
            # 5. Output projection: hidden_size^2 * seq_length
            
            qkv_flops = 3 * hidden_size**2 * seq_length  # QKV projections
            attention_flops = (2 * hidden_size * seq_length**2 +  # QK and Attention*V
                              seq_length**2 * num_heads)  # Softmax
            output_flops = hidden_size**2 * seq_length  # Output projection
            
            return qkv_flops + attention_flops + output_flops
    
    def _estimate_feed_forward_flops(self, seq_length: int, hidden_size: int, num_layers: int) -> int:
        """
        Estimate FLOPs for feed forward networks in transformers.
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden size dimension
            num_layers: Number of transformer layers
            
        Returns:
            Estimated FLOPs for feed forward networks
        """
        if self.model is None:
            return 0
        
        # Detect if MLP uses 4x or other expansion factor
        expansion_factor = 4  # Default for most transformer models
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'intermediate_size') and hasattr(config, 'hidden_size'):
                expansion_factor = config.intermediate_size / config.hidden_size
            elif hasattr(config, 'mlp_dim') and hasattr(config, 'd_model'):
                expansion_factor = config.mlp_dim / config.d_model
            elif hasattr(config, 'ffn_dim') and hasattr(config, 'd_model'):
                expansion_factor = config.ffn_dim / config.d_model
        
        # Per-layer feed forward FLOPs
        # 1. First linear: hidden_size * (hidden_size * expansion_factor) * seq_length
        # 2. Second linear: (hidden_size * expansion_factor) * hidden_size * seq_length
        # 3. Activation and other operations: ~hidden_size * expansion_factor * seq_length
        ff_flops_per_layer = (hidden_size * (hidden_size * expansion_factor) * seq_length +
                             (hidden_size * expansion_factor) * hidden_size * seq_length +
                             hidden_size * expansion_factor * seq_length)
        
        # Total feed forward FLOPs across all layers
        return ff_flops_per_layer * num_layers
    
    def _estimate_embedding_flops(self, seq_length: int, hidden_size: int) -> int:
        """
        Estimate FLOPs for embedding operations.
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden size dimension
            
        Returns:
            Estimated FLOPs for embedding operations
        """
        if self.model is None:
            return 0
        
        # Embedding lookup is typically not counted in FLOPs
        # But we count position embeddings and layer norm
        
        # Position embeddings
        pos_embedding_flops = seq_length * hidden_size
        
        # Layer norm on embeddings
        layer_norm_flops = 2 * seq_length * hidden_size
        
        return pos_embedding_flops + layer_norm_flops
    
    def _estimate_layer_norm_flops(self, seq_length: int, hidden_size: int, num_layers: int) -> int:
        """
        Estimate FLOPs for layer normalization operations.
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden size dimension
            num_layers: Number of transformer layers
            
        Returns:
            Estimated FLOPs for layer normalization
        """
        if self.model is None:
            return 0
        
        # Each transformer layer typically has 2 layer norms
        # Layer norm operations: 5 * seq_length * hidden_size
        # (mean, variance, normalize, scale, shift)
        layer_norm_flops_per_layer = 5 * seq_length * hidden_size * 2
        
        return layer_norm_flops_per_layer * num_layers
    
    def _estimate_conv_flops(self, height: int, width: int, channels: int) -> int:
        """
        Estimate FLOPs for convolutional operations.
        
        Args:
            height: Image height
            width: Image width
            channels: Number of input channels
            
        Returns:
            Estimated FLOPs for convolutional operations
        """
        if self.model is None:
            return 0
        
        # Count conv layers and estimate FLOPs
        conv_flops = 0
        
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d)):
                # Extract kernel dimensions
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                
                # Extract in/out channels
                in_channels = module.in_channels
                out_channels = module.out_channels
                
                # Calculate output dimensions
                padding = module.padding
                if isinstance(padding, int):
                    padding = (padding, padding)
                    
                stride = module.stride
                if isinstance(stride, int):
                    stride = (stride, stride)
                
                out_height = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
                out_width = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
                
                # Update for next layer
                height, width = out_height, out_width
                
                # FLOPs for this conv layer: multiply-adds per output element * number of output elements
                flops_per_element = kernel_size[0] * kernel_size[1] * in_channels
                conv_flops += flops_per_element * out_channels * out_height * out_width
        
        # If we didn't find any conv layers, make a rough estimate based on ViT/ResNet
        if conv_flops == 0:
            model_type = self._detect_model_type()
            if model_type == "cnn":
                # Rough estimate based on ResNet-like architecture
                conv_flops = 2 * height * width * channels * 1000000  # 1M is a rough approximation
        
        return conv_flops
    
    def _estimate_pooling_flops(self, height: int, width: int, channels: int) -> int:
        """
        Estimate FLOPs for pooling operations.
        
        Args:
            height: Image height
            width: Image width
            channels: Number of input channels
            
        Returns:
            Estimated FLOPs for pooling operations
        """
        # Pooling operations are relatively cheap compared to convolutions
        # We estimate based on common architecture patterns
        model_type = self._detect_model_type()
        
        if model_type == "cnn":
            # Typical CNN has ~5 pooling layers
            pooling_layers = 5
            
            # Average pooling FLOPs: kernel_size^2 operations per output element
            avg_kernel_size = 2
            pooling_flops = pooling_layers * (avg_kernel_size**2) * height * width * channels / 4
            
            return int(pooling_flops)
        else:
            return 0
    
    def _estimate_dense_flops(self) -> int:
        """
        Estimate FLOPs for fully connected / dense layers.
        
        Returns:
            Estimated FLOPs for fully connected operations
        """
        if self.model is None:
            return 0
        
        dense_flops = 0
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                # FLOPs for linear layer: input_features * output_features
                input_features = module.in_features
                output_features = module.out_features
                
                # Assume batch_size=1 and multiply by actual batch size later
                dense_flops += input_features * output_features
        
        return dense_flops
    
    def _estimate_image_encoder_flops(self) -> int:
        """
        Estimate FLOPs for image encoder in multimodal models.
        
        Returns:
            Estimated FLOPs for image encoder
        """
        if self.model is None:
            return 0
        
        # Try to identify image encoder in multimodal model
        image_encoder = None
        
        # Common attribute names for image encoder
        image_encoder_names = ['vision_model', 'image_encoder', 'visual_encoder', 'vision']
        
        for name in image_encoder_names:
            if hasattr(self.model, name):
                image_encoder = getattr(self.model, name)
                break
        
        if image_encoder is not None:
            # Get image dimensions
            height, width, channels = self._get_image_dimensions()
            
            # Use CNN estimation for image encoder
            if height > 0 and width > 0:
                return self._estimate_conv_flops(height, width, channels) + self._estimate_dense_flops()
        
        # Fallback to estimation based on common architectures
        height, width, channels = self._get_image_dimensions()
        if height > 0 and width > 0:
            return 1000000000  # ~1B FLOPs for typical vision encoder
        
        return 500000000  # ~0.5B FLOPs as default
    
    def _estimate_text_encoder_flops(self) -> int:
        """
        Estimate FLOPs for text encoder in multimodal models.
        
        Returns:
            Estimated FLOPs for text encoder
        """
        if self.model is None:
            return 0
        
        # Try to identify text encoder in multimodal model
        text_encoder = None
        
        # Common attribute names for text encoder
        text_encoder_names = ['text_model', 'text_encoder', 'language_model', 'text']
        
        for name in text_encoder_names:
            if hasattr(self.model, name):
                text_encoder = getattr(self.model, name)
                break
        
        if text_encoder is not None:
            # Get sequence length
            seq_length = self._get_sequence_length()
            
            # Try to get text encoder config
            hidden_size, num_heads, num_layers = 0, 0, 0
            
            if hasattr(text_encoder, 'config'):
                config = text_encoder.config
                for attr in ['hidden_size', 'd_model', 'dim']:
                    if hasattr(config, attr):
                        hidden_size = getattr(config, attr)
                        break
                
                for attr in ['num_attention_heads', 'n_head']:
                    if hasattr(config, attr):
                        num_heads = getattr(config, attr)
                        break
                
                for attr in ['num_hidden_layers', 'n_layer']:
                    if hasattr(config, attr):
                        num_layers = getattr(config, attr)
                        break
            
            # Use transformer estimation if we have the parameters
            if seq_length > 0 and hidden_size > 0 and num_heads > 0 and num_layers > 0:
                attention_flops = self._estimate_attention_flops(seq_length, hidden_size, num_heads)
                feed_forward_flops = self._estimate_feed_forward_flops(seq_length, hidden_size, num_layers)
                embedding_flops = self._estimate_embedding_flops(seq_length, hidden_size)
                layer_norm_flops = self._estimate_layer_norm_flops(seq_length, hidden_size, num_layers)
                
                return attention_flops + feed_forward_flops + embedding_flops + layer_norm_flops
        
        # Fallback to estimation based on common architectures
        seq_length = self._get_sequence_length()
        if seq_length > 0:
            return 500000000  # ~0.5B FLOPs for typical text encoder
        
        return 200000000  # ~0.2B FLOPs as default
    
    def _estimate_fusion_flops(self) -> int:
        """
        Estimate FLOPs for fusion mechanism in multimodal models.
        
        Returns:
            Estimated FLOPs for fusion mechanism
        """
        if self.model is None:
            return 0
        
        # Try to identify fusion module
        fusion_module = None
        
        # Common attribute names for fusion module
        fusion_names = ['fusion_model', 'fusion_layer', 'fusion', 'multimodal_encoder']
        
        for name in fusion_names:
            if hasattr(self.model, name):
                fusion_module = getattr(self.model, name)
                break
        
        if fusion_module is not None:
            # Estimate based on linear layers in fusion module
            dense_flops = 0
            
            for module in fusion_module.modules():
                if isinstance(module, torch.nn.Linear):
                    dense_flops += module.in_features * module.out_features
            
            if dense_flops > 0:
                return dense_flops
        
        # Fallback to estimation based on common architectures
        # Most fusion is just a linear layer or simple attention
        return 50000000  # ~50M FLOPs for typical fusion
    
    def _estimate_audio_encoder_flops(self) -> int:
        """
        Estimate FLOPs for audio encoder in audio models.
        
        Returns:
            Estimated FLOPs for audio encoder
        """
        if self.model is None:
            return 0
        
        # Try to identify audio encoder
        audio_encoder = None
        
        # Common attribute names for audio encoder
        audio_encoder_names = ['audio_model', 'audio_encoder', 'encoder', 'wav2vec2']
        
        for name in audio_encoder_names:
            if hasattr(self.model, name):
                audio_encoder = getattr(self.model, name)
                break
        
        if audio_encoder is not None:
            # Count conv layers and linear layers
            conv_flops = 0
            dense_flops = 0
            
            for module in audio_encoder.modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # Rough estimate for audio convolutions
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    
                    # Assume audio length of 16000 (1 second at 16kHz)
                    audio_length = 16000
                    conv_flops += in_channels * out_channels * kernel_size * audio_length
                
                elif isinstance(module, torch.nn.Linear):
                    dense_flops += module.in_features * module.out_features
            
            if conv_flops > 0 or dense_flops > 0:
                return conv_flops + dense_flops
        
        # Fallback to estimation based on common architectures
        return 800000000  # ~0.8B FLOPs for typical audio encoder
    
    def _estimate_feature_extraction_flops(self) -> int:
        """
        Estimate FLOPs for feature extraction in audio models.
        
        Returns:
            Estimated FLOPs for feature extraction operations
        """
        if self.model is None:
            return 0
        
        # Try to identify feature extraction module
        feature_module = None
        
        # Common attribute names for feature extraction
        feature_names = ['feature_extractor', 'feature_projection', 'preprocessor']
        
        for name in feature_names:
            if hasattr(self.model, name):
                feature_module = getattr(self.model, name)
                break
        
        if feature_module is not None:
            # Estimate based on common audio preprocessing
            # STFT + Mel spectrogram + normalization
            
            # Assume audio length of 16000 (1 second at 16kHz)
            audio_length = 16000
            
            # STFT operations: ~10 FLOPs per audio sample
            stft_flops = 10 * audio_length
            
            # Mel filterbank: ~20 FLOPs per frequency bin per frame
            # Assume 80 mel bins and 50 frames per second
            mel_flops = 20 * 80 * 50
            
            # Normalization: ~3 FLOPs per mel bin per frame
            norm_flops = 3 * 80 * 50
            
            return stft_flops + mel_flops + norm_flops
        
        # Fallback to estimation
        return 10000000  # ~10M FLOPs for audio preprocessing
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the FLOPs metrics, including hardware-specific information.
        
        Returns:
            Dictionary of FLOPs metrics and related hardware information
        """
        # Make sure FLOPs have been calculated
        if not self.has_estimated and self.model is not None and self.inputs is not None:
            self.stop()
        
        # Convert to various units
        gflops = self.flops_count / 1e9  # GFLOPs (billions)
        tflops = self.flops_count / 1e12  # TFLOPs (trillions)
        
        # Get model type
        model_type = self._detect_model_type() if self.model else "unknown"
        
        # Get batch size if available
        batch_size = self._get_batch_size() if self.model and self.inputs else 1
        
        # Get hardware efficiency factor
        efficiency = self._get_hardware_efficiency_factor() if self.model else 1.0
        
        # Basic metrics dictionary
        metrics = {
            "flops": self.flops_count,
            "gflops": gflops,
            "tflops": tflops,
            "model_type": model_type,
            "device_type": self.device_type,
            "hardware_efficiency": efficiency,
            "batch_size": batch_size
        }
        
        # Add flops efficiency metrics
        if self.model and batch_size > 0:
            # Theoretical throughput (items/s) at 100% efficiency
            if self.device_type == "cuda" and hasattr(torch.cuda, "get_device_properties"):
                # Try to get GPU TFLOPS capability
                try:
                    device_id = 0  # Default to first GPU
                    if isinstance(self.inputs, torch.Tensor) and hasattr(self.inputs, "device"):
                        device_id = self.inputs.device.index or 0
                    
                    props = torch.cuda.get_device_properties(device_id)
                    gpu_flops = props.multi_processor_count * props.max_threads_per_multi_processor * 2  # FMA = 2 ops
                    gpu_clock = props.clock_rate / 1e3  # Convert to MHz
                    theoretical_tflops = gpu_flops * gpu_clock / 1e6  # Theoretical TFLOPS
                    
                    # Add to metrics
                    metrics["gpu_theoretical_tflops"] = theoretical_tflops
                    
                    # Efficiency percentage if theoretical is available
                    metrics["flops_utilization_pct"] = min(100.0, (tflops / theoretical_tflops) * 100) if theoretical_tflops > 0 else 0
                except Exception as e:
                    # Silently fail if we can't get device properties
                    pass
        
        # Add parameters count if available
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            metrics.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "flops_per_parameter": self.flops_count / total_params if total_params > 0 else 0
            })
            
            # Add model precision information
            precision_info = self._get_model_precision_info()
            metrics.update(precision_info)
        
        return metrics
    
    def _get_model_precision_info(self) -> Dict[str, Any]:
        """
        Get information about model precision (data types of parameters).
        
        Returns:
            Dictionary with precision information
        """
        if self.model is None:
            return {}
        
        # Count parameters by dtype
        dtype_counts = {}
        total_params = 0
        
        for param in self.model.parameters():
            dtype_str = str(param.dtype).split(".")[-1]
            total_params += param.numel()
            
            if dtype_str in dtype_counts:
                dtype_counts[dtype_str] += param.numel()
            else:
                dtype_counts[dtype_str] = param.numel()
        
        # Convert counts to percentages
        precision_info = {}
        for dtype, count in dtype_counts.items():
            precision_info[f"precision_{dtype}_pct"] = (count / total_params) * 100 if total_params > 0 else 0
        
        # Add dominant precision
        if dtype_counts:
            dominant_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
            precision_info["dominant_precision"] = dominant_dtype
            
            # Add specific precision flag for validation
            precision_info["precision_float"] = "float16" in dtype_counts or "float32" in dtype_counts
        
        return precision_info
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed FLOPs breakdown by model component with hardware-specific information.
        
        Returns:
            Dictionary with detailed FLOPs breakdown and hardware information
        """
        # Get base metrics
        metrics = self.get_metrics()
        
        # Add detailed FLOPs breakdown if available
        if self.detailed_flops:
            # Calculate component percentages
            if self.flops_count > 0:
                component_percentages = {}
                for component, flops in self.detailed_flops.items():
                    if isinstance(flops, (int, float)) and not isinstance(flops, bool):
                        component_percentages[f"{component}_pct"] = (flops / self.flops_count) * 100
                
                # Add component percentages to metrics
                metrics.update(component_percentages)
            
            # Add raw detailed breakdown
            metrics["detailed_flops"] = self.detailed_flops
        
        # Add hardware-specific details for this device
        if self.device_type == "cuda":
            # Add CUDA-specific information if available
            try:
                metrics["cuda_device_name"] = torch.cuda.get_device_name(0)
                metrics["cuda_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
                metrics["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                
                if self._has_tensor_core_operations():
                    metrics["tensor_core_eligible"] = True
            except Exception:
                # Skip if CUDA information can't be retrieved
                pass
        
        elif self.device_type == "cpu":
            # Try to add CPU information
            if self._has_cpu_vector_extensions():
                metrics["cpu_vector_extensions"] = True
        
        # Add model-specific advanced metrics
        model_type = metrics.get("model_type", "unknown")
        
        if model_type == "transformer":
            # Add transformer-specific metrics if we have the input information
            if self.model and self.inputs:
                hidden_size, num_heads, num_layers = self._get_transformer_config()
                seq_length = self._get_sequence_length()
                
                if hidden_size > 0 and seq_length > 0:
                    # Calculate theoretical attention scaling
                    attention_scaling = {
                        "attention_complexity": "O(n²)",  # Standard attention is O(n²)
                        "sequence_length": seq_length,
                        "attention_scaling_factor": seq_length * seq_length / 1000000  # n² scaling factor
                    }
                    
                    # Check for optimized attention mechanisms
                    if any(name.lower() in str(self.model.__class__).lower() for name in ["flash", "triton", "xformers"]):
                        attention_scaling["attention_optimized"] = True
                        attention_scaling["attention_complexity"] = "O(n) to O(n log n)"  # Optimized attention can be sub-quadratic
                    
                    # Add transformer architecture details
                    transformer_details = {
                        "hidden_size": hidden_size,
                        "num_attention_heads": num_heads,
                        "num_hidden_layers": num_layers,
                        "flops_per_token": self.flops_count / (seq_length * self._get_batch_size()) if seq_length > 0 else 0
                    }
                    
                    metrics.update(attention_scaling)
                    metrics.update(transformer_details)
        
        elif model_type == "cnn":
            # Add CNN-specific metrics
            if self.model and self.inputs:
                height, width, channels = self._get_image_dimensions()
                
                if height > 0 and width > 0:
                    cnn_details = {
                        "image_height": height,
                        "image_width": width,
                        "input_channels": channels,
                        "flops_per_pixel": self.flops_count / (height * width * self._get_batch_size()) if height > 0 and width > 0 else 0
                    }
                    
                    metrics.update(cnn_details)
        
        return metrics


class FLOPsMetricFactory:
    """Factory class for creating appropriate FLOPs metrics based on hardware."""
    
    @staticmethod
    def create(device: Any) -> FLOPsMetric:
        """
        Create a FLOPs metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            FLOPsMetric instance configured for the device
        """
        device_type = FLOPsMetricFactory._get_device_type(device)
        return FLOPsMetric(device_type)
    
    @staticmethod
    def _get_device_type(device: Any) -> str:
        """
        Extract device type from the device object.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            Device type string
        """
        device_type = "cpu"
        
        # Handle PyTorch devices
        if isinstance(device, torch.device):
            device_type = device.type
        # Handle hardware backend devices
        elif isinstance(device, dict) and "device" in device:
            device_type = device["device"]
        # Handle strings
        elif isinstance(device, str):
            device_type = device.split(":")[0]  # Handle "cuda:0" format
        
        return device_type