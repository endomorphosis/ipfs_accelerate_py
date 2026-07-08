"""
HuggingFace Integration Module for Benchmark Suite

This module provides integration with HuggingFace Transformers for benchmarking
300+ model architectures. It implements helpers for model discovery, organization
by architecture type, and efficient loading across different hardware backends.
"""

import os
import logging
import importlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Try importing transformers, handle gracefully if not available
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("HuggingFace Transformers not available. Some functionality will be limited.")

class ModelArchitectureRegistry:
    """Registry of HuggingFace model architectures for benchmarking."""
    
    # Architecture type categories
    ENCODER_MODELS = {
        'albert', 'bert', 'camembert', 'canine', 'convbert', 'deberta', 'distilbert', 
        'electra', 'flaubert', 'layoutlm', 'mobilebert', 'mpnet', 'roberta', 'roformer', 
        'xlm-roberta'
    }
    
    DECODER_MODELS = {
        'bloom', 'ctrl', 'falcon', 'gemma', 'gpt2', 'gpt-j', 'gpt-neo', 'gpt-neox', 
        'gptj', 'llama', 'mistral', 'mixtral', 'opt', 'phi', 'pythia'
    }
    
    ENCODER_DECODER_MODELS = {
        'bart', 'blenderbot', 'flax-t5', 'fsmt', 'led', 'longt5', 'marian', 'mbart', 
        'mt5', 'pegasus', 't5', 'umt5'
    }
    
    VISION_MODELS = {
        'beit', 'clip', 'convnext', 'cvt', 'deit', 'deta', 'detr', 'dinov2', 'dpt',
        'efficientformer', 'efficientnet', 'focalnet', 'levit', 'mobilenet', 'mobilevit',
        'perceiver', 'poolformer', 'regnet', 'resnet', 'segformer', 'swin', 'vit'
    }
    
    AUDIO_MODELS = {
        'audio-spectrogram-transformer', 'bark', 'clap', 'data2vec-audio', 'encodec',
        'hubert', 'musicgen', 'sew', 'sew-d', 'speecht5', 'unispeech', 'unispeech-sat',
        'wav2vec2', 'wavlm', 'whisper'
    }
    
    MULTIMODAL_MODELS = {
        'blip', 'blip-2', 'bridgetower', 'donut', 'flava', 'git', 'idefics', 'llava',
        'owlvit', 'paligemma', 'perceiver', 'siglip', 'vision-text-dual-encoder',
        'xclip'
    }
    
    DIFFUSION_MODELS = {
        'controlnet', 'stable-diffusion', 'stable-diffusion-xl', 'vq-diffusion'
    }
    
    @staticmethod
    def get_model_type(model_name: str) -> str:
        """
        Determine the model type based on the model name.
        
        Args:
            model_name: Name of the model or model path
        
        Returns:
            Model type category ('encoder', 'decoder', 'encoder-decoder', 'vision', 
            'audio', 'multimodal', 'diffusion', or 'unknown')
        """
        # Get base name without path
        if '/' in model_name:
            parts = model_name.split('/')
            base_name = parts[-1].lower()
            org_name = parts[0].lower() if len(parts) > 1 else ""
        else:
            base_name = model_name.lower()
            org_name = ""
        
        # Check each category
        for name in ModelArchitectureRegistry.ENCODER_MODELS:
            if name in base_name:
                return 'encoder'
        
        for name in ModelArchitectureRegistry.DECODER_MODELS:
            if name in base_name:
                return 'decoder'
        
        for name in ModelArchitectureRegistry.ENCODER_DECODER_MODELS:
            if name in base_name:
                return 'encoder-decoder'
        
        for name in ModelArchitectureRegistry.VISION_MODELS:
            if name in base_name:
                return 'vision'
        
        for name in ModelArchitectureRegistry.AUDIO_MODELS:
            if name in base_name:
                return 'audio'
        
        for name in ModelArchitectureRegistry.MULTIMODAL_MODELS:
            if name in base_name:
                return 'multimodal'
        
        for name in ModelArchitectureRegistry.DIFFUSION_MODELS:
            if name in base_name:
                return 'diffusion'
        
        # Handle special cases based on organization
        if org_name in ["openai", "gpt"]:
            return 'decoder'
        
        if org_name in ["google", "t5"]:
            return 'encoder-decoder'
        
        # Default to unknown
        return 'unknown'
    
    @staticmethod
    def get_model_family(model_name: str) -> str:
        """
        Determine the model family based on the model name.
        
        Args:
            model_name: Name of the model or model path
        
        Returns:
            Model family name (bert, gpt, t5, etc.) or 'unknown'
        """
        # Get base name without path
        if '/' in model_name:
            parts = model_name.split('/')
            base_name = parts[-1].lower()
            org_name = parts[0].lower() if len(parts) > 1 else ""
        else:
            base_name = model_name.lower()
            org_name = ""
        
        # Check for common model families
        common_families = [
            'albert', 'bart', 'bert', 'bloom', 'clip', 'deberta', 'detr', 'distilbert',
            'electra', 'falcon', 'gpt2', 'gpt-j', 'gpt-neo', 'llama', 'mixtral', 'opt', 
            'roberta', 't5', 'vit', 'whisper'
        ]
        
        for family in common_families:
            if family in base_name:
                return family
        
        # Handle hyphenated families
        if "gpt-" in base_name:
            return "gpt"
        
        if "xlm-" in base_name:
            return "xlm"
        
        # Default to unknown
        return 'unknown'
    
    @staticmethod
    def get_model_parameters(model_name: str) -> Optional[float]:
        """
        Estimate the number of parameters for a model based on its name.
        
        Args:
            model_name: Name of the model or model path
        
        Returns:
            Estimated number of parameters in millions, or None if unknown
        """
        # Get base name without path
        if '/' in model_name:
            parts = model_name.split('/')
            base_name = parts[-1].lower()
        else:
            base_name = model_name.lower()
        
        # Extract numbers from name
        import re
        numbers = re.findall(r'(\d+)[bm]', base_name)
        if numbers:
            # Handle B (billion) or M (million) suffixes
            if 'b' in base_name.lower():
                return float(numbers[0]) * 1000  # Convert B to M
            else:
                return float(numbers[0])
        
        # Handle common model sizes
        if 'tiny' in base_name:
            return 60  # Approximation
        if 'mini' in base_name:
            return 100  # Approximation
        if 'small' in base_name:
            return 250  # Approximation
        if 'medium' in base_name:
            return 500  # Approximation
        if 'base' in base_name:
            return 750  # Approximation
        if 'large' in base_name:
            return 1000  # Approximation
        if 'xl' in base_name:
            return 1500  # Approximation
        if 'xxl' in base_name:
            return 2000  # Approximation
        
        # Default to None
        return None
    
    @staticmethod
    def get_modality(model_type: str) -> str:
        """
        Determine the modality based on the model type.
        
        Args:
            model_type: Model type category
        
        Returns:
            Modality ('text', 'vision', 'audio', 'multimodal')
        """
        if model_type in ['encoder', 'decoder', 'encoder-decoder']:
            return 'text'
        elif model_type == 'vision':
            return 'vision'
        elif model_type == 'audio':
            return 'audio'
        elif model_type in ['multimodal', 'diffusion']:
            return 'multimodal'
        else:
            return 'unknown'
    
    @staticmethod
    def get_all_model_architectures() -> List[str]:
        """
        Get a list of all supported model architectures.
        
        Returns:
            List of model architecture names
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("HuggingFace Transformers not available. Cannot list model architectures.")
            return []
        
        # Gather all model architectures from different categories
        architectures = []
        architectures.extend(ModelArchitectureRegistry.ENCODER_MODELS)
        architectures.extend(ModelArchitectureRegistry.DECODER_MODELS)
        architectures.extend(ModelArchitectureRegistry.ENCODER_DECODER_MODELS)
        architectures.extend(ModelArchitectureRegistry.VISION_MODELS)
        architectures.extend(ModelArchitectureRegistry.AUDIO_MODELS)
        architectures.extend(ModelArchitectureRegistry.MULTIMODAL_MODELS)
        architectures.extend(ModelArchitectureRegistry.DIFFUSION_MODELS)
        
        return sorted(list(set(architectures)))
    
    @staticmethod
    def group_models_by_type() -> Dict[str, List[str]]:
        """
        Group model architectures by type.
        
        Returns:
            Dictionary mapping model types to lists of architectures
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("HuggingFace Transformers not available. Cannot group model architectures.")
            return {}
        
        return {
            'encoder': sorted(list(ModelArchitectureRegistry.ENCODER_MODELS)),
            'decoder': sorted(list(ModelArchitectureRegistry.DECODER_MODELS)),
            'encoder-decoder': sorted(list(ModelArchitectureRegistry.ENCODER_DECODER_MODELS)),
            'vision': sorted(list(ModelArchitectureRegistry.VISION_MODELS)),
            'audio': sorted(list(ModelArchitectureRegistry.AUDIO_MODELS)),
            'multimodal': sorted(list(ModelArchitectureRegistry.MULTIMODAL_MODELS)),
            'diffusion': sorted(list(ModelArchitectureRegistry.DIFFUSION_MODELS))
        }
        
class ModelLoader:
    """
    Helper class for loading HuggingFace models with different hardware backends.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory to use for model caching
        """
        self.cache_dir = cache_dir or os.environ.get("TRANSFORMERS_CACHE")
        self.loaded_models = {}  # Cache of loaded models
    
    def load_model(self, 
                 model_name: str, 
                 hardware: str = 'cpu',
                 precision: str = 'fp32',
                 batch_size: int = 1,
                 simulation_mode: bool = False,
                 **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a HuggingFace model with the specified hardware backend.
        
        Args:
            model_name: Name of the model or model path
            hardware: Hardware backend (cpu, cuda, rocm, mps, webgpu, webnn, etc.)
            precision: Precision to use (fp32, fp16, bf16, int8, etc.)
            batch_size: Batch size for the model
            simulation_mode: Whether to use simulation mode for hardware
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata) - the loaded model and metadata about it
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("HuggingFace Transformers not available. Cannot load model.")
            raise ImportError("HuggingFace Transformers not available")
        
        # Create cache key for this model configuration
        cache_key = f"{model_name}_{hardware}_{precision}_{batch_size}"
        
        # Check if model is already loaded
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Get model type to determine how to load it
        model_type = ModelArchitectureRegistry.get_model_type(model_name)
        model_family = ModelArchitectureRegistry.get_model_family(model_name)
        
        # Set up device based on hardware
        device = self._get_device(hardware, simulation_mode)
        
        # Determine dtype based on precision
        dtype = self._get_dtype(precision)
        
        # Load model based on model type
        try:
            if model_type == 'encoder':
                model, metadata = self._load_encoder_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'decoder':
                model, metadata = self._load_decoder_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'encoder-decoder':
                model, metadata = self._load_encoder_decoder_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'vision':
                model, metadata = self._load_vision_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'audio':
                model, metadata = self._load_audio_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'multimodal':
                model, metadata = self._load_multimodal_model(model_name, device, dtype, batch_size, **kwargs)
            elif model_type == 'diffusion':
                model, metadata = self._load_diffusion_model(model_name, device, dtype, batch_size, **kwargs)
            else:
                model, metadata = self._load_generic_model(model_name, device, dtype, batch_size, **kwargs)
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name} on {hardware}: {e}")
            raise
        
        # Add model metadata
        metadata.update({
            'model_name': model_name,
            'model_type': model_type,
            'model_family': model_family,
            'hardware': hardware,
            'precision': precision,
            'batch_size': batch_size,
            'parameters_million': ModelArchitectureRegistry.get_model_parameters(model_name),
            'modality': ModelArchitectureRegistry.get_modality(model_type),
            'simulation_mode': simulation_mode
        })
        
        # Cache model
        self.loaded_models[cache_key] = (model, metadata)
        
        return model, metadata
    
    def _get_device(self, hardware: str, simulation_mode: bool) -> str:
        """
        Get the device string for the specified hardware.
        
        Args:
            hardware: Hardware type (cpu, cuda, rocm, mps, etc.)
            simulation_mode: Whether to use simulation mode
        
        Returns:
            Device string for PyTorch or other frameworks
        """
        # If simulation mode, return 'cpu' regardless
        if simulation_mode:
            return 'cpu'
        
        # Check for PyTorch device
        try:
            import torch
            
            if hardware == 'cpu':
                return 'cpu'
            elif hardware == 'cuda':
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    return 'cpu'
            elif hardware == 'rocm':
                if torch.cuda.is_available() and torch.version.hip is not None:
                    return 'cuda'  # ROCm uses cuda device in PyTorch
                else:
                    logger.warning("ROCm requested but not available. Falling back to CPU.")
                    return 'cpu'
            elif hardware == 'mps':
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    logger.warning("MPS requested but not available. Falling back to CPU.")
                    return 'cpu'
            elif hardware in ['webgpu', 'webnn']:
                logger.warning(f"{hardware} is not directly supported by PyTorch. Using CPU.")
                return 'cpu'
            else:
                logger.warning(f"Unknown hardware {hardware}. Falling back to CPU.")
                return 'cpu'
        except ImportError:
            logger.warning("PyTorch not available. Using 'cpu' as device.")
            return 'cpu'
    
    def _get_dtype(self, precision: str) -> Any:
        """
        Get the data type for the specified precision.
        
        Args:
            precision: Precision string (fp32, fp16, bf16, int8, etc.)
        
        Returns:
            Data type object for the specified precision
        """
        try:
            import torch
            
            if precision == 'fp32':
                return torch.float32
            elif precision == 'fp16':
                return torch.float16
            elif precision == 'bf16':
                return torch.bfloat16
            elif precision == 'int8':
                return torch.int8
            else:
                logger.warning(f"Unknown precision {precision}. Falling back to fp32.")
                return torch.float32
        except ImportError:
            logger.warning("PyTorch not available. Ignoring precision setting.")
            return None
    
    def _load_encoder_model(self, 
                          model_name: str, 
                          device: str, 
                          dtype: Any, 
                          batch_size: int,
                          **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load an encoder model (BERT, RoBERTa, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 'bert' in model_name.lower():
            model_class = transformers.BertModel
            tokenizer_class = transformers.BertTokenizer
        elif 'roberta' in model_name.lower():
            model_class = transformers.RobertaModel
            tokenizer_class = transformers.RobertaTokenizer
        elif 'distilbert' in model_name.lower():
            model_class = transformers.DistilBertModel
            tokenizer_class = transformers.DistilBertTokenizer
        elif 'albert' in model_name.lower():
            model_class = transformers.AlbertModel
            tokenizer_class = transformers.AlbertTokenizer
        else:
            # Use auto classes for other encoder models
            model_class = transformers.AutoModel
            tokenizer_class = transformers.AutoTokenizer
        
        # Load tokenizer
        tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if tokenizer is not None:
            # Tokenize with batch size
            dummy_text = ["Hello, world!"] * batch_size
            dummy_input = tokenizer(dummy_text, return_tensors='pt', padding=True)
            
            # Move inputs to device
            if device != 'cpu':
                for key in dummy_input:
                    if hasattr(dummy_input[key], 'to'):
                        dummy_input[key] = dummy_input[key].to(device)
        else:
            dummy_input = None
        
        metadata = {
            'tokenizer': tokenizer,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_decoder_model(self, 
                          model_name: str, 
                          device: str, 
                          dtype: Any, 
                          batch_size: int,
                          **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a decoder model (GPT-2, LLaMA, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 'gpt2' in model_name.lower():
            model_class = transformers.GPT2Model
            tokenizer_class = transformers.GPT2Tokenizer
        elif 'llama' in model_name.lower():
            model_class = transformers.LlamaModel
            tokenizer_class = transformers.LlamaTokenizer
        elif 'opt' in model_name.lower():
            model_class = transformers.OPTModel
            tokenizer_class = transformers.GPT2Tokenizer
        elif 'bloom' in model_name.lower():
            model_class = transformers.BloomModel
            tokenizer_class = transformers.BloomTokenizerFast
        elif 'falcon' in model_name.lower():
            model_class = transformers.FalconModel
            tokenizer_class = transformers.AutoTokenizer
        else:
            # Use auto classes for other decoder models
            model_class = transformers.AutoModelForCausalLM
            tokenizer_class = transformers.AutoTokenizer
        
        # Load tokenizer
        tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if tokenizer is not None:
            # Tokenize with batch size
            dummy_text = ["Hello, world!"] * batch_size
            dummy_input = tokenizer(dummy_text, return_tensors='pt', padding=True)
            
            # Move inputs to device
            if device != 'cpu':
                for key in dummy_input:
                    if hasattr(dummy_input[key], 'to'):
                        dummy_input[key] = dummy_input[key].to(device)
        else:
            dummy_input = None
        
        metadata = {
            'tokenizer': tokenizer,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_encoder_decoder_model(self, 
                                   model_name: str, 
                                   device: str, 
                                   dtype: Any, 
                                   batch_size: int,
                                   **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load an encoder-decoder model (T5, BART, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 't5' in model_name.lower():
            model_class = transformers.T5Model
            tokenizer_class = transformers.T5Tokenizer
        elif 'bart' in model_name.lower():
            model_class = transformers.BartModel
            tokenizer_class = transformers.BartTokenizer
        elif 'pegasus' in model_name.lower():
            model_class = transformers.PegasusModel
            tokenizer_class = transformers.PegasusTokenizer
        elif 'mbart' in model_name.lower():
            model_class = transformers.MBartModel
            tokenizer_class = transformers.MBartTokenizer
        else:
            # Use auto classes for other encoder-decoder models
            model_class = transformers.AutoModelForSeq2SeqLM
            tokenizer_class = transformers.AutoTokenizer
        
        # Load tokenizer
        tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if tokenizer is not None:
            # Tokenize with batch size
            dummy_text = ["Hello, world!"] * batch_size
            dummy_input = tokenizer(dummy_text, return_tensors='pt', padding=True)
            
            # Add decoder input ids for seq2seq models
            if "t5" in model_name.lower() or "bart" in model_name.lower():
                dummy_input["decoder_input_ids"] = tokenizer(
                    ["Translation:"] * batch_size, return_tensors='pt', padding=True
                ).input_ids
            
            # Move inputs to device
            if device != 'cpu':
                for key in dummy_input:
                    if hasattr(dummy_input[key], 'to'):
                        dummy_input[key] = dummy_input[key].to(device)
        else:
            dummy_input = None
        
        metadata = {
            'tokenizer': tokenizer,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_vision_model(self, 
                         model_name: str, 
                         device: str, 
                         dtype: Any, 
                         batch_size: int,
                         **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a vision model (ViT, DeiT, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        from PIL import Image
        import numpy as np
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 'vit' in model_name.lower():
            model_class = transformers.ViTModel
            processor_class = transformers.ViTImageProcessor
        elif 'deit' in model_name.lower():
            model_class = transformers.DeiTModel
            processor_class = transformers.DeiTImageProcessor
        elif 'beit' in model_name.lower():
            model_class = transformers.BeitModel
            processor_class = transformers.BeitImageProcessor
        elif 'convnext' in model_name.lower():
            model_class = transformers.ConvNextModel
            processor_class = transformers.ConvNextImageProcessor
        else:
            # Use auto classes for other vision models
            model_class = transformers.AutoModel
            processor_class = transformers.AutoImageProcessor
        
        # Load processor
        processor = processor_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if processor is not None:
            try:
                # Create dummy image
                import torch
                dummy_images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(batch_size)]
                
                # Process images
                dummy_input = processor(dummy_images, return_tensors='pt')
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input for vision model: {e}")
                dummy_input = None
        else:
            dummy_input = None
        
        metadata = {
            'processor': processor,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_audio_model(self, 
                        model_name: str, 
                        device: str, 
                        dtype: Any, 
                        batch_size: int,
                        **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load an audio model (Whisper, Wav2Vec2, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        import numpy as np
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 'whisper' in model_name.lower():
            model_class = transformers.WhisperModel
            processor_class = transformers.WhisperProcessor
        elif 'wav2vec2' in model_name.lower():
            model_class = transformers.Wav2Vec2Model
            processor_class = transformers.Wav2Vec2Processor
        elif 'hubert' in model_name.lower():
            model_class = transformers.HubertModel
            processor_class = transformers.Wav2Vec2Processor
        else:
            # Use auto classes for other audio models
            model_class = transformers.AutoModel
            processor_class = transformers.AutoProcessor
        
        # Load processor
        processor = processor_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if processor is not None:
            try:
                import torch
                # Create dummy audio input - 1 second of audio at 16kHz
                sample_rate = 16000
                dummy_audio = [np.random.rand(sample_rate).astype(np.float32) for _ in range(batch_size)]
                
                # Process audio
                dummy_input = processor(dummy_audio, sampling_rate=sample_rate, return_tensors='pt')
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input for audio model: {e}")
                dummy_input = None
        else:
            dummy_input = None
        
        metadata = {
            'processor': processor,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_multimodal_model(self, 
                             model_name: str, 
                             device: str, 
                             dtype: Any, 
                             batch_size: int,
                             **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a multimodal model (CLIP, BLIP, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        import numpy as np
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate model class
        if 'clip' in model_name.lower():
            model_class = transformers.CLIPModel
            processor_class = transformers.CLIPProcessor
        elif 'blip' in model_name.lower():
            model_class = transformers.BlipModel
            processor_class = transformers.BlipProcessor
        elif 'flava' in model_name.lower():
            model_class = transformers.FlavaModel
            processor_class = transformers.FlavaProcessor
        else:
            # Use auto classes for other multimodal models
            model_class = transformers.AutoModel
            processor_class = transformers.AutoProcessor
        
        # Load processor
        processor = processor_class.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        if processor is not None:
            try:
                import torch
                # Create dummy image and text inputs
                dummy_images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(batch_size)]
                dummy_texts = ["Hello, world!"] * batch_size
                
                # Process inputs
                dummy_input = processor(text=dummy_texts, images=dummy_images, return_tensors='pt')
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input for multimodal model: {e}")
                dummy_input = None
        else:
            dummy_input = None
        
        metadata = {
            'processor': processor,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__
        }
        
        return model, metadata
    
    def _load_diffusion_model(self, 
                            model_name: str, 
                            device: str, 
                            dtype: Any, 
                            batch_size: int,
                            **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a diffusion model (Stable Diffusion, etc.).
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Determine the appropriate pipeline class
        if 'stable-diffusion' in model_name.lower():
            try:
                from diffusers import StableDiffusionPipeline
                model_class = StableDiffusionPipeline
                is_diffusers = True
            except ImportError:
                model_class = transformers.AutoModel
                is_diffusers = False
        else:
            # Use auto classes for other diffusion models
            model_class = transformers.AutoModel
            is_diffusers = False
        
        # Load model and processor
        if is_diffusers:
            # Load diffusers pipeline
            model = model_class.from_pretrained(model_name, cache_dir=self.cache_dir)
            processor = None
        else:
            # Load transformers model and processor
            model = model_class.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                **config_kwargs,
                **kwargs
            )
            processor = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        dummy_input = None
        if is_diffusers:
            # Create dummy prompt for diffusers
            dummy_input = {"prompt": ["a photo of a cat"] * batch_size}
        elif processor is not None:
            try:
                import torch
                import numpy as np
                # Create dummy text input
                dummy_texts = ["a photo of a cat"] * batch_size
                
                # Process inputs
                dummy_input = processor(text=dummy_texts, return_tensors='pt')
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input for diffusion model: {e}")
                dummy_input = None
        
        metadata = {
            'processor': processor,
            'dummy_input': dummy_input,
            'model_class': model_class.__name__,
            'is_diffusers': is_diffusers
        }
        
        return model, metadata
    
    def _load_generic_model(self, 
                          model_name: str, 
                          device: str, 
                          dtype: Any, 
                          batch_size: int,
                          **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a generic model with auto classes.
        
        Args:
            model_name: Name of the model or model path
            device: Device to load the model on
            dtype: Data type for the model
            batch_size: Batch size for the model
            **kwargs: Additional arguments to pass to the model loading function
        
        Returns:
            Tuple of (model, metadata)
        """
        import transformers
        
        # Set auto config to pass dtype
        config_kwargs = {}
        if dtype is not None:
            config_kwargs['torch_dtype'] = dtype
        
        # Load model
        model = transformers.AutoModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            **config_kwargs,
            **kwargs
        )
        
        # Load tokenizer or processor
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        except Exception:
            tokenizer = None
        
        if tokenizer is None:
            try:
                processor = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
            except Exception:
                processor = None
        else:
            processor = None
        
        # Move model to device
        if device != 'cpu':
            model = model.to(device)
        
        # Create dummy input for benchmark
        dummy_input = None
        if tokenizer is not None:
            try:
                import torch
                # Tokenize with batch size
                dummy_text = ["Hello, world!"] * batch_size
                dummy_input = tokenizer(dummy_text, return_tensors='pt', padding=True)
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input with tokenizer: {e}")
        elif processor is not None:
            try:
                import torch
                import numpy as np
                # Create dummy input based on processor type
                # This is a fallback that may not work for all models
                dummy_text = ["Hello, world!"] * batch_size
                dummy_input = processor(text=dummy_text, return_tensors='pt')
                
                # Move inputs to device
                if device != 'cpu':
                    for key in dummy_input:
                        if hasattr(dummy_input[key], 'to'):
                            dummy_input[key] = dummy_input[key].to(device)
            except Exception as e:
                logger.warning(f"Failed to create dummy input with processor: {e}")
        
        metadata = {
            'tokenizer': tokenizer,
            'processor': processor,
            'dummy_input': dummy_input,
            'model_class': model.__class__.__name__
        }
        
        return model, metadata

# Function to get a list of priority models for benchmarking
def get_priority_models(priority: str = "high") -> List[str]:
    """
    Get a list of priority models for benchmarking.
    
    Args:
        priority: Priority level (critical, high, medium, low, all)
    
    Returns:
        List of model names
    """
    # Critical priority models (core models for essential functionality)
    critical_models = [
        "bert-base-uncased",
        "gpt2",
        "t5-small",
        "roberta-base",
        "distilbert-base-uncased",
        "facebook/bart-base",
        "microsoft/deberta-base",
        "google/vit-base-patch16-224",
        "openai/clip-vit-base-patch32",
        "facebook/wav2vec2-base-960h",
        "openai/whisper-tiny",
        "meta-llama/Llama-2-7b-hf"
    ]
    
    # High priority models (important models for broad coverage)
    high_models = critical_models + [
        "albert-base-v2",
        "xlm-roberta-base",
        "facebook/opt-350m",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-j-6b",
        "google/flan-t5-small",
        "facebook/deit-small-patch16-224",
        "facebook/convnext-tiny-224",
        "microsoft/beit-base-patch16-224",
        "stabilityai/stable-diffusion-2-base",
        "facebook/blip-image-captioning-base",
        "tiiuae/falcon-7b"
    ]
    
    # Medium priority models (additional models for wider coverage)
    medium_models = high_models + [
        "camembert-base",
        "bert-base-multilingual-cased",
        "flaubert/flaubert_small_cased",
        "allenai/longformer-base-4096",
        "google/electra-small-discriminator",
        "facebook/esm2-t12-35M-UR50D",
        "hf-internal-testing/tiny-random-wav2vec2",
        "facebook/musicgen-small",
        "google/mt5-small",
        "google/pegasus-cnn_dailymail",
        "google/mt5-small",
        "facebook/bart-large-cnn",
        "deepmind/vision-perceiver",
        "facebook/dinov2-small",
        "facebook/dpr-question_encoder-single-nq-base"
    ]
    
    # Low priority models (niche or specialized models)
    low_models = medium_models + [
        "google/bigbird-roberta-base",
        "google/bigbird-pegasus-large-arxiv",
        "google/reformer-crime-and-punishment",
        "funnel-transformer/small-base",
        "facebook/levit-128S",
        "google/vit-hybrid-base-bit-384",
        "facebook/data2vec-audio-base-960h",
        "google/canine-s",
        "google/fnet-base",
        "facebook/fasterformer-128",
        "google/owlvit-base-patch32",
        "facebook/mask2former-swin-base-coco-panoptic",
        "microsoft/mpnet-base",
        "facebook/dpr-ctx_encoder-single-nq-base",
        "bert-base-japanese"
    ]
    
    # Map priority to model list
    priority_map = {
        "critical": critical_models,
        "high": high_models,
        "medium": medium_models,
        "low": low_models,
        "all": low_models  # Same as low for now, but could be extended
    }
    
    return priority_map.get(priority, high_models)

# Utility function to generate benchmark configurations for all models
def generate_model_benchmark_configs(priority: str = "high", 
                                  hardware: List[str] = ["cpu", "cuda"],
                                  batch_sizes: List[int] = [1, 8],
                                  precisions: List[str] = ["fp32"],
                                  test_types: List[str] = ["inference"]) -> List[Dict[str, Any]]:
    """
    Generate benchmark configurations for models based on priority.
    
    Args:
        priority: Priority level (critical, high, medium, low, all)
        hardware: List of hardware backends to benchmark on
        batch_sizes: List of batch sizes to benchmark
        precisions: List of precision formats to benchmark
        test_types: List of test types to run
    
    Returns:
        List of benchmark configurations
    """
    # Get models for the specified priority
    models = get_priority_models(priority)
    
    # Generate configurations
    configs = []
    for model_name in models:
        for hw in hardware:
            for batch_size in batch_sizes:
                for precision in precisions:
                    for test_type in test_types:
                        config = {
                            "model_name": model_name,
                            "hardware": hw,
                            "batch_size": batch_size,
                            "precision": precision,
                            "test_type": test_type,
                        }
                        configs.append(config)
    
    return configs