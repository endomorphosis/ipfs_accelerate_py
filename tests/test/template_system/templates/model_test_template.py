"""
Model test template for generating model-specific test files.

This module provides templates for generating tests for specific model types.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from .base_template import BaseTemplate

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTestTemplate(BaseTemplate):
    """
    Template for model-specific tests.
    
    This template generates tests for specific models, such as BERT, T5, ViT, etc.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the model test template.
        
        Args:
            model_name: Name of the model (e.g., bert-base-uncased)
            model_type: Type of model (text, vision, audio, multimodal)
            **kwargs: Additional template parameters
        """
        super().__init__(model_name, **kwargs)
        self.model_name = model_name
        self.model_type = model_type
        self.framework = kwargs.get('framework', 'transformers')
        self.batch_size = kwargs.get('batch_size', 1)
        
        # Determine appropriate output directory
        if not self.output_dir:
            # Get the model group (e.g., bert, t5, vit)
            model_group = model_name.split('-')[0].lower()
            
            # Map model_type to directory
            type_dir = {
                'text': 'text',
                'vision': 'vision',
                'audio': 'audio',
                'multimodal': 'multimodal'
            }.get(model_type.lower(), 'text')
            
            # Set output directory
            self.output_dir = os.path.join('test', 'models', type_dir, model_group)
    
    def generate_imports(self) -> str:
        """
        Generate model-specific import statements.
        
        Returns:
            Import statements as a string
        """
        imports = super().generate_imports()
        
        # Add framework-specific imports
        if self.framework == 'transformers':
            imports += """
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from common.hardware_detection import detect_hardware, skip_if_no_cuda
from common.model_helpers import load_model, get_sample_inputs_for_model
"""
        elif self.framework == 'torch':
            imports += """
import torch
import torchvision
from common.hardware_detection import detect_hardware, skip_if_no_cuda
"""
        elif self.framework == 'tensorflow':
            imports += """
import tensorflow as tf
from common.hardware_detection import detect_hardware
"""
        elif self.framework == 'onnx':
            imports += """
import numpy as np
import onnxruntime as ort
from common.hardware_detection import detect_hardware
"""
        
        return imports
    
    def generate_test_class(self) -> str:
        """
        Generate the model test class.
        
        Returns:
            Test class content as a string
        """
        # Create a class name from the model name
        class_name = ''.join(word.capitalize() for word in 
                        self.model_name.replace('-', '_').split('_'))
        
        # Basic class structure
        if self.model_type == 'text':
            return self._generate_text_model_test_class(class_name)
        elif self.model_type == 'vision':
            return self._generate_vision_model_test_class(class_name)
        elif self.model_type == 'audio':
            return self._generate_audio_model_test_class(class_name)
        elif self.model_type == 'multimodal':
            return self._generate_multimodal_model_test_class(class_name)
        else:
            return self._generate_generic_model_test_class(class_name)
    
    def _generate_text_model_test_class(self, class_name: str) -> str:
        """Generate a test class for text models."""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.model_name} model.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test with model details and hardware detection.\"\"\"
        self.model_name = "{self.model_name}"
        self.model_type = "{self.model_type}"
        self.setup_hardware()
    
    def setup_hardware(self):
        \"\"\"Set up hardware detection for the template.\"\"\"
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # Qualcomm AI Engine support
        self.has_qualcomm = 'qti' in sys.modules or 'qnn_wrapper' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {{self.device}}")
        
    def get_model(self):
        \"\"\"Load model from HuggingFace.\"\"\"
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            return None, None
    
    @pytest.mark.model
    @pytest.mark.text
    def test_basic_inference(self):
        \"\"\"Run a basic inference test with the model.\"\"\"
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            pytest.skip("Failed to load model or tokenizer")
        
        try:
            # Prepare input
            text = "This is a sample text for testing the {self.model_name} model."
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Output shape: {{outputs.last_hidden_state.shape}}")
            
            logger.info("Basic inference test passed")
        except Exception as e:
            logger.error(f"Error during inference: {{e}}")
            pytest.fail(f"Inference failed: {{e}}")
    
    @pytest.mark.model
    @pytest.mark.text
    @pytest.mark.slow
    def test_batch_inference(self):
        \"\"\"Run a batch inference test with the model.\"\"\"
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            pytest.skip("Failed to load model or tokenizer")
        
        try:
            # Prepare batch input
            texts = [
                "This is the first sample text for testing batch inference.",
                "This is the second sample text for testing batch inference."
            ]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == len(texts), f"Batch size should be {{len(texts)}}"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Batch output shape: {{outputs.last_hidden_state.shape}}")
            
            logger.info("Batch inference test passed")
        except Exception as e:
            logger.error(f"Error during batch inference: {{e}}")
            pytest.fail(f"Batch inference failed: {{e}}")
    
    @pytest.mark.model
    @pytest.mark.text
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device):
        \"\"\"Test model compatibility with different devices.\"\"\"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from transformers import AutoModel
            
            # Load model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            
            logger.info(f"Model loaded on {{device}}")
            assert model.device.type == device, f"Model should be on {{device}}"
            
            logger.info(f"Device compatibility test passed for {{device}}")
        except Exception as e:
            logger.error(f"Error loading model on {{device}}: {{e}}")
            pytest.fail(f"Device compatibility test failed for {{device}}: {{e}}")
"""
    
    def _generate_vision_model_test_class(self, class_name: str) -> str:
        """Generate a test class for vision models."""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.model_name} vision model.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test with model details and hardware detection.\"\"\"
        self.model_name = "{self.model_name}"
        self.model_type = "{self.model_type}"
        self.setup_hardware()
    
    def setup_hardware(self):
        \"\"\"Set up hardware detection for the template.\"\"\"
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {{self.device}}")
    
    def get_model(self):
        \"\"\"Load vision model.\"\"\"
        try:
            from transformers import AutoFeatureExtractor, AutoModel
            
            # Get feature extractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, feature_extractor
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            return None, None
    
    def get_sample_image(self):
        \"\"\"Get a sample image for testing.\"\"\"
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Check if a test image already exists
            if os.path.exists("test.jpg"):
                return Image.open("test.jpg")
            
            # Download a sample image
            url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_images/000000039769.jpg"
            response = requests.get(url)
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error getting sample image: {{e}}")
            return None
    
    @pytest.mark.model
    @pytest.mark.vision
    def test_basic_inference(self):
        \"\"\"Run a basic inference test with the model.\"\"\"
        model, feature_extractor = self.get_model()
        
        if model is None or feature_extractor is None:
            pytest.skip("Failed to load model or feature extractor")
        
        # Get sample image
        image = self.get_sample_image()
        if image is None:
            pytest.skip("Failed to get sample image")
        
        try:
            # Prepare input
            inputs = feature_extractor(images=image, return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            logger.info(f"Output shape: {{outputs.last_hidden_state.shape}}")
            
            logger.info("Basic inference test passed")
        except Exception as e:
            logger.error(f"Error during inference: {{e}}")
            pytest.fail(f"Inference failed: {{e}}")
    
    @pytest.mark.model
    @pytest.mark.vision
    @pytest.mark.slow
    def test_batch_inference(self):
        \"\"\"Run a batch inference test with the model.\"\"\"
        model, feature_extractor = self.get_model()
        
        if model is None or feature_extractor is None:
            pytest.skip("Failed to load model or feature extractor")
        
        # Get sample image
        image = self.get_sample_image()
        if image is None:
            pytest.skip("Failed to get sample image")
        
        try:
            # Create a batch of the same image
            images = [image] * 2
            
            # Prepare input
            inputs = feature_extractor(images=images, return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == len(images), f"Batch size should be {{len(images)}}"
            logger.info(f"Batch output shape: {{outputs.last_hidden_state.shape}}")
            
            logger.info("Batch inference test passed")
        except Exception as e:
            logger.error(f"Error during batch inference: {{e}}")
            pytest.fail(f"Batch inference failed: {{e}}")
"""
    
    def _generate_audio_model_test_class(self, class_name: str) -> str:
        """Generate a test class for audio models."""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.model_name} audio model.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test with model details and hardware detection.\"\"\"
        self.model_name = "{self.model_name}"
        self.model_type = "{self.model_type}"
        self.setup_hardware()
    
    def setup_hardware(self):
        \"\"\"Set up hardware detection for the template.\"\"\"
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {{self.device}}")
    
    def get_model(self):
        \"\"\"Load audio model.\"\"\"
        try:
            from transformers import AutoProcessor, AutoModel
            
            # Get processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            return None, None
    
    def get_sample_audio(self):
        \"\"\"Get a sample audio for testing.\"\"\"
        try:
            import librosa
            
            # Check if test audio already exists
            if os.path.exists("test.wav"):
                return librosa.load("test.wav", sr=16000)[0]
            elif os.path.exists("test.mp3"):
                return librosa.load("test.mp3", sr=16000)[0]
            
            # Create a simple sine wave if no test audio is available
            duration = 3  # seconds
            sample_rate = 16000
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            return audio
        except Exception as e:
            logger.error(f"Error getting sample audio: {{e}}")
            return None
    
    @pytest.mark.model
    @pytest.mark.audio
    def test_basic_inference(self):
        \"\"\"Run a basic inference test with the model.\"\"\"
        model, processor = self.get_model()
        
        if model is None or processor is None:
            pytest.skip("Failed to load model or processor")
        
        # Get sample audio
        audio = self.get_sample_audio()
        if audio is None:
            pytest.skip("Failed to get sample audio")
        
        try:
            # Prepare input
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            # The actual output attribute depends on the model
            logger.info(f"Output keys: {{list(outputs.keys() if hasattr(outputs, 'keys') else outputs._fields)}}")
            
            logger.info("Basic inference test passed")
        except Exception as e:
            logger.error(f"Error during inference: {{e}}")
            pytest.fail(f"Inference failed: {{e}}")
    
    @pytest.mark.model
    @pytest.mark.audio
    @pytest.mark.slow
    def test_batch_inference(self):
        \"\"\"Run a batch inference test with the model.\"\"\"
        model, processor = self.get_model()
        
        if model is None or processor is None:
            pytest.skip("Failed to load model or processor")
        
        # Get sample audio
        audio = self.get_sample_audio()
        if audio is None:
            pytest.skip("Failed to get sample audio")
        
        try:
            # Create a batch of the same audio
            audios = [audio] * 2
            
            # Prepare input (specific method depends on the processor)
            inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            logger.info(f"Output keys: {{list(outputs.keys() if hasattr(outputs, 'keys') else outputs._fields)}}")
            
            logger.info("Batch inference test passed")
        except Exception as e:
            logger.error(f"Error during batch inference: {{e}}")
            pytest.fail(f"Batch inference failed: {{e}}")
"""
    
    def _generate_multimodal_model_test_class(self, class_name: str) -> str:
        """Generate a test class for multimodal models."""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.model_name} multimodal model.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test with model details and hardware detection.\"\"\"
        self.model_name = "{self.model_name}"
        self.model_type = "{self.model_type}"
        self.setup_hardware()
    
    def setup_hardware(self):
        \"\"\"Set up hardware detection for the template.\"\"\"
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {{self.device}}")
    
    def get_model(self):
        \"\"\"Load multimodal model.\"\"\"
        try:
            from transformers import AutoProcessor, AutoModel
            
            # Get processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            return None, None
    
    def get_sample_data(self):
        \"\"\"Get sample data for testing.\"\"\"
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Check if a test image already exists
            if os.path.exists("test.jpg"):
                image = Image.open("test.jpg")
            else:
                # Download a sample image
                url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_images/000000039769.jpg"
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            
            # Sample text
            text = "A picture of a cat"
            
            return {{
                'image': image,
                'text': text
            }}
        except Exception as e:
            logger.error(f"Error getting sample data: {{e}}")
            return None
    
    @pytest.mark.model
    @pytest.mark.multimodal
    def test_basic_inference(self):
        \"\"\"Run a basic inference test with the model.\"\"\"
        model, processor = self.get_model()
        
        if model is None or processor is None:
            pytest.skip("Failed to load model or processor")
        
        # Get sample data
        sample_data = self.get_sample_data()
        if sample_data is None:
            pytest.skip("Failed to get sample data")
        
        try:
            # Prepare input
            inputs = processor(text=sample_data['text'], images=sample_data['image'], return_tensors="pt")
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            logger.info(f"Output keys: {{list(outputs.keys() if hasattr(outputs, 'keys') else outputs._fields)}}")
            
            logger.info("Basic inference test passed")
        except Exception as e:
            logger.error(f"Error during inference: {{e}}")
            pytest.fail(f"Inference failed: {{e}}")
    
    @pytest.mark.model
    @pytest.mark.multimodal
    @pytest.mark.slow
    def test_batch_inference(self):
        \"\"\"Run a batch inference test with the model.\"\"\"
        model, processor = self.get_model()
        
        if model is None or processor is None:
            pytest.skip("Failed to load model or processor")
        
        # Get sample data
        sample_data = self.get_sample_data()
        if sample_data is None:
            pytest.skip("Failed to get sample data")
        
        try:
            # Create a batch
            images = [sample_data['image']] * 2
            texts = [sample_data['text'], "Another text description"]
            
            # Prepare input
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            logger.info(f"Output keys: {{list(outputs.keys() if hasattr(outputs, 'keys') else outputs._fields)}}")
            
            logger.info("Batch inference test passed")
        except Exception as e:
            logger.error(f"Error during batch inference: {{e}}")
            pytest.fail(f"Batch inference failed: {{e}}")
"""
    
    def _generate_generic_model_test_class(self, class_name: str) -> str:
        """Generate a generic test class for any model type."""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.model_name} model.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test with model details and hardware detection.\"\"\"
        self.model_name = "{self.model_name}"
        self.model_type = "{self.model_type}"
        self.setup_hardware()
    
    def setup_hardware(self):
        \"\"\"Set up hardware detection for the template.\"\"\"
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {{self.device}}")
    
    def get_model(self):
        \"\"\"Load model.\"\"\"
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Try to determine the model type
            if 'bert' in self.model_name.lower() or 't5' in self.model_name.lower() or 'gpt' in self.model_name.lower():
                # Text model
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                return model, tokenizer
            else:
                # Generic model
                model = AutoModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                return model, None
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            return None, None
    
    @pytest.mark.model
    def test_basic_load(self):
        \"\"\"Test basic model loading.\"\"\"
        model, _ = self.get_model()
        
        if model is None:
            pytest.skip("Failed to load model")
        
        logger.info(f"Model {self.model_name} loaded successfully")
        
        # Check model properties
        logger.info(f"Model type: {{type(model).__name__}}")
        logger.info(f"Model device: {{model.device}}")
        
        assert model.device.type == self.device, f"Model should be on {{self.device}}"
        
        logger.info("Model load test passed")
"""
    
    def customize_content(self, content: str) -> str:
        """
        Add model-specific customizations.
        
        Args:
            content: The generated content
            
        Returns:
            The customized content
        """
        content = super().customize_content(content)
        
        # Add model-specific imports
        if 'bert' in self.model_name.lower():
            content = content.replace('import torch', 'import torch\nfrom transformers import BertModel, BertTokenizer')
        elif 't5' in self.model_name.lower():
            content = content.replace('import torch', 'import torch\nfrom transformers import T5Model, T5Tokenizer')
        elif 'vit' in self.model_name.lower():
            content = content.replace('import torch', 'import torch\nfrom transformers import ViTModel, ViTFeatureExtractor')
        elif 'whisper' in self.model_name.lower():
            content = content.replace('import torch', 'import torch\nfrom transformers import WhisperModel, WhisperProcessor')
        elif 'gpt' in self.model_name.lower():
            content = content.replace('import torch', 'import torch\nfrom transformers import GPT2Model, GPT2Tokenizer')
        
        return content
    
    def before_generate(self) -> None:
        """Set up before generating the template."""
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def after_generate(self) -> None:
        """Clean up after generating the template."""
        # Log the generated file
        output_path = self.get_output_path()
        logger.info(f"Generated model test file: {output_path}")
        
        # Add model-specific metadata if needed
        # ...