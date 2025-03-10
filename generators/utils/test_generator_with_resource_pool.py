#!/usr/bin/env python
"""
Test generator with resource pool integration and hardware awareness.
This script generates optimized test files for models with hardware-specific configurations.
"""

import os
import sys
import json
import logging
import argparse
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import required components
try:
    from generators.utils.resource_pool import get_global_resource_pool
    # For syntactic validation only, not trying to import 
    # hardware_detection that doesn't exist here yet
    CUDA, ROCM, MPS, OPENVINO, CPU, WEBNN, WEBGPU = "cuda", "rocm", "mps", "openvino", "cpu", "webnn", "webgpu"
except ImportError as e:
    logger.error(f"Required module not found: {e}")
    logger.error("Make sure resource_pool.py is in your path")
    CUDA, ROCM, MPS, OPENVINO, CPU, WEBNN, WEBGPU = "cuda", "rocm", "mps", "openvino", "cpu", "webnn", "webgpu"

# Try to import model classification components
try:
    from model_family_classifier import classify_model, ModelFamilyClassifier
except ImportError as e:
    logger.warning(f"Model family classifier not available: {e}")
    logger.warning("Will use basic model classification based on model name")
    classify_model = None
    ModelFamilyClassifier = None

# Try to import hardware-model integration
try:
    from hardware_model_integration import (
        HardwareAwareModelClassifier,
        get_hardware_aware_model_classification
    )
    HARDWARE_MODEL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware-model integration not available: {e}")
    logger.warning("Will use basic hardware-model integration")
    HARDWARE_MODEL_INTEGRATION_AVAILABLE = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test generator with resource pool integration")
    parser.add_argument("--model", type=str, required=True, help="Model name to generate tests for")
    parser.add_argument("--output-dir", type=str, default="./skills", help="Output directory for generated tests")
    parser.add_argument("--timeout", type=float, default=0.1, help="Resource cleanup timeout (minutes)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear resource cache before running")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], 
    default="auto", help="Force specific device for testing")
    parser.add_argument("--hw-cache", type=str, help="Path to hardware detection cache")
    parser.add_argument("--model-db", type=str, help="Path to model database")
    parser.add_argument("--use-model-family", action="store_true", 
    help="Use model family classifier for optimal template selection")
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment and configure logging"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Clear resource pool if requested:
    if args.clear_cache:
        pool = get_global_resource_pool()
        pool.clear()
        logger.info("Resource pool cleared")

def load_dependencies():
    """Load common dependencies with resource pooling"""
    logger.info("Loading dependencies using resource pool")
    pool = get_global_resource_pool()
    
    # Load common libraries
    torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
    transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
    
    # Check if dependencies were loaded successfully:
    if torch is None or transformers is None:
        logger.error("Failed to load required dependencies")
        return False
    
    logger.info("Dependencies loaded successfully")
    return True

def get_hardware_aware_classification(model_name, hw_cache_path=None, model_db_path=None):
    """
    Get hardware-aware model classification
    
    Args:
        model_name: Model name to classify
        hw_cache_path: Optional path to hardware detection cache
        model_db_path: Optional path to model database
        
    Returns:
        Dictionary with hardware-aware classification results
    """
    # Use hardware-model integration if available:
    if HARDWARE_MODEL_INTEGRATION_AVAILABLE:
        try:
            logger.info("Using hardware-model integration for classification")
            classification = get_hardware_aware_model_classification(
                model_name=model_name,
                hw_cache_path=hw_cache_path,
                model_db_path=model_db_path
            )
            return classification
        except Exception as e:
            logger.warning(f"Error using hardware-model integration: {e}")
    
    # Fallback to simpler classification with hardware detection
    logger.info("Using basic hardware-model integration")
    
    # Simplified hardware detection for syntax validation
    hardware_result = {
        "cuda": HAS_CUDA if 'HAS_CUDA' in globals() else False,
        "mps": HAS_MPS if 'HAS_MPS' in globals() else False,
        "best_available": "cpu",
        "torch_device": "cpu"
    }
    hardware_info = {k: v for k, v in hardware_result.items() if isinstance(v, bool)}
    best_hardware = hardware_result.get('best_available', CPU)
    torch_device = hardware_result.get('torch_device', 'cpu')
    
    # Classify model if classifier is available
    model_family = "default"
    subfamily = None
    if classify_model:
        try:
            # Get hardware compatibility information for more accurate classification
            hw_compatibility = {
                "cuda": {"compatible": hardware_info.get("cuda", False)},
                "mps": {"compatible": hardware_info.get("mps", False)},
                "rocm": {"compatible": hardware_info.get("rocm", False)},
                "openvino": {"compatible": hardware_info.get("openvino", False)},
                "webnn": {"compatible": hardware_info.get("webnn", False)},
                "webgpu": {"compatible": hardware_info.get("webgpu", False)}
            }
            
            # Call classify_model with model name and hardware compatibility
            classification = classify_model(
                model_name=model_name,
                hw_compatibility=hw_compatibility,
                model_db_path=model_db_path
            )
            
            model_family = classification.get("family", "default")
            subfamily = classification.get("subfamily")
            confidence = classification.get("confidence", 0)
            logger.info(f"Model classified as: {model_family} (subfamily: {subfamily}, confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"Error classifying model: {e}")
    
    # Build and return classification dictionary
    return {
        "family": model_family,
        "subfamily": subfamily,
        "best_hardware": best_hardware,
        "torch_device": torch_device,
        "hardware_info": hardware_info
    }

def generate_test_file(model_name, output_dir="./", model_family="default",
                      model_subfamily=None, hardware_info=None):
    """
    Generate test file for a model with hardware-specific configurations.
    
    Args:
        model_name: Name of the model to generate tests for
        output_dir: Directory to write the test file to
        model_family: Model family for template selection
        model_subfamily: Optional model subfamily for template selection
        hardware_info: Dictionary with hardware availability information
        
    Returns:
        Path to the generated test file
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name
    normalized_name = model_name.replace("/", "_").replace("-", "_").lower()
    file_name = f"test_hf_{normalized_name}.py"
    file_path = os.path.join(output_dir, file_name)
    
    # Prepare hardware support information
    if hardware_info is None:
        hardware_info = {}
    
    best_hardware = hardware_info.get("best_hardware", "cpu")
    torch_device = hardware_info.get("torch_device", "cpu")
    
    # Determine available hardware for import statements
    has_cuda = hardware_info.get("cuda", False)
    has_mps = hardware_info.get("mps", False)
    has_rocm = hardware_info.get("rocm", False)
    has_openvino = hardware_info.get("openvino", False)
    has_webnn = hardware_info.get("webnn", False)
    has_webgpu = hardware_info.get("webgpu", False)
    
    # Prepare template context
    context = {
        "model_name": model_name,
        "model_family": model_family,
        "model_subfamily": model_subfamily,
        "normalized_name": normalized_name,
        "best_hardware": best_hardware,
        "torch_device": torch_device,
        "has_cuda": has_cuda,
        "has_mps": has_mps,
        "has_rocm": has_rocm,
        "has_openvino": has_openvino,
        "has_webnn": has_webnn,
        "has_webgpu": has_webgpu,
        "generated_at": datetime.now().isoformat(),
        "generator": __file__
    }
    
    # Select appropriate template based on model family
    template = get_template_for_model(model_family, model_subfamily)
    
    # Render template with context
    test_content = render_template(template, context)
    
    # Write test file
    with open(file_path, "w") as f:
        f.write(test_content)
    
    logger.info(f"Generated test file: {file_path}")
    return file_path

def get_template_for_model(model_family, model_subfamily=None):
    """
    Select appropriate template based on model family and subfamily.
    
    Args:
        model_family: Model family (e.g., "bert", "t5", "vit")
        model_subfamily: Optional model subfamily
        
    Returns:
        Template string for the model family
    """
    # Basic template selection based on model family
    if model_family == "bert":
        template = TEMPLATES["bert"]
    elif model_family == "t5":
        template = TEMPLATES["t5"]
    elif model_family == "vit":
        template = TEMPLATES["vit"]
    elif model_family == "clip":
        template = TEMPLATES["clip"]
    elif model_family == "llama":
        template = TEMPLATES["llama"]
    elif model_family == "whisper":
        template = TEMPLATES["whisper"]
    elif model_family == "wav2vec2":
        template = TEMPLATES["wav2vec2"]
    elif model_family in ["clap", "audio"]:
        template = TEMPLATES["clap"]
    else:
        # Default to generic text model
        template = TEMPLATES["default"]
    
    return template

def render_template(template, context):
    """
    Render a template with the given context.
    
    Args:
        template: Template string
        context: Dictionary with variables for template rendering
        
    Returns:
        Rendered template string
    """
    # Simple template rendering with string formatting
    rendered = template
    for key, value in context.items():
        placeholder = f"{{{key}}}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    
    return rendered

# Template definitions
TEMPLATES = {
    "default": '''#!/usr/bin/env python3
"""
Test for {model_name} with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import os
import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test {model_name} with resource pool integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        """Test that model loaded successfully."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        """Test basic inference."""
        # Prepare input
        text = "This is a test."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
''',
    
    # More templates for specific model families would go here
    "bert": '''#!/usr/bin/env python3
"""
BERT model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import os
import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test {model_name} with resource pool integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        """Test that model loaded successfully."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        """Test basic inference."""
        # Prepare input
        text = "This is a test."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Test embedding extraction
        embeddings = outputs.last_hidden_state
        self.assertEqual(embeddings.shape[0], 1)  # Batch size
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
''',
    
    # Simple template for T5
    "t5": '''#!/usr/bin/env python3
"""
T5 model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} T5 model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModelForSeq2SeqLM.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_inference(self):
        """Test inference with T5 model"""
        input_text = "translate English to German: Hello, how are you?"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Decode output
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Translation: {decoded}")
        
        self.assertTrue(decoded)

if __name__ == "__main__":
    unittest.main()
''',
    
    # Simple template for CLIP
    "clip": '''#!/usr/bin/env python3
"""
CLIP model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} CLIP model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        cls.PIL = cls.pool.get_resource("PIL", constructor=lambda: __import__("PIL"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None or cls.PIL is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.CLIPProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.CLIPModel.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_text_inference(self):
        """Test text feature extraction"""
        text = ["a photo of a cat", "a photo of a dog"]
        inputs = self.processor(text=text, images=None, return_tensors="pt", padding=True)
        
        # Move to device
        text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        
        with self.torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        
        self.assertEqual(text_features.shape[0], 2)  # Batch size
        
    def create_dummy_image(self):
        """Create a dummy image for testing"""
        from PIL import Image
        import numpy as np
        
        # Create a dummy image (3x224x224)
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        return image

if __name__ == "__main__":
    unittest.main()
''',
    
    # Simple template for vision models
    "vit": '''#!/usr/bin/env python3
"""
Vision Transformer model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} ViT model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        cls.PIL = cls.pool.get_resource("PIL", constructor=lambda: __import__("PIL"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None or cls.PIL is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoFeatureExtractor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModelForImageClassification.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def create_dummy_image(self):
        """Create a dummy image for testing"""
        from PIL import Image
        import numpy as np
        
        # Create a dummy image (3x224x224)
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        return image
    
    def test_inference(self):
        """Test inference with ViT model"""
        image = self.create_dummy_image()
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        self.assertEqual(len(logits.shape), 2)  # [batch_size, num_classes]

if __name__ == "__main__":
    unittest.main()
''',

    # Add other templates for different model types
    "whisper": '''#!/usr/bin/env python3
"""
Whisper model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} Whisper model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        cls.numpy = cls.pool.get_resource("numpy", constructor=lambda: __import__("numpy"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None or cls.numpy is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.WhisperProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.WhisperForConditionalGeneration.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def create_dummy_audio(self):
        """Create dummy audio for testing"""
        import numpy as np
        
        # Create 2 seconds of dummy audio at 16kHz
        sample_rate = 16000
        dummy_audio = np.random.random(2 * sample_rate).astype(np.float32)
        return dummy_audio, sample_rate
    
    def test_inference(self):
        """Test inference with Whisper model"""
        audio, sample_rate = self.create_dummy_audio()
        
        # Process inputs
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            # Test inference with a smaller max length for speed
            generated_ids = self.model.generate(inputs.input_features, max_length=20)
            
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Transcription: {transcription}")
        
        self.assertIsInstance(transcription, str)

if __name__ == "__main__":
    unittest.main()
''',

    "wav2vec2": '''#!/usr/bin/env python3
"""
Wav2Vec2 model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} Wav2Vec2 model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        cls.numpy = cls.pool.get_resource("numpy", constructor=lambda: __import__("numpy"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None or cls.numpy is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.Wav2Vec2Processor.from_pretrained("{model_name}")
            cls.model = cls.transformers.Wav2Vec2ForCTC.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def create_dummy_audio(self):
        """Create dummy audio for testing"""
        import numpy as np
        
        # Create 2 seconds of dummy audio at 16kHz
        sample_rate = 16000
        dummy_audio = np.random.random(2 * sample_rate).astype(np.float32)
        return dummy_audio, sample_rate
    
    def test_inference(self):
        """Test inference with Wav2Vec2 model"""
        audio, sample_rate = self.create_dummy_audio()
        
        # Process inputs
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        self.assertTrue(len(logits.shape) >= 2)  # [batch_size, sequence_length, vocab_size]

if __name__ == "__main__":
    unittest.main()
''',

    "llama": '''#!/usr/bin/env python3
"""
LLaMA model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} LLaMA model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModelForCausalLM.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_inference(self):
        """Test inference with LLaMA model"""
        prompt = "Hello, my name is"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            # Generate only a few tokens for testing
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated: {generated_text}")
        
        self.assertTrue(len(generated_text) > len(prompt))

if __name__ == "__main__":
    unittest.main()
''',

    "clap": '''#!/usr/bin/env python3
"""
CLAP model test with resource pool integration.
Generated by test_generator_with_resource_pool.py on {generated_at}
"""

import unittest
import logging
from generators.utils.resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test for {model_name} CLAP model"""
    
    @classmethod
    def setUpClass(cls):
        # Get dependencies from resource pool
        cls.pool = get_global_resource_pool()
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        cls.numpy = cls.pool.get_resource("numpy", constructor=lambda: __import__("numpy"))
        
        # Check dependencies
        if cls.torch is None or cls.transformers is None or cls.numpy is None:
            raise unittest.SkipTest("Dependencies not available")
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.ClapModel.from_pretrained("{model_name}")
            
            # Move to device
            cls.device = "{torch_device}"
            if cls.device \!= "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def create_dummy_audio(self):
        """Create dummy audio for testing"""
        import numpy as np
        
        # Create 2 seconds of dummy audio at 48kHz (CLAP uses higher sample rate)
        sample_rate = 48000
        dummy_audio = np.random.random(2 * sample_rate).astype(np.float32)
        return dummy_audio, sample_rate
    
    def test_text_inference(self):
        """Test text feature extraction"""
        text = ["piano", "drum", "human voice", "violin"]
        
        # Process text inputs
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        
        # Move to device
        if self.device \!= "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        self.assertEqual(text_features.shape[0], 4)  # Batch size

if __name__ == "__main__":
    unittest.main()
'''
}

def main():
    """Main function."""
    args = parse_args()
    setup_environment(args)
    
    # Load dependencies
    if not load_dependencies():
        logger.error("Failed to load dependencies. Exiting.")
        return 1
    
    # Get hardware-aware classification
    classification = get_hardware_aware_classification(
        model_name=args.model,
        hw_cache_path=args.hw_cache,
        model_db_path=args.model_db
    )
    
    # Generate test file
    output_file = generate_test_file(
        model_name=args.model,
        output_dir=args.output_dir,
        model_family=classification.get("family", "default"),
        model_subfamily=classification.get("subfamily"),
        hardware_info=classification
    )
    
    logger.info(f"Test file generated successfully: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
