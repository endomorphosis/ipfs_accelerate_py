#!/usr/bin/env python3
"""
Architecture-aware test generator for HuggingFace models.
This generator creates test files for different model architectures with proper indentation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Template directory - look for templates in the current directory or in a templates subfolder
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

# Architecture mapping for templates
ARCHITECTURE_MAPPING = {
    # Encoder-only models
    "bert": "encoder_only",
    "roberta": "encoder_only",
    "albert": "encoder_only",
    "electra": "encoder_only",
    "camembert": "encoder_only",
    "distilbert": "encoder_only",
    
    # Decoder-only models
    "gpt2": "decoder_only",
    "gptj": "decoder_only",
    "gpt_neo": "decoder_only",
    "llama": "decoder_only",
    "opt": "decoder_only",
    "falcon": "decoder_only",
    "mistral": "decoder_only",
    "phi": "decoder_only",
    
    # Encoder-decoder models
    "t5": "encoder_decoder",
    "bart": "encoder_decoder",
    "pegasus": "encoder_decoder",
    "mbart": "encoder_decoder",
    "m2m_100": "encoder_decoder",
    "led": "encoder_decoder",
    
    # Vision models
    "vit": "vision",
    "swin": "vision",
    "beit": "vision",
    "deit": "vision",
    "convnext": "vision",
    "sam": "vision",
    
    # Multimodal models
    "clip": "multimodal",
    "blip": "multimodal",
    "llava": "multimodal",
    "flava": "multimodal",
    "idefics": "multimodal",
    
    # Audio models
    "wav2vec2": "audio",
    "hubert": "audio",
    "whisper": "audio",
    "clap": "audio",
    "encodec": "audio"
}

# Template models for each architecture
DEFAULT_TEMPLATES = {
    "encoder_only": "bert",
    "decoder_only": "gpt2",
    "encoder_decoder": "t5",
    "vision": "vit",
    "multimodal": "clip",
    "audio": "wav2vec2"
}

# Model registry templates for different architectures
MODEL_REGISTRY_TEMPLATES = {
    "encoder_only": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}""",
    "decoder_only": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}""",
    "encoder_decoder": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}""",
    "vision": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}""",
    "multimodal": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}""",
    "audio": """# Models registry
{model_upper}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{model_description}",
        "class": "{model_class}",
    }}
}}"""
}

# Test class templates for different architectures
TEST_CLASS_TEMPLATES = {
    "encoder_only": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "fill-mask"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "The man worked as a [MASK]."

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}""",
    "decoder_only": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "text-generation"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "Once upon a time,"

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}""",
    "encoder_decoder": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "translation_en_to_de"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "Hello, how are you?"

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}""",
    "vision": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "image-classification"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Test inputs will be created during testing
        self.test_input = None

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}""",
    "multimodal": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "image-to-text"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Test inputs will be created during testing
        self.test_text = "A photo of"
        self.test_input = None

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}""",
    "audio": """class Test{model_pascal}Models:
    \"\"\"Test class for {model_family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {model_upper}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {model_upper}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {model_upper}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "automatic-speech-recognition"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Test inputs will be created during testing
        self.test_input = None

        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}"""
}

# Pipeline test method templates for different architectures
PIPELINE_TEST_TEMPLATES = {
    "encoder_only": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input
            # For text models
            pipeline_input = self.test_text

            # Run inference
            output = pipeline(pipeline_input)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results""",
    "decoder_only": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Set padding token for decoder-only models
            if pipeline.tokenizer.pad_token is None:
                pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
            
            # Prepare test input
            pipeline_input = self.test_text
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": 20,
                "do_sample": False
            }

            # Run inference
            output = pipeline(pipeline_input, **gen_kwargs)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results""",
    "encoder_decoder": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input
            pipeline_input = self.test_text
            
            # Generation parameters for encoder-decoder models
            gen_kwargs = {
                "max_length": 50,
                "do_sample": False
            }

            # Run inference
            output = pipeline(pipeline_input, **gen_kwargs)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results""",
    "vision": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input for vision models
            if HAS_TORCH:
                # Create a random image tensor
                self.test_input = torch.randn(1, 3, 224, 224)
                pipeline_input = self.test_input
            else:
                # Skip inference if torch is not available
                pipeline_input = None
                logger.warning("Skipping inference because torch is not available")

            # Run inference if input is available
            if pipeline_input is not None:
                output = pipeline(pipeline_input)
                
                # Store results
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "none"
            else:
                # Mark as success but note that inference was skipped
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "skipped_inference"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results""",
    "multimodal": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input for multimodal models
            if HAS_TORCH:
                # Create a random image tensor
                self.test_input = {
                    "text": self.test_text,
                    "image": torch.randn(1, 3, 224, 224)
                }
                pipeline_input = self.test_input
            else:
                # Skip inference if torch is not available
                pipeline_input = None
                logger.warning("Skipping inference because torch is not available")

            # Run inference if input is available
            if pipeline_input is not None:
                output = pipeline(pipeline_input)
                
                # Store results
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "none"
            else:
                # Mark as success but note that inference was skipped
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "skipped_inference"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results""",
    "audio": """def test_pipeline(self, device="auto"):
        \"\"\"Test the model using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input for audio models
            if HAS_TORCH:
                # Create a random audio tensor (sample_rate=16000, 2 seconds of audio)
                sample_rate = 16000
                self.test_input = torch.randn(sample_rate * 2)
                pipeline_input = self.test_input
            else:
                # Skip inference if torch is not available
                pipeline_input = None
                logger.warning("Skipping inference because torch is not available")

            # Run inference if input is available
            if pipeline_input is not None:
                output = pipeline(pipeline_input)
                
                # Store results
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "none"
            else:
                # Mark as success but note that inference was skipped
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "skipped_inference"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results"""
}

# Run tests method template (shared across architectures)
RUN_TESTS_TEMPLATE = """def run_tests(self, all_hardware=False):
        \"\"\"Run all tests for this model.\"\"\"
        # Test on default device
        self.test_pipeline()
        
        # Build results
        return {
            "results": self.results,
            "examples": self.examples,
            "hardware": HW_CAPABILITIES,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        }"""

# Main method template (shared across architectures)
MAIN_TEMPLATE = """def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {model_family} models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled")
    
    # Run test
    model_id = args.model or "{model_id}"
    tester = Test{model_pascal}Models(model_id)
    results = tester.run_tests()
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values())
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {{test_name}}: {{result.get('pipeline_error', 'Unknown error')}}")

if __name__ == "__main__":
    main()"""

# File header template
HEADER_TEMPLATE = """#!/usr/bin/env python3

import os
import sys
import json
import time
import datetime
import logging
import argparse
import traceback
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
        
        # Check MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            capabilities["mps"] = torch.mps.is_available()
    
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()"""

# Model-specific constants for different architectures
MODEL_CONSTANTS = {
    "encoder_only": {
        "bert": {
            "model_id": "bert-base-uncased",
            "model_class": "BertModel",
            "model_description": "bert base model"
        },
        "roberta": {
            "model_id": "roberta-base",
            "model_class": "RobertaModel",
            "model_description": "roberta base model"
        },
        "albert": {
            "model_id": "albert-base-v2",
            "model_class": "AlbertModel",
            "model_description": "albert base model"
        }
    },
    "decoder_only": {
        "gpt2": {
            "model_id": "gpt2",
            "model_class": "GPT2LMHeadModel",
            "model_description": "gpt2 base model"
        },
        "llama": {
            "model_id": "meta-llama/Llama-2-7b-hf",
            "model_class": "LlamaForCausalLM",
            "model_description": "llama 2 7b model"
        },
        "phi": {
            "model_id": "microsoft/phi-2",
            "model_class": "PhiForCausalLM",
            "model_description": "phi-2 model"
        }
    },
    "encoder_decoder": {
        "t5": {
            "model_id": "t5-small",
            "model_class": "T5ForConditionalGeneration",
            "model_description": "t5 small model"
        },
        "bart": {
            "model_id": "facebook/bart-base",
            "model_class": "BartForConditionalGeneration",
            "model_description": "bart base model"
        }
    },
    "vision": {
        "vit": {
            "model_id": "google/vit-base-patch16-224",
            "model_class": "ViTForImageClassification",
            "model_description": "vit base model"
        },
        "swin": {
            "model_id": "microsoft/swin-base-patch4-window7-224",
            "model_class": "SwinForImageClassification",
            "model_description": "swin base model"
        }
    },
    "multimodal": {
        "clip": {
            "model_id": "openai/clip-vit-base-patch32",
            "model_class": "CLIPModel",
            "model_description": "clip base model"
        },
        "blip": {
            "model_id": "Salesforce/blip-image-captioning-base",
            "model_class": "BlipForConditionalGeneration",
            "model_description": "blip base model"
        }
    },
    "audio": {
        "wav2vec2": {
            "model_id": "facebook/wav2vec2-base-960h",
            "model_class": "Wav2Vec2ForCTC",
            "model_description": "wav2vec2 base model"
        },
        "whisper": {
            "model_id": "openai/whisper-small",
            "model_class": "WhisperForConditionalGeneration",
            "model_description": "whisper small model"
        }
    }
}

def get_model_constants(model_family, architecture=None):
    """Get model-specific constants."""
    if architecture is None:
        # Try to determine architecture from model family
        architecture = ARCHITECTURE_MAPPING.get(model_family, "encoder_only")
    
    # Check if model family exists in constants
    if model_family in MODEL_CONSTANTS.get(architecture, {}):
        return MODEL_CONSTANTS[architecture][model_family]
    
    # If not found, use a default template based on architecture
    if architecture == "encoder_only":
        return {
            "model_id": f"{model_family}-base",
            "model_class": f"{model_family.capitalize()}Model",
            "model_description": f"{model_family} base model"
        }
    elif architecture == "decoder_only":
        return {
            "model_id": model_family,
            "model_class": f"{model_family.capitalize()}ForCausalLM",
            "model_description": f"{model_family} base model"
        }
    elif architecture == "encoder_decoder":
        return {
            "model_id": f"{model_family}-base",
            "model_class": f"{model_family.capitalize()}ForConditionalGeneration",
            "model_description": f"{model_family} base model"
        }
    elif architecture == "vision":
        return {
            "model_id": f"google/{model_family}-base-patch16-224",
            "model_class": f"{model_family.capitalize()}ForImageClassification",
            "model_description": f"{model_family} base model"
        }
    elif architecture == "multimodal":
        return {
            "model_id": f"{model_family}-base",
            "model_class": f"{model_family.capitalize()}Model",
            "model_description": f"{model_family} base model"
        }
    elif architecture == "audio":
        return {
            "model_id": f"{model_family}-base",
            "model_class": f"{model_family.capitalize()}ForCTC",
            "model_description": f"{model_family} base model"
        }
    else:
        # Default fallback
        return {
            "model_id": f"{model_family}-base",
            "model_class": f"{model_family.capitalize()}Model",
            "model_description": f"{model_family} base model"
        }

def pascal_case(s):
    """Convert a string to PascalCase."""
    # Split by non-alphanumeric characters
    words = ''.join(c if c.isalnum() else ' ' for c in s).split()
    # Capitalize each word and join
    return ''.join(word.capitalize() for word in words)

def generate_test_file(model_family, output_dir=".", template_model=None):
    """Generate a test file for a specific model architecture."""
    # Determine the architecture
    architecture = ARCHITECTURE_MAPPING.get(model_family, "encoder_only")
    
    # If template model is provided, use its architecture
    if template_model:
        template_architecture = ARCHITECTURE_MAPPING.get(template_model, architecture)
        # Use the template's architecture if it exists
        if template_architecture != architecture:
            logger.info(f"Using {template_architecture} architecture from template model {template_model}")
            architecture = template_architecture
    
    # Get model-specific constants
    model_constants = get_model_constants(model_family, architecture)
    model_id = model_constants["model_id"]
    model_class = model_constants["model_class"]
    model_description = model_constants["model_description"]
    
    # Format the model name for class name
    model_pascal = pascal_case(model_family)
    model_upper = model_family.upper()
    
    # Format the registry template
    registry = MODEL_REGISTRY_TEMPLATES[architecture].format(
        model_upper=model_upper,
        model_id=model_id,
        model_description=model_description,
        model_class=model_class
    )
    
    # Format the test class template
    test_class = TEST_CLASS_TEMPLATES[architecture].format(
        model_pascal=model_pascal,
        model_family=model_family,
        model_id=model_id,
        model_upper=model_upper
    )
    
    # Format the pipeline test method template
    pipeline_test = PIPELINE_TEST_TEMPLATES[architecture]
    
    # Format the run tests method template
    run_tests = RUN_TESTS_TEMPLATE
    
    # Format the main method template
    main = MAIN_TEMPLATE.format(
        model_family=model_family,
        model_id=model_id,
        model_pascal=model_pascal
    )
    
    # Assemble the file content
    content = f"{HEADER_TEMPLATE}\n\n{registry}\n\n{test_class}\n    \n    {pipeline_test}\n    \n    {run_tests}\n\n{main}\n"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the file
    output_path = os.path.join(output_dir, f"test_hf_{model_family}.py")
    with open(output_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test files for HuggingFace models")
    parser.add_argument("--family", type=str, required=True,
                        help="Model family (e.g., bert, gpt2, t5)")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for generated files")
    parser.add_argument("--template", type=str,
                        help="Template model to use for architecture detection")
    args = parser.parse_args()
    
    # Generate the test file
    output_path = generate_test_file(args.family, args.output, args.template)
    
    # Verify the generated file
    try:
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Try to compile to check for syntax errors
        compile(content, output_path, 'exec')
        logger.info(f"Syntax verified for {output_path}")
    except Exception as e:
        logger.error(f"Syntax error in generated file {output_path}: {e}")
        sys.exit(1)
    
    logger.info("Test file generation completed successfully")

if __name__ == "__main__":
    main()