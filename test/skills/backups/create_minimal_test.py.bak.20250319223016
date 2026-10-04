#!/usr/bin/env python3
"""
Script to create minimal Hugging Face test files with correct indentation.
This script creates clean, minimal test files for each model family.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"create_minimal_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Common model families
MODEL_FAMILIES = {
    "bert": {
        "model_id": "bert-base-uncased",
        "model_class": "BertModel",
        "tokenizer_class": "BertTokenizer",
        "task": "fill-mask",
        "test_text": "The man worked as a [MASK].",
        "architecture_type": "encoder_only"
    },
    "gpt2": {
        "model_id": "gpt2",
        "model_class": "GPT2LMHeadModel",
        "tokenizer_class": "GPT2Tokenizer",
        "task": "text-generation",
        "test_text": "Once upon a time",
        "architecture_type": "decoder_only"
    },
    "t5": {
        "model_id": "t5-small",
        "model_class": "T5ForConditionalGeneration",
        "tokenizer_class": "T5Tokenizer",
        "task": "translation_en_to_fr",
        "test_text": "translate English to French: Hello, how are you?",
        "architecture_type": "encoder_decoder"
    },
    "vit": {
        "model_id": "google/vit-base-patch16-224",
        "model_class": "ViTForImageClassification",
        "processor_class": "ViTImageProcessor",
        "task": "image-classification",
        "test_image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "architecture_type": "vision"
    }
}

def verify_python_syntax(file_path):
    """
    Verify the Python syntax of a file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code to check syntax
        compile(content, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def generate_minimal_imports():
    """Generate minimal import statements."""
    return """#!/usr/bin/env python3

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
"""

def generate_hardware_detection():
    """Generate hardware detection function."""
    return """
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
HW_CAPABILITIES = check_hardware()
"""

def generate_minimal_class(family, family_info):
    """Generate a minimal test class with correct indentation."""
    
    class_name = f"Test{family.capitalize()}Models"
    registry_name = f"{family.upper()}_MODELS_REGISTRY"
    model_id = family_info.get("model_id", "")
    model_class = family_info.get("model_class", "")
    task = family_info.get("task", "")
    architecture_type = family_info.get("architecture_type", "")
    
    # Create a minimal model registry
    registry = f"""
# Models registry
{registry_name} = {{
    "{model_id}": {{
        "description": "{family} base model",
        "class": "{model_class}",
    }}
}}
"""
    
    # Create the class with proper indentation
    class_def = f"""
class {class_name}:
    \"\"\"Test class for {family} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Use registry information
        if self.model_id not in {registry_name}:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {registry_name}["{model_id}"]
        else:
            self.model_info = {registry_name}[self.model_id]
        
        # Define model parameters
        self.task = "{task}"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
"""
    
    # Add appropriate test inputs based on architecture type
    if architecture_type == "vision":
        class_def += f'        self.test_image_url = "{family_info.get("test_image_url", "http://images.cocodataset.org/val2017/000000039769.jpg")}"\n'
    else:
        class_def += f'        self.test_text = "{family_info.get("test_text", "Test input")}"\n'
    
    # Add hardware preference
    class_def += """
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    def test_pipeline(self, device="auto"):
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
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
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
"""
    
    # Add pipeline input based on architecture type
    if architecture_type == "vision":
        class_def += """            # For vision models
            import requests
            from PIL import Image
            from io import BytesIO
            
            response = requests.get(self.test_image_url)
            pipeline_input = Image.open(BytesIO(response.content))
"""
    else:
        class_def += """            # For text models
            pipeline_input = self.test_text
"""
    
    # Complete the test_pipeline method
    class_def += """
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
            logger.error(f"Error testing pipeline: {e}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results
    
    def run_tests(self, all_hardware=False):
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
        }
"""
    
    return registry + class_def

def generate_main_function(family, family_info):
    """Generate main function for the test script."""
    model_id = family_info.get("model_id", "")
    class_name = f"Test{family.capitalize()}Models"
    
    return f"""
def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {family} models")
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
    tester = {class_name}(model_id)
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
    main()
"""

def create_minimal_test_file(family, output_dir=None):
    """Create a minimal test file for a specific model family."""
    if family not in MODEL_FAMILIES:
        logger.error(f"Unknown model family: {family}")
        return None
    
    family_info = MODEL_FAMILIES[family]
    
    # Create the minimal test file
    content = ""
    content += generate_minimal_imports()
    content += generate_hardware_detection()
    content += generate_minimal_class(family, family_info)
    content += generate_main_function(family, family_info)
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_hf_{family}.py")
    else:
        output_path = f"test_hf_{family}.py"
    
    # Write the file
    with open(output_path, "w") as f:
        f.write(content)
    
    # Verify syntax
    is_valid, error = verify_python_syntax(output_path)
    if is_valid:
        logger.info(f"✅ Created minimal test file for {family}: {output_path}")
        return output_path
    else:
        logger.error(f"❌ Syntax error in {output_path}: {error}")
        return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create minimal HuggingFace test files")
    parser.add_argument("--families", nargs="+", choices=MODEL_FAMILIES.keys(), default=list(MODEL_FAMILIES.keys()),
                        help="Model families to create test files for")
    parser.add_argument("--output-dir", type=str, default="fixed_tests",
                        help="Output directory for test files")
    
    args = parser.parse_args()
    
    # Create each test file
    success_count = 0
    failure_count = 0
    
    for family in args.families:
        output_path = create_minimal_test_file(family, args.output_dir)
        if output_path:
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- Files created: {success_count}")
    logger.info(f"- Failed: {failure_count}")
    logger.info(f"- Output directory: {args.output_dir}")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())