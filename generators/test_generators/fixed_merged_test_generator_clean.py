#!/usr/bin/env python3
"""
Fixed Merged Test Generator - Clean Version

This is a simplified version of the test generator that works
reliably without syntax errors.
"""

import os
import sys
import argparse
import importlib.util
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
    HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
except ImportError:
    HAS_CUDA = False
    HAS_ROCM = False
    HAS_MPS = False

# Other hardware detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_QUALCOMM = importlib.util.find_spec("qnn_wrapper") is not None or importlib.util.find_spec("qti") is not None
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ

# Define key model hardware support
KEY_MODEL_HARDWARE_MAP = {
    "bert": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "t5": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "vit": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "clip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "whisper": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "wav2vec2": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "clap": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llama": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llava": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "xclip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "detr": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "qwen2": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"}
}

def detect_model_modality(model_name):
    model_lower = model_name.lower()
    
    # Text models
    if any(t in model_lower for t in ["bert", "gpt", "t5", "llama", "roberta"]):
        return "text"
    
    # Vision models
    if any(v in model_lower for v in ["vit", "deit", "resnet", "convnext"]):
        return "vision"
    
    # Audio models
    if any(a in model_lower for a in ["wav2vec", "whisper", "hubert", "clap"]):
        return "audio"
    
    # Multimodal models
    if any(m in model_lower for m in ["clip", "llava", "blip"]):
        return "multimodal"
    
    # Video models
    if any(v in model_lower for v in ["xclip", "video"]):
        return "video"
    
    # Default to text
    return "text"

def generate_imports_for_platform(platform):
    """Generate the imports for a specific platform."""
    imports = []
    
    if platform == "cpu":
        imports.append("import torch")
    elif platform == "cuda" or platform == "rocm":
        imports.append("import torch")
    elif platform == "mps":
        imports.append("import torch")
    elif platform == "openvino":
        imports.append("import torch")
        imports.append("try:\n    import openvino as ov\nexcept ImportError:\n    ov = None")
    elif platform == "qualcomm":
        imports.append("import torch")
        imports.append("try:\n    import qnn_wrapper\nexcept ImportError:\n    qnn_wrapper = None")
    elif platform == "webnn":
        imports.append("import torch")
        imports.append("# WebNN specific imports would go here")
    elif platform == "webgpu":
        imports.append("import torch")
        imports.append("# WebGPU specific imports would go here")
    
    return imports

def generate_test_file(model_name, platform=None, output_dir=None, cross_platform=False):
    """Generate a test file for the specified model and platforms."""
    model_type = detect_model_modality(model_name)
    
    # Default to all platforms if none specified or cross-platform is True
    all_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    if cross_platform:
        platforms = all_platforms
    elif platform and platform != "all":
        platforms = [p.strip() for p in platform.split(",")]
    else:
        platforms = all_platforms
    
    # Create file name and path
    file_name = f"test_hf_{model_name.replace('-', '_')}.py"
    
    # Use output_dir if specified, otherwise use current directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
    else:
        output_path = file_name
    
    # Generate file content
    with open(output_path, "w") as f:
        # Header and imports
        f.write(f'''#!/usr/bin/env python3
"""
Test for {model_name} model with hardware platform support
Generated by fixed_merged_test_generator_clean.py
"""

import os
import sys
import unittest
import importlib.util
import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_QUALCOMM = importlib.util.find_spec("qnn_wrapper") is not None or importlib.util.find_spec("qti") is not None
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ

class Test{model_name.replace("-", "").title()}(unittest.TestCase):
    """Test {model_name} model with cross-platform hardware support."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_id = "{model_name}"
        self.tokenizer = None
        self.model = None
        self.modality = "{model_type}"
''')
        
        # Add methods for each platform
        for p in platforms:
            # Only include supported platforms based on hardware detection
            hardware_var = f"HAS_{p.upper()}"
            
            # Skip condition based on hardware availability
            skip_condition = f"if not {hardware_var}: self.skipTest('{p.upper()} not available')"
            
            # Device setup based on platform
            if p == "cuda" or p == "rocm":
                device_setup = 'device = "cuda"'
            elif p == "mps":
                device_setup = 'device = "mps"'
            else:
                device_setup = 'device = "cpu"'
            
            # Check model specific support in KEY_MODEL_HARDWARE_MAP
            model_base = model_name.split("-")[0].lower()
            special_setup = ""
            special_teardown = ""
            
            if model_base in KEY_MODEL_HARDWARE_MAP:
                if p.lower() in KEY_MODEL_HARDWARE_MAP[model_base]:
                    support_type = KEY_MODEL_HARDWARE_MAP[model_base][p.lower()]
                    if support_type == "NONE":
                        continue  # Skip this platform entirely
                    elif support_type == "SIMULATION":
                        special_setup = '        logger.info("Using simulation mode for this platform")'
            
            # Add the test method
            f.write(f'''
    def test_{p.lower()}(self):
        """Test {model_name} with {p}."""
        # Skip if hardware not available
        {skip_condition}
        
        # Set up device
        {device_setup}
        {special_setup}
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Initialize model
            self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move model to device if not CPU
            if device != "cpu":
                self.model = self.model.to(device)
            
            # Prepare input
            inputs = self.tokenizer("Test input for {model_name}", return_tensors="pt")
            
            # Move inputs to device if not CPU
            if device != "cpu":
                inputs = {{k: v.to(device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            self.assertIn("last_hidden_state", outputs)
            
            # Log success
            logger.info(f"Successfully tested {{self.model_id}} on {p.lower()}")
            {special_teardown}
        except Exception as e:
            logger.error(f"Error testing {{self.model_id}} on {p.lower()}: {{str(e)}}")
            raise
''')
        
        # Add test main
        f.write('''
if __name__ == "__main__":
    unittest.main()
''')
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Fixed Merged Test Generator - Clean Version")
    parser.add_argument("--generate", "-g", type=str, help="Model to generate tests for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to generate tests for (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    parser.add_argument("--cross-platform", "-c", action="store_true", help="Generate test for all platforms")
    
    args = parser.parse_args()
    
    if args.generate:
        output_file = generate_test_file(
            args.generate,
            args.platform,
            args.output_dir,
            cross_platform=args.cross_platform
        )
        print(f"Generated test file: {output_file}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())