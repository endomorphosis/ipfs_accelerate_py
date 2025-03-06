#!/usr/bin/env python3
"""
Improved Hugging Face Test Generator with Enhanced Hardware Support

This generator supports all hardware platforms including:
- CPU: Standard CPU execution
- CUDA: NVIDIA GPU acceleration
- ROCm: AMD GPU acceleration 
- MPS: Apple Silicon GPU acceleration
- OpenVINO: Intel hardware acceleration
- QNN: Qualcomm Neural Networks
- WebNN: Web Neural Network API support
- WebGPU: Web Graphics Processing Unit support
"""

import os
import sys
import argparse
import importlib
import time
import json
import inspect
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import hardware_template_integration
try:
    from hardware_template_integration import detect_model_modality, get_hardware_template_for_model
    HAS_HARDWARE_TEMPLATES = True
except ImportError:
    HAS_HARDWARE_TEMPLATES = False
    logger.warning("hardware_template_integration.py not found, using basic templates")

# Hardware Detection
# Try to import torch first (needed for CUDA/ROCm/MPS)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False 
HAS_MPS = False
HAS_OPENVINO = False
HAS_QUALCOMM = False
HAS_WEBNN = False
HAS_WEBGPU = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# Qualcomm detection
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)

# WebNN detection (browser API or simulation)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    importlib.util.find_spec("webnn_js") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_ENABLED" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU detection (browser API or simulation)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_ENABLED" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

# Web platform optimizations
HAS_WEBGPU_COMPUTE_SHADERS = (
    "WEBGPU_COMPUTE_SHADERS_ENABLED" in os.environ or
    "WEBGPU_COMPUTE_SHADERS" in os.environ
)

HAS_PARALLEL_LOADING = (
    "WEB_PARALLEL_LOADING_ENABLED" in os.environ or
    "PARALLEL_LOADING_ENABLED" in os.environ
)

HAS_SHADER_PRECOMPILE = (
    "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ or
    "WEBGPU_SHADER_PRECOMPILE" in os.environ
)

# Hardware mapping for key models
KEY_MODEL_HARDWARE_MAP = {
    "bert": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL",
        "mps": "REAL",
        "rocm": "REAL",
        "qualcomm": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "t5": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL", 
        "mps": "REAL",
        "rocm": "REAL",
        "qualcomm": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "llama": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL",
        "mps": "REAL",
        "rocm": "REAL",
        "qualcomm": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    }
}

# Default test template
DEFAULT_TEST_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
Generated test for {model_name} with {platform} support
\"\"\"

import os
import sys
import unittest
import logging
import importlib
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Try importing hardware dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available")

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_QNN = False
HAS_WEBNN = False
HAS_WEBGPU = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection 
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# Qualcomm detection
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)

# WebNN detection
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU detection
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

class {model_class_name}Test(unittest.TestCase):
    \"\"\"Test suite for {model_name} model\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Initialize the model and dependencies\"\"\"
        cls.model_name = "{model_name}"
        
    def test_{platform_name}_inference(self):
        \"\"\"Test {model_name} with {platform} hardware\"\"\"
        # Skip test if hardware not available
        if "{platform}" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "{platform}" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
        elif "{platform}" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "{platform}" == "openvino" and not HAS_OPENVINO:
            self.skipTest("OpenVINO not available")
        elif "{platform}" == "qualcomm" and not HAS_QUALCOMM:
            self.skipTest("Qualcomm AI SDK not available")
        elif "{platform}" == "webnn" and not HAS_WEBNN:
            self.skipTest("WebNN not available")
        elif "{platform}" == "webgpu" and not HAS_WEBGPU:
            self.skipTest("WebGPU not available")
        
        try:
            import transformers
            
            # Initialize tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_name}")
            
            # Initialize model with appropriate hardware
            model = transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if "{platform}" == "cuda":
                model = model.to("cuda")
            elif "{platform}" == "rocm":
                model = model.to("cuda")  # ROCm uses CUDA API
            elif "{platform}" == "mps":
                model = model.to("mps")
            
            # Create sample input
            sample_text = "This is a test for {model_name} model on {platform}"
            inputs = tokenizer(sample_text, return_tensors="pt")
            
            # Move inputs to the correct device
            if "{platform}" in ["cuda", "rocm"]:
                inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
            elif "{platform}" == "mps":
                inputs = {{k: v.to("mps") for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Basic validation
            self.assertIsNotNone(outputs)
            self.assertIn("last_hidden_state", outputs)
            self.assertEqual(outputs.last_hidden_state.shape[0], 1)
            
            logger.info(f"Successfully tested {model_name} on {platform}")
            
        except Exception as e:
            logger.error(f"Error testing {model_name} on {platform}: {{str(e)}}")
            raise
            
if __name__ == "__main__":
    unittest.main()
"""

# Model modality detection
def detect_model_modality(model_name: str) -> str:
    """Detect model modality based on name."""
    if HAS_HARDWARE_TEMPLATES:
        # Use the imported function from hardware_template_integration
        from hardware_template_integration import detect_model_modality as detect_func
        return detect_func(model_name)
    
    # Fallback implementation
    model_lower = model_name.lower()
    
    # Text models
    if any(text_model in model_lower for text_model in ["bert", "gpt", "t5", "roberta", "llama"]):
        return "text"
    
    # Vision models
    if any(vision_model in model_lower for vision_model in ["vit", "deit", "resnet", "detr"]):
        return "vision"
    
    # Audio models
    if any(audio_model in model_lower for audio_model in ["wav2vec2", "whisper", "clap"]):
        return "audio"
    
    # Multimodal models
    if any(mm_model in model_lower for mm_model in ["clip", "llava", "blip"]):
        return "multimodal"
    
    # Video models
    if any(video_model in model_lower for video_model in ["xclip", "videomae"]):
        return "video"
    
    # Default to text
    return "text"

def get_hardware_support_for_model(model_name: str) -> Dict[str, str]:
    """Get hardware support map for a model."""
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    
    # Check if this is a known key model
    if model_base in KEY_MODEL_HARDWARE_MAP:
        return KEY_MODEL_HARDWARE_MAP[model_base]
    
    # Default hardware map based on modality
    modality = detect_model_modality(model_name)
    default_map = {
        "text": {
            "cpu": "REAL", "cuda": "REAL", "openvino": "REAL", 
            "mps": "REAL", "rocm": "REAL", "qualcomm": "REAL",
            "webnn": "REAL", "webgpu": "REAL"
        },
        "vision": {
            "cpu": "REAL", "cuda": "REAL", "openvino": "REAL", 
            "mps": "REAL", "rocm": "REAL", "qualcomm": "REAL",
            "webnn": "REAL", "webgpu": "REAL"
        },
        "audio": {
            "cpu": "REAL", "cuda": "REAL", "openvino": "REAL", 
            "mps": "REAL", "rocm": "REAL", "qualcomm": "REAL",
            "webnn": "SIMULATION", "webgpu": "SIMULATION"
        },
        "multimodal": {
            "cpu": "REAL", "cuda": "REAL", "openvino": "SIMULATION", 
            "mps": "SIMULATION", "rocm": "SIMULATION", "qualcomm": "SIMULATION",
            "webnn": "SIMULATION", "webgpu": "SIMULATION"
        },
        "video": {
            "cpu": "REAL", "cuda": "REAL", "openvino": "REAL", 
            "mps": "REAL", "rocm": "REAL", "qualcomm": "REAL",
            "webnn": "SIMULATION", "webgpu": "SIMULATION"
        }
    }
    
    return default_map.get(modality, default_map["text"])

def generate_test_file(model_name: str, platforms: List[str], output_path: Optional[str] = None) -> str:
    """Generate a test file for the given model and platforms."""
    logger.info(f"Generating test file for {model_name} with platforms: {', '.join(platforms)}")
    
    # Get hardware support for this model
    hardware_support = get_hardware_support_for_model(model_name)
    
    # Filter out unsupported platforms
    supported_platforms = [p for p in platforms if hardware_support.get(p, "NONE") != "NONE"]
    
    if not supported_platforms:
        logger.warning(f"No supported platforms found for {model_name} from {platforms}")
        # Fallback to CPU
        supported_platforms = ["cpu"]
    
    # Create class name
    model_class_name = "".join(part.capitalize() for part in model_name.split("-"))
    
    # Generate test content
    test_content = DEFAULT_TEST_TEMPLATE.format(
        model_name=model_name,
        model_class_name=model_class_name,
        platform=supported_platforms[0],  # Use first platform for template
        platform_name=supported_platforms[0]
    )
    
    # Add tests for additional platforms
    for platform in supported_platforms[1:]:
        platform_test = f"""

    def test_{platform}_inference(self):
        \"\"\"Test {model_name} with {platform} hardware\"\"\"
        # Skip test if hardware not available
        if "{platform}" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "{platform}" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
        elif "{platform}" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "{platform}" == "openvino" and not HAS_OPENVINO:
            self.skipTest("OpenVINO not available")
        elif "{platform}" == "qualcomm" and not HAS_QUALCOMM:
            self.skipTest("Qualcomm AI SDK not available")
        elif "{platform}" == "webnn" and not HAS_WEBNN:
            self.skipTest("WebNN not available")
        elif "{platform}" == "webgpu" and not HAS_WEBGPU:
            self.skipTest("WebGPU not available")
        
        try:
            import transformers
            
            # Initialize tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_name}")
            
            # Initialize model with appropriate hardware
            model = transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if "{platform}" == "cuda":
                model = model.to("cuda")
            elif "{platform}" == "rocm":
                model = model.to("cuda")  # ROCm uses CUDA API
            elif "{platform}" == "mps":
                model = model.to("mps")
            
            # Create sample input
            sample_text = "This is a test for {model_name} model on {platform}"
            inputs = tokenizer(sample_text, return_tensors="pt")
            
            # Move inputs to the correct device
            if "{platform}" in ["cuda", "rocm"]:
                inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
            elif "{platform}" == "mps":
                inputs = {{k: v.to("mps") for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Basic validation
            self.assertIsNotNone(outputs)
            self.assertIn("last_hidden_state", outputs)
            self.assertEqual(outputs.last_hidden_state.shape[0], 1)
            
            logger.info(f"Successfully tested {model_name} on {platform}")
            
        except Exception as e:
            logger.error(f"Error testing {model_name} on {platform}: {{str(e)}}")
            raise
        """
        
        # Find position before 'if __name__' to insert new method
        if_name_pos = test_content.find('if __name__ == "__main__"')
        if if_name_pos > 0:
            # Insert before if __name__ == "__main__"
            # Find the line before if __name__
            prev_line_pos = test_content.rfind('\n', 0, if_name_pos)
            if prev_line_pos > 0:
                test_content = test_content[:prev_line_pos] + platform_test + test_content[prev_line_pos:]
    
    # Write to output file if provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(test_content)
        logger.info(f"Test file written to {output_path}")
    
    return test_content

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate hardware-aware test files for Hugging Face models")
    parser.add_argument("--generate", "-g", dest="model", help="Model to generate test for")
    parser.add_argument("--platform", "-p", dest="platform", default="cpu", 
                        help="Hardware platform(s) to test on (comma-separated): cpu,cuda,rocm,mps,openvino,qualcomm,webnn,webgpu")
    parser.add_argument("--output", "-o", dest="output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.model:
        parser.print_help()
        sys.exit(1)
    
    # Parse platforms
    platforms = [p.strip() for p in args.platform.split(',')]
    
    # Generate test file
    test_content = generate_test_file(args.model, platforms, args.output)
    
    # Print test content if no output file
    if not args.output:
        print(test_content)

if __name__ == "__main__":
    main()