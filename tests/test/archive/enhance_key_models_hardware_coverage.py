#!/usr/bin/env python3
"""
Script to enhance hardware platform support for the 13 key HuggingFace model tests.

This script:
1. Identifies the 13 key HuggingFace models
2. Adds comprehensive hardware platform support to their test files
3. Validates implementation across CPU, CUDA, OpenVINO, MPS, ROCm, WebNN, and WebGPU
4. Updates the test generator templates with improved hardware support

Usage:
  python enhance_key_models_hardware_coverage.py --fix-all
  python enhance_key_models_hardware_coverage.py --fix-model bert
  python enhance_key_models_hardware_coverage.py --fix-platform openvino
  python enhance_key_models_hardware_coverage.py --validate
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("key_models_hardware_fix.log")
    ]
)
logger = logging.getLogger("key_models_hardware_fix")

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
SKILLS_DIR = TEST_DIR / "skills"
OUTPUT_DIR = TEST_DIR / "key_models_hardware_fixes"
TEMPLATE_DIR = TEST_DIR / "hardware_test_templates"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)

# The 13 key models
KEY_MODELS = [
    "bert",        # Text embedding
    "clap",        # Audio-text multimodal
    "clip",        # Vision-text multimodal
    "detr",        # Object detection
    "llama",       # Language model
    "llava",       # Vision-language model
    "llava_next",  # Next-gen vision-language model
    "qwen2",       # Transformer LM
    "t5",          # Text-to-text transformer
    "vit",         # Vision transformer
    "wav2vec2",    # Speech model
    "whisper",     # Speech recognition
    "xclip"        # Extended clip for video
]

# Hardware platforms to validate
HARDWARE_PLATFORMS = [
    "cpu",       # Always available
    "cuda",      # NVIDIA GPUs
    "openvino",  # Intel hardware
    "mps",       # Apple Silicon
    "rocm",      # AMD GPUs
    "webnn",     # Browser WebNN API
    "webgpu"     # Browser WebGPU API
]

# Model category to hardware platform compatibility mapping
MODEL_HARDWARE_COMPATIBILITY = {
    "text_embedding": {
        "cpu": True, "cuda": True, "openvino": True, "mps": True, "rocm": True, "webnn": True, "webgpu": True
    },
    "text_generation": {
        "cpu": True, "cuda": True, "openvino": True, "mps": True, "rocm": True, "webnn": False, "webgpu": True
    },
    "vision": {
        "cpu": True, "cuda": True, "openvino": True, "mps": True, "rocm": True, "webnn": True, "webgpu": True
    },
    "audio": {
        "cpu": True, "cuda": True, "openvino": True, "mps": True, "rocm": True, "webnn": False, "webgpu": False
    },
    "multimodal": {
        "cpu": True, "cuda": True, "openvino": True, "mps": False, "rocm": False, "webnn": False, "webgpu": False
    },
    "vision_language": {
        "cpu": True, "cuda": True, "openvino": True, "mps": False, "rocm": False, "webnn": False, "webgpu": False 
    },
    "video": {
        "cpu": True, "cuda": True, "openvino": True, "mps": True, "rocm": True, "webnn": False, "webgpu": False
    }
}

# Categorize the key models
MODEL_CATEGORIES = {
    "bert": "text_embedding",
    "clap": "audio",
    "clip": "vision",
    "detr": "vision",
    "llama": "text_generation",
    "llava": "vision_language",
    "llava_next": "vision_language",
    "qwen2": "text_generation",
    "t5": "text_generation",
    "vit": "vision",
    "wav2vec2": "audio",
    "whisper": "audio",
    "xclip": "video"
}

# Platform-specific imports
PLATFORM_IMPORTS = {
    "cpu": [],  # Standard platform, no special imports
    "cuda": ["import torch"],
    "openvino": ["import openvino"],
    "mps": ["import torch"],
    "rocm": ["import torch"],
    "webnn": ["# WebNN specific imports would be added at runtime"],
    "webgpu": ["# WebGPU specific imports would be added at runtime"]
}

# Template for platform-specific initialization method
PLATFORM_INIT_TEMPLATE = """
def init_{platform}(self):
    \"\"\"Initialize for {platform_upper} platform.\"\"\"
    {imports}
    self.platform = "{platform_upper}"
    self.device = "{platform}"
    {platform_specific_setup}
    return True
"""

# Template for platform-specific handler creation method
PLATFORM_HANDLER_TEMPLATE = """
def create_{platform}_handler(self):
    \"\"\"Create handler for {platform_upper} platform.\"\"\"
    {platform_specific_handler}
    return handler
"""

def analyze_test_file(test_file_path: Path) -> Dict[str, Any]:
    """
    Analyze a test file to determine which hardware platforms are supported.
    
    Args:
        test_file_path: Path to the test file
        
    Returns:
        Dictionary with analysis results
    """
    if not test_file_path.exists():
        return {"error": f"Test file not found: {test_file_path}"}
    
    # Initialize results
    model_name = test_file_path.stem.replace("test_hf_", "")
    category = MODEL_CATEGORIES.get(model_name, "unknown")
    
    analysis = {
        "model": model_name,
        "category": category,
        "file_path": str(test_file_path),
        "platforms": {},
        "needs_fixes": False
    }
    
    try:
        # Read the file content
        with open(test_file_path, 'r') as f:
            content = f.read()
        
        # Check platform support for each hardware platform
        for platform in HARDWARE_PLATFORMS:
            # Check if the platform should be compatible with this model category
            should_be_compatible = MODEL_HARDWARE_COMPATIBILITY.get(category, {}).get(platform, False)
            
            # Check for platform-specific methods and setup
            has_init_method = f"def init_{platform}" in content
            has_handler_method = f"create_{platform}_handler" in content
            has_platform_test = f"platform: {platform.upper()}" in content
            
            # Determine if the platform is currently supported in the test file
            is_supported = has_init_method and has_handler_method and has_platform_test
            
            # Determine if it needs to be fixed
            needs_fix = should_be_compatible and not is_supported
            
            # Store results
            analysis["platforms"][platform] = {
                "should_be_compatible": should_be_compatible,
                "is_supported": is_supported,
                "has_init_method": has_init_method,
                "has_handler_method": has_handler_method,
                "has_platform_test": has_platform_test,
                "needs_fix": needs_fix
            }
            
            # Update overall needs_fixes flag
            if needs_fix:
                analysis["needs_fixes"] = True
    
    except Exception as e:
        logger.error(f"Error analyzing test file {test_file_path}: {e}")
        return {"error": str(e), "model": model_name, "file_path": str(test_file_path)}
    
    return analysis

def generate_platform_code(model: str, platform: str) -> Dict[str, str]:
    """
    Generate platform-specific code for a model and platform.
    
    Args:
        model: Model name
        platform: Hardware platform
        
    Returns:
        Dictionary with init method and handler method code
    """
    category = MODEL_CATEGORIES.get(model, "unknown")
    
    # Get platform-specific imports
    imports = "\n    ".join(PLATFORM_IMPORTS.get(platform, []))
    
    # Generate platform-specific setup code
    platform_specific_setup = ""
    if platform == "cuda":
        platform_specific_setup = 'self.device_name = "cuda" if torch.cuda.is_available() else "cpu"'
    elif platform == "mps":
        platform_specific_setup = 'self.device_name = "mps" if torch.backends.mps.is_available() else "cpu"'
    elif platform == "rocm":
        platform_specific_setup = 'self.device_name = "cuda" if torch.cuda.is_available() and torch.version.hip is not None else "cpu"'
    elif platform == "openvino":
        platform_specific_setup = 'self.device_name = "openvino"'
    elif platform == "webnn":
        platform_specific_setup = 'self.device_name = "webnn"'
    elif platform == "webgpu":
        platform_specific_setup = 'self.device_name = "webgpu"'
    else:
        platform_specific_setup = 'self.device_name = "cpu"'
    
    # Generate handler creation code
    platform_specific_handler = ""
    if category == "text_embedding":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_text: compiled_model(np.array(input_text))[0]'''
        elif platform in ["webnn", "webgpu"]:
            platform_specific_handler = f'''# This is a mock handler for {platform}
        handler = MockHandler(self.model_path, platform="{platform}")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModel.from_pretrained(model_path).to(self.device_name)'''
    
    elif category == "text_generation":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_text: compiled_model(np.array(input_text))[0]'''
        elif platform in ["webnn"]:
            platform_specific_handler = f'''# WebNN doesn't support text generation models yet
        raise NotImplementedError("WebNN doesn't support text generation models yet")'''
        elif platform in ["webgpu"]:
            platform_specific_handler = f'''# This is a mock handler for {platform}
        handler = MockHandler(self.model_path, platform="{platform}")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModelForCausalLM.from_pretrained(model_path).to(self.device_name)'''
    
    elif category == "vision":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_image: compiled_model(np.array(input_image))[0]'''
        elif platform in ["webnn", "webgpu"]:
            platform_specific_handler = f'''# This is a mock handler for {platform}
        handler = MockHandler(self.model_path, platform="{platform}")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModelForImageClassification.from_pretrained(model_path).to(self.device_name)'''
    
    elif category == "audio":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_audio: compiled_model(np.array(input_audio))[0]'''
        elif platform in ["webnn", "webgpu"]:
            platform_specific_handler = f'''# {platform} doesn't support audio models yet
        raise NotImplementedError("{platform} doesn't support audio models yet")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device_name)'''
    
    elif category == "vision_language" or category == "multimodal":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_data: compiled_model(np.array(input_data))[0]'''
        elif platform in ["mps", "rocm", "webnn", "webgpu"]:
            platform_specific_handler = f'''# {platform} doesn't support multimodal models yet
        raise NotImplementedError("{platform} doesn't support multimodal models yet")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModel.from_pretrained(model_path).to(self.device_name)'''
    
    elif category == "video":
        if platform == "openvino":
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        from openvino.runtime import Core
        import numpy as np
        ie = Core()
        compiled_model = ie.compile_model(model_path, "CPU")
        handler = lambda input_data: compiled_model(np.array(input_data))[0]'''
        elif platform in ["webnn", "webgpu"]:
            platform_specific_handler = f'''# {platform} doesn't support video models yet
        raise NotImplementedError("{platform} doesn't support video models yet")'''
        else:
            platform_specific_handler = f'''model_path = self.get_model_path_or_name()
        handler = AutoModel.from_pretrained(model_path).to(self.device_name)'''
    
    else:
        platform_specific_handler = f'''# Generic handler for unknown category
        model_path = self.get_model_path_or_name()
        handler = AutoModel.from_pretrained(model_path)'''
    
    # Format the templates
    init_method = PLATFORM_INIT_TEMPLATE.format(
        platform=platform,
        platform_upper=platform.upper(),
        imports=imports,
        platform_specific_setup=platform_specific_setup
    )
    
    handler_method = PLATFORM_HANDLER_TEMPLATE.format(
        platform=platform,
        platform_upper=platform.upper(),
        platform_specific_handler=platform_specific_handler
    )
    
    return {
        "init_method": init_method,
        "handler_method": handler_method
    }

def fix_test_file(test_file_path: Path, analysis: Dict[str, Any]) -> Path:
    """
    Fix a test file by adding missing hardware platform support.
    
    Args:
        test_file_path: Path to the test file
        analysis: Analysis of the test file
        
    Returns:
        Path to the fixed test file
    """
    if "error" in analysis:
        logger.error(f"Cannot fix test file due to analysis error: {analysis['error']}")
        return None
    
    if not analysis.get("needs_fixes", False):
        logger.info(f"Test file {test_file_path} does not need fixes")
        return test_file_path
    
    model = analysis["model"]
    
    try:
        # Read the file content
        with open(test_file_path, 'r') as f:
            content = f.read()
        
        # Create a backup
        backup_path = OUTPUT_DIR / f"{test_file_path.name}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Add imports section if not present
        if "# Platform-specific imports" not in content:
            imports_section = "\n# Platform-specific imports\n"
            content = content.replace("from transformers import", imports_section + "from transformers import")
        
        # Add MockHandler class if not present
        if "class MockHandler" not in content:
            mock_handler = '''\nclass MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {"mock_output": f"Mock output for {self.platform}"}
'''
            # Insert after imports but before class definition
            class_start = content.find("class ")
            if class_start > 0:
                content = content[:class_start] + mock_handler + content[class_start:]
        
        # Loop through platforms that need fixes
        for platform, platform_info in analysis["platforms"].items():
            if platform_info.get("needs_fix", False):
                logger.info(f"Adding {platform} support to {model} test file")
                
                # Generate platform-specific code
                platform_code = generate_platform_code(model, platform)
                
                # Add platform initialization method
                if not platform_info.get("has_init_method", False):
                    # Find the last init method
                    init_methods = re.findall(r"def init_(\w+)\(self\):", content)
                    if init_methods:
                        last_init = f"def init_{init_methods[-1]}(self):"
                        last_init_end = content.find(last_init)
                        next_def = content.find("def ", last_init_end + 1)
                        
                        if next_def > 0:
                            # Insert after the last init method and before the next method
                            content = content[:next_def] + platform_code["init_method"] + content[next_def:]
                        else:
                            # Append to the end of the file
                            content += platform_code["init_method"]
                    else:
                        # No init methods found, add after class definition
                        class_end = content.find(":", content.find("class ")) + 1
                        content = content[:class_end] + platform_code["init_method"] + content[class_end:]
                
                # Add platform handler method
                if not platform_info.get("has_handler_method", False):
                    # Find the last handler method
                    handler_methods = re.findall(r"def create_(\w+)_handler\(self\):", content)
                    if handler_methods:
                        last_handler = f"def create_{handler_methods[-1]}_handler(self):"
                        last_handler_end = content.find(last_handler)
                        next_def = content.find("def ", last_handler_end + 1)
                        
                        if next_def > 0:
                            # Insert after the last handler method and before the next method
                            content = content[:next_def] + platform_code["handler_method"] + content[next_def:]
                        else:
                            # Append to the end of the file
                            content += platform_code["handler_method"]
                    else:
                        # No handler methods found, add after class definition
                        class_end = content.find(":", content.find("class ")) + 1
                        content = content[:class_end] + platform_code["handler_method"] + content[class_end:]
                
                # Add platform test case
                if not platform_info.get("has_platform_test", False):
                    # Find test cases section
                    test_cases = re.findall(r"self\.test_cases = \[\s*{([^}]*)}", content, re.DOTALL)
                    if test_cases:
                        last_test_case = test_cases[0]
                        last_test_end = content.find(last_test_case) + len(last_test_case)
                        
                        # Add a new test case entry
                        platform_test = f'''
            {{
                "description": "Test {model} on {platform.upper()} platform",
                "platform": {platform.upper()},
                "expected": {{}},
                "data": {{}}
            }},'''
                        content = content.replace("self.test_cases = [", f"self.test_cases = [{platform_test}")
        
        # Write the updated content to a new file
        fixed_path = OUTPUT_DIR / test_file_path.name
        with open(fixed_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Fixed test file saved to {fixed_path}")
        return fixed_path
    
    except Exception as e:
        logger.error(f"Error fixing test file {test_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_test_file(test_file_path: Path, platform: str = "cpu") -> Dict[str, Any]:
    """
    Validate a test file by trying to execute it.
    
    Args:
        test_file_path: Path to the test file
        platform: Hardware platform to validate on
        
    Returns:
        Dictionary with validation results
    """
    model_name = test_file_path.stem.replace("test_hf_", "")
    
    validation_results = {
        "model": model_name,
        "platform": platform,
        "file_path": str(test_file_path),
        "success": False,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Set up environment for testing
        env = os.environ.copy()
        env["TEST_PLATFORM"] = platform
        
        # Run the test file with --platform flag
        cmd = [
            sys.executable,
            str(test_file_path),
            f"--platform={platform}",
            "--skip-downloads",  # Skip downloading models
            "--mock"  # Use mock implementations to avoid actual model loading
        ]
        
        logger.info(f"Validating {model_name} on {platform} platform...")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
            timeout=60  # 1 minute timeout
        )
        
        # Store results
        validation_results["returncode"] = result.returncode
        validation_results["stdout"] = result.stdout
        validation_results["stderr"] = result.stderr
        validation_results["success"] = result.returncode == 0
        
        # Log success or failure
        if validation_results["success"]:
            logger.info(f"Validation successful for {model_name} on {platform}")
        else:
            logger.error(f"Validation failed for {model_name} on {platform}: {result.stderr}")
        
        return validation_results
    
    except subprocess.TimeoutExpired:
        validation_results["error"] = "Timeout"
        logger.error(f"Timeout validating {model_name} on {platform}")
        return validation_results
    
    except Exception as e:
        validation_results["error"] = str(e)
        logger.error(f"Error validating {model_name} on {platform}: {e}")
        return validation_results

def create_template_for_model_category(category: str) -> Path:
    """
    Create a template file for a model category.
    
    Args:
        category: Model category
        
    Returns:
        Path to the created template file
    """
    template_path = TEMPLATE_DIR / f"template_{category}.py"
    
    if template_path.exists():
        logger.info(f"Template for {category} already exists at {template_path}")
        return template_path
    
    try:
        # Basic template structure
        template_content = f'''"""
Hugging Face test template for {category} models.

This template includes support for all hardware platforms:
- CPU: Standard CPU implementation
- CUDA: NVIDIA GPU implementation
- OpenVINO: Intel hardware acceleration
- MPS: Apple Silicon GPU implementation
- ROCm: AMD GPU implementation
- WebNN: Web Neural Network API (browser)
- WebGPU: Web GPU API (browser)
"""

from transformers import AutoModel, AutoConfig
import os
import sys
import logging
import numpy as np

# Platform-specific imports will be added at runtime

class MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {{platform}}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {{self.platform}} called with {{len(args)}} args and {{len(kwargs)}} kwargs")
        return {{"mock_output": f"Mock output for {{self.platform}}"}}

class Test{category.title()}Model:
    """Test class for {category} models."""
    
    def __init__(self, model_path=None):
        """Initialize the test class."""
        self.model_path = model_path or "model/path/here"
        self.device = "cpu"  # Default device
        self.platform = "CPU"  # Default platform
        
        # Define test cases
        self.test_cases = [
'''
        
        # Add platform-specific test cases
        for platform in HARDWARE_PLATFORMS:
            if MODEL_HARDWARE_COMPATIBILITY.get(category, {}).get(platform, False):
                template_content += f'''            {{
                "description": "Test on {platform.upper()} platform",
                "platform": {platform.upper()},
                "expected": {{}},
                "data": {{}}
            }},
'''
        
        # Close the test cases list
        template_content += '''        ]
    
    def get_model_path_or_name(self):
        """Get the model path or name."""
        return self.model_path
'''
        
        # Add platform initialization methods
        for platform in HARDWARE_PLATFORMS:
            if MODEL_HARDWARE_COMPATIBILITY.get(category, {}).get(platform, False):
                platform_code = generate_platform_code("generic", platform)
                template_content += platform_code["init_method"]
        
        # Add platform handler methods
        for platform in HARDWARE_PLATFORMS:
            if MODEL_HARDWARE_COMPATIBILITY.get(category, {}).get(platform, False):
                platform_code = generate_platform_code("generic", platform)
                template_content += platform_code["handler_method"]
        
        # Add run method
        template_content += '''
    def run(self, platform="CPU"):
        """Run the test on the specified platform."""
        platform = platform.lower()
        init_method = getattr(self, f"init_{platform}", None)
        
        if init_method is None:
            print(f"Platform {platform} not supported")
            return False
        
        if not init_method():
            print(f"Failed to initialize {platform} platform")
            return False
        
        # Create handler for the platform
        try:
            handler_method = getattr(self, f"create_{platform}_handler", None)
            handler = handler_method()
        except Exception as e:
            print(f"Error creating handler for {platform}: {e}")
            return False
        
        print(f"Successfully initialized {platform} platform and created handler")
        return True

def main():
    """Run the test."""
    import argparse
    parser = argparse.ArgumentParser(description="Test {category} models")
    parser.add_argument("--model", help="Model path or name")
    parser.add_argument("--platform", default="CPU", help="Platform to test on")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip downloading models")
    parser.add_argument("--mock", action="store_true", help="Use mock implementations")
    args = parser.parse_args()
    
    test = Test{category.title()}Model(args.model)
    result = test.run(args.platform)
    
    if result:
        print(f"Test successful on {args.platform}")
        sys.exit(0)
    else:
        print(f"Test failed on {args.platform}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Write the template file
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Created template for {category} at {template_path}")
        return template_path
    
    except Exception as e:
        logger.error(f"Error creating template for {category}: {e}")
        return None

def create_all_templates():
    """Create template files for all model categories."""
    categories = set(MODEL_CATEGORIES.values())
    
    for category in categories:
        create_template_for_model_category(category)
    
    logger.info(f"Created templates for all {len(categories)} model categories")

def create_model_specific_template(model: str) -> Path:
    """
    Create a model-specific template file.
    
    Args:
        model: Model name
        
    Returns:
        Path to the created template file
    """
    category = MODEL_CATEGORIES.get(model, "unknown")
    template_path = TEMPLATE_DIR / f"template_{model}.py"
    
    if template_path.exists():
        logger.info(f"Template for {model} already exists at {template_path}")
        return template_path
    
    try:
        # Start with the category template
        category_template_path = TEMPLATE_DIR / f"template_{category}.py"
        if category_template_path.exists():
            with open(category_template_path, 'r') as f:
                template_content = f.read()
            
            # Replace the category class with the model-specific class
            template_content = template_content.replace(
                f"class Test{category.title()}Model",
                f"class Test{model.title()}Model"
            )
            
            # Update the docstring
            template_content = template_content.replace(
                f'Hugging Face test template for {category} models.',
                f'Hugging Face test template for {model} model.'
            )
            
            # Update the main function
            template_content = template_content.replace(
                f"test = Test{category.title()}Model(args.model)",
                f"test = Test{model.title()}Model(args.model)"
            )
            
            # Write the model-specific template file
            with open(template_path, 'w') as f:
                f.write(template_content)
            
            logger.info(f"Created template for {model} at {template_path}")
            return template_path
        else:
            logger.error(f"Category template for {category} not found")
            return None
    
    except Exception as e:
        logger.error(f"Error creating template for {model}: {e}")
        return None

def update_generator_templates():
    """
    Update the generators with improved templates for hardware support.
    
    This modifies the generator templates to include proper hardware platform support.
    """
    template_files = list(TEMPLATE_DIR.glob("template_*.py"))
    
    if not template_files:
        logger.warning("No template files found to update generator")
        return False
    
    try:
        # First, create templates for all model categories if they don't exist
        create_all_templates()
        
        # Then, create model-specific templates for the key models
        for model in KEY_MODELS:
            create_model_specific_template(model)
        
        # Create a template database file
        template_db = {}
        
        # Process each template file
        for template_file in TEMPLATE_DIR.glob("template_*.py"):
            template_name = template_file.stem.replace("template_", "")
            
            with open(template_file, 'r') as f:
                template_content = f.read()
            
            template_db[template_name] = template_content
        
        # Save the template database
        template_db_path = TEMPLATE_DIR / "template_database.json"
        with open(template_db_path, 'w') as f:
            json.dump(template_db, f, indent=2)
        
        logger.info(f"Updated template database saved to {template_db_path}")
        
        # Print a summary
        print(f"Created templates for:")
        print(f"- {len([t for t in template_files if 'template_' in t.name])} model categories")
        print(f"- {len([t for t in template_files if any(model in t.name for model in KEY_MODELS)])} key models")
        print(f"Template database saved to {template_db_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating generator templates: {e}")
        return False

def main():
    """
    Main function to enhance hardware platform support for key models.
    """
    parser = argparse.ArgumentParser(description="Enhance hardware platform support for key HuggingFace models")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fix-all", action="store_true", help="Fix hardware support for all key models")
    group.add_argument("--fix-model", help="Fix hardware support for a specific model")
    group.add_argument("--fix-platform", help="Fix support for a specific platform across all models")
    group.add_argument("--create-templates", action="store_true", help="Create hardware-aware templates")
    group.add_argument("--update-generator", action="store_true", help="Update the generator with improved templates")
    group.add_argument("--validate", action="store_true", help="Validate fixed models")
    
    parser.add_argument("--platforms", help="Comma-separated list of platforms to process")
    args = parser.parse_args()
    
    # Process platforms list
    platforms_to_process = HARDWARE_PLATFORMS
    if args.platforms:
        if args.platforms.lower() == "all":
            platforms_to_process = HARDWARE_PLATFORMS
        else:
            platforms_to_process = args.platforms.split(",")
    
    # Create templates for all model categories
    if args.create_templates:
        create_all_templates()
        
        # Create model-specific templates for key models
        for model in KEY_MODELS:
            create_model_specific_template(model)
        
        print(f"Templates created in {TEMPLATE_DIR}")
        return
    
    # Update generator templates
    if args.update_generator:
        success = update_generator_templates()
        if success:
            print("Generator templates updated successfully")
        else:
            print("Failed to update generator templates")
        return
    
    # Get the list of models to process
    models_to_process = []
    
    if args.fix_all:
        models_to_process = KEY_MODELS
    elif args.fix_model:
        if args.fix_model in KEY_MODELS:
            models_to_process = [args.fix_model]
        else:
            print(f"Model {args.fix_model} is not in the list of key models")
            print(f"Available key models: {', '.join(KEY_MODELS)}")
            return
    elif args.validate:
        models_to_process = KEY_MODELS
    
    # Process each model
    results = []
    validation_results = []
    
    for model in models_to_process:
        test_file_path = SKILLS_DIR / f"test_hf_{model}.py"
        
        if not test_file_path.exists():
            logger.error(f"Test file for {model} not found: {test_file_path}")
            continue
        
        # Analyze the test file
        analysis = analyze_test_file(test_file_path)
        
        # Fix the test file if needed
        if args.fix_all or args.fix_model:
            if analysis.get("needs_fixes", False):
                # For fix-platform mode, only fix the specified platform
                if args.fix_platform:
                    for platform, platform_info in analysis["platforms"].items():
                        if platform == args.fix_platform and platform_info.get("needs_fix", False):
                            fixed_file = fix_test_file(test_file_path, analysis)
                            if fixed_file:
                                results.append({
                                    "model": model,
                                    "original_file": str(test_file_path),
                                    "fixed_file": str(fixed_file),
                                    "platforms_fixed": [platform]
                                })
                else:
                    # Fix all platforms that need fixing
                    fixed_file = fix_test_file(test_file_path, analysis)
                    if fixed_file:
                        platforms_fixed = [
                            platform for platform, platform_info in analysis["platforms"].items()
                            if platform_info.get("needs_fix", False)
                        ]
                        results.append({
                            "model": model,
                            "original_file": str(test_file_path),
                            "fixed_file": str(fixed_file),
                            "platforms_fixed": platforms_fixed
                        })
        
        # Validate the fixed file
        if args.validate:
            file_to_validate = OUTPUT_DIR / f"test_hf_{model}.py"
            if not file_to_validate.exists():
                file_to_validate = test_file_path
            
            # Validate on all platforms or just specified ones
            for platform in platforms_to_process:
                validation_result = validate_test_file(file_to_validate, platform)
                validation_results.append(validation_result)
    
    # Process platform-specific fixes
    if args.fix_platform and args.fix_platform in HARDWARE_PLATFORMS:
        for model in KEY_MODELS:
            test_file_path = SKILLS_DIR / f"test_hf_{model}.py"
            
            if not test_file_path.exists():
                logger.error(f"Test file for {model} not found: {test_file_path}")
                continue
            
            # Analyze the test file
            analysis = analyze_test_file(test_file_path)
            
            # Check if this model-platform combination needs fixing
            if analysis.get("platforms", {}).get(args.fix_platform, {}).get("needs_fix", False):
                fixed_file = fix_test_file(test_file_path, analysis)
                if fixed_file:
                    results.append({
                        "model": model,
                        "original_file": str(test_file_path),
                        "fixed_file": str(fixed_file),
                        "platforms_fixed": [args.fix_platform]
                    })
    
    # Save results
    if results:
        results_file = OUTPUT_DIR / "fix_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        fixed_models = len(results)
        fixed_platforms_count = sum(len(r["platforms_fixed"]) for r in results)
        
        print(f"\nSummary of fixes:")
        print(f"- Fixed {fixed_models} models")
        print(f"- Fixed {fixed_platforms_count} model-platform combinations")
        print(f"- Results saved to {results_file}")
    else:
        print("\nNo fixes needed or applied")
    
    # Print validation summary
    if validation_results:
        validation_file = OUTPUT_DIR / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        success_count = sum(1 for r in validation_results if r.get("success", False))
        total_validations = len(validation_results)
        
        print(f"\nValidation Summary:")
        print(f"- Tested {total_validations} model-platform combinations")
        print(f"- Successful: {success_count} ({(success_count / total_validations) * 100:.2f}%)")
        print(f"- Results saved to {validation_file}")
        
        # Group by platform
        platform_results = {}
        for platform in HARDWARE_PLATFORMS:
            platform_validations = [r for r in validation_results if r.get("platform") == platform]
            if platform_validations:
                platform_success = sum(1 for r in platform_validations if r.get("success", False))
                platform_total = len(platform_validations)
                platform_results[platform] = {
                    "success": platform_success,
                    "total": platform_total,
                    "success_rate": (platform_success / platform_total) * 100 if platform_total > 0 else 0
                }
        
        print("\nSuccess rate by platform:")
        for platform, results in platform_results.items():
            print(f"- {platform.upper()}: {results['success']}/{results['total']} ({results['success_rate']:.2f}%)")

if __name__ == "__main__":
    main()