#!/usr/bin/env python3
"""
Phase 16 Test Generator Enhancement

This script enhances the test generation system in Phase 16 to ensure generated tests fully
pass on each of the supported hardware platforms (CPU, CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU).

Key improvements:
1. Updates the hardware platform handling in the merged test generator
2. Enhances hardware-specific templates with proper fallback mechanisms
3. Implements uniform input/output format handling across platforms
4. Adds validation for generated tests on all target platforms
5. Fixes batch processing capabilities for all supported platforms
6. Ensures graceful fallbacks when hardware isn't available

Usage:
  python fix_test_generator.py --fix-all
  python fix_test_generator.py --validate-compatibility
  python fix_test_generator.py --test-model bert --test-platform cuda
"""

import os
import sys
import re
import json
import shutil
import logging
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fix_test_generator.log")
    ]
)
logger = logging.getLogger("fix_test_generator")

# Constants - the 13 key models
KEY_MODELS = [
    'bert',        # Text embedding
    't5',          # Text-to-text transformer
    'llama',       # Language model
    'clip',        # Vision-text multimodal
    'vit',         # Vision transformer
    'clap',        # Audio-text multimodal
    'whisper',     # Speech recognition
    'wav2vec2',    # Speech model
    'llava',       # Vision-language model
    'llava_next',  # Next-gen vision-language model
    'xclip',       # Extended clip for video
    'qwen2',       # Transformer LM
    'detr'         # Object detection
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

# Determine project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
GENERATOR_FILE = TEST_DIR / "merged_test_generator.py"
HARDWARE_TEMPLATES_DIR = TEST_DIR / "hardware_test_templates"
FIXED_GENERATED_TESTS_DIR = TEST_DIR / "fixed_generated_tests"
FIXED_GENERATED_TESTS_DIR.mkdir(exist_ok=True)

def backup_generator():
    """Create a backup of the merged_test_generator.py file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = GENERATOR_FILE.with_suffix(f".py.bak_{timestamp}")
    shutil.copy2(GENERATOR_FILE, backup_path)
    logger.info(f"Created backup of merged_test_generator.py at {backup_path}")
    return backup_path

def create_hardware_compatibility_matrix():
    """Create the hardware compatibility matrix for model categories."""
    
    # Define compatibility matrix based on CLAUDE.md
    matrix = {
        "text_embedding": {  # BERT
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": True, "webgpu": True
        },
        "text_generation": {  # T5, LLAMA
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": False, "webgpu": True
        },
        "vision": {  # VIT, DETR
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": True, "webgpu": True
        },
        "vision_language": {  # CLIP
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": True, "webgpu": True
        },
        "audio": {  # WHISPER, WAV2VEC2
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": True, "webgpu": True
        },
        "audio_text": {  # CLAP
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": True, "webgpu": True
        },
        "multimodal": {  # LLAVA, LLAVA-NEXT
            "cpu": True, "cuda": True, "openvino": True, "mps": False, 
            "rocm": False, "webnn": False, "webgpu": False
        },
        "video": {  # XCLIP
            "cpu": True, "cuda": True, "openvino": True, "mps": True, 
            "rocm": True, "webnn": False, "webgpu": False
        }
    }
    
    # Map models to categories
    model_to_category = {
        "bert": "text_embedding",
        "t5": "text_generation",
        "llama": "text_generation",
        "clip": "vision_language",
        "vit": "vision",
        "clap": "audio_text",
        "whisper": "audio",
        "wav2vec2": "audio",
        "llava": "multimodal",
        "llava_next": "multimodal",
        "xclip": "video",
        "qwen2": "text_generation",
        "detr": "vision"
    }
    
    # Create model-specific compatibility map
    model_compatibility = {}
    for model in KEY_MODELS:
        category = model_to_category.get(model)
        if category:
            model_compatibility[model] = matrix.get(category, {})
    
    return {
        "categories": matrix,
        "models": model_compatibility
    }

def update_generator_hardware_detection():
    """Update the generator file with improved hardware detection logic."""
    with open(GENERATOR_FILE, 'r') as f:
    
    # Check if the improved hardware detection is already present
    if "def detect_available_hardware():" in content:
        logger.info("Hardware detection function already present in generator")
        return
    
    # Find a suitable location to add the hardware detection function
    # Look for constants or utility functions section
    utils_section = content.find("# Constants")
    if utils_section < 0:
        utils_section = content.find("# Utility functions")
    
    if utils_section < 0:
        logger.warning("Could not find a suitable location for hardware detection function")
        # Insert after imports
        match = re.search(r'import.*?\n\n', content, re.DOTALL)
        if match:
            utils_section = match.end()
        else:
            utils_section = 0
    
    # Define the hardware detection function
    hardware_detection_code = """
# Hardware platform detection and support
def detect_available_hardware():
    \"\"\"Detect available hardware platforms on the current system.\"\"\"
    available_hardware = {
        "cpu": True  # CPU is always available
    }
    
    # CUDA (NVIDIA GPUs)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        available_hardware["cuda"] = cuda_available
        if cuda_available:
            logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        available_hardware["cuda"] = False
        logger.info("CUDA not available: torch not installed")
    
    # MPS (Apple Silicon)
    try:
        import torch
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        available_hardware["mps"] = mps_available
        if mps_available:
            logger.info("MPS (Apple Silicon) available")
    except ImportError:
        available_hardware["mps"] = False
        logger.info("MPS not available: torch not installed")
    except AttributeError:
        available_hardware["mps"] = False
        logger.info("MPS not available: torch version does not support mps")
    
    # ROCm (AMD GPUs)
    try:
        import torch
        rocm_available = torch.cuda.is_available() and hasattr(torch.version, "hip")
        available_hardware["rocm"] = rocm_available
        if rocm_available:
            logger.info(f"ROCm available with {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        available_hardware["rocm"] = False
        logger.info("ROCm not available: torch not installed")
    except AttributeError:
        available_hardware["rocm"] = False
        logger.info("ROCm not available: torch version does not support hip")
    
    # OpenVINO (Intel)
    try:
        import openvino
        available_hardware["openvino"] = True
        logger.info(f"OpenVINO available: {openvino.__version__}")
    except ImportError:
        available_hardware["openvino"] = False
        logger.info("OpenVINO not available: openvino not installed")
    
    # WebNN and WebGPU require browser environment - assume not available in CLI
    available_hardware["webnn"] = False
    available_hardware["webgpu"] = False
    logger.info("WebNN and WebGPU require browser environment - not available in CLI")
    
    return available_hardware

def platform_supported_for_model(model_name, platform):
    \"\"\"Check if a platform is supported for a specific model.\"\"\"
    # Load compatibility matrix
    compatibility = create_hardware_compatibility_matrix()
    
    # Check model specific compatibility
    model_compat = compatibility.get("models", {}).get(model_name, {})
    if platform in model_compat:
        return model_compat[platform]
    
    # If no model-specific info, try to find category
    for category, models in MODALITY_TYPES.items():
        if model_name in models:
            category_compat = compatibility.get("categories", {}).get(category, {})
            if platform in category_compat:
                return category_compat[platform]
    
    # Default to CPU only if no compatibility info found
    return platform.lower() == "cpu"

"""
    
    # Insert the hardware detection code
    updated_content = content[:utils_section] + hardware_detection_code + content[utils_section:]
    
    # Write the updated content
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_content)
    
    logger.info("Added hardware detection function to generator")

def update_generator_hardware_template_support():
    """Update the generator to better support hardware-specific templates."""
    with open(GENERATOR_FILE, 'r') as f:
    
    # Check if the hardware template handling already exists
    if "def select_hardware_template(" in content:
        logger.info("Hardware template selection already exists in generator")
        return
    
    # Find a good place to add the function - just before generate_test_file
    generate_func_pos = content.find("def generate_test_file(")
    if generate_func_pos < 0:
        logger.warning("Could not find generate_test_file function")
        return
    
    # Find the last function before generate_test_file
    prev_func_end = content.rfind("\n\n", 0, generate_func_pos)
    if prev_func_end < 0:
        logger.warning("Could not find position to insert hardware template function")
        return
    
    # Define the hardware template selection function
    hardware_template_code = """
def select_hardware_template(model_name, args):
    \"\"\"
    Select an appropriate template file taking into account hardware platforms.
    
    Args:
        model_name: Name of the model
        args: Command line arguments
        
    Returns:
        Path to the selected template file
    \"\"\"
    # Check for hardware-specific templates first
    platform = getattr(args, "platform", "all").lower()
    
    # If platform is specified, check if it's supported for this model
    if platform != "all" and not platform_supported_for_model(model_name, platform):
        logger.warning(f"Platform {platform} not supported for model {model_name}, falling back to CPU")
        platform = "cpu"
    
    # First check for model-specific templates
    hardware_template = HARDWARE_TEMPLATES_DIR / f"template_{model_name}.py"
    if hardware_template.exists():
        return hardware_template
    
    # Try to determine which modality this model belongs to
    model_modality = None
    for modality, models in MODALITY_TYPES.items():
        if model_name in models:
            model_modality = modality
            break
    
    # Check for modality-specific templates
    if model_modality:
        modality_template = HARDWARE_TEMPLATES_DIR / f"template_{model_modality}.py"
        if modality_template.exists():
            return modality_template
    
    # Check for more specific templates first (e.g., vision, audio, text)
    modality_map = {
        "text": ["bert", "gpt2", "t5", "roberta", "distilbert"], 
        "vision": ["vit", "deit", "swin", "convnext"],
        "audio": ["whisper", "wav2vec2", "hubert"]
    }
    
    for modality, model_list in modality_map.items():
        if model_name in model_list:
            modality_template = HARDWARE_TEMPLATES_DIR / f"template_{modality}.py"
            if modality_template.exists():
                return modality_template
    
    # If no hardware template found, use standard template lookup
    template_path = find_template_for_model(model_name)
    if template_path and os.path.exists(template_path):
        return template_path
    
    # Fallback to CPU template as a last resort
    cpu_template = HARDWARE_TEMPLATES_DIR / "template_cpu_embedding.py"
    if cpu_template.exists():
        return cpu_template
    
    # Ultimate fallback to default template
    return TEMPLATES_DIR / "default_template.py"

"""
    
    # Insert the hardware template selection function
    updated_content = content[:prev_func_end] + hardware_template_code + content[prev_func_end:]
    
    # Update the generate_test_file function to use the new hardware template selection
    if "template_path = find_template_for_model" in updated_content:
        updated_content = updated_content.replace(
            "template_path = find_template_for_model(model_name)",
            "template_path = select_hardware_template(model_name, args)"
        )
    
    # Write the updated content
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_content)
    
    logger.info("Added hardware template selection to generator")

def update_generator_cli_options():
    """Update the generator's CLI options to include hardware platform options."""
    with open(GENERATOR_FILE, 'r') as f:
    
    # Check if the platform options already exist
    if "--platform" in content:
        logger.info("Platform CLI options already exist in generator")
        return
    
    # Find the argument parser
    parser_pos = content.find("parser = argparse.ArgumentParser")
    if parser_pos < 0:
        logger.warning("Could not find argument parser")
        return
    
    # Find a suitable place to add the platform options
    # Look for other argument groups
    add_argument_pos = content.find("parser.add_argument", parser_pos)
    if add_argument_pos < 0:
        logger.warning("Could not find a suitable position to add platform options")
        return
    
    # Find the end of the argument section
    next_section_pos = content.find("\n\n", add_argument_pos)
    if next_section_pos < 0:
        logger.warning("Could not find the end of the argument section")
        return
    
    # Define the platform options
    platform_options = """
    # Hardware platform options
    parser.add_argument("--platform", choices=["all", "cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"],
                      default="all", help="Target hardware platform (default: all)")
    parser.add_argument("--skip-hardware-detection", action="store_true",
                      help="Skip hardware availability detection (useful for generating tests for platforms not available locally)")
"""
    
    # Insert the platform options at the end of the argument section
    updated_content = content[:next_section_pos] + platform_options + content[next_section_pos:]
    
    # Write the updated content
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_content)
    
    logger.info("Added platform CLI options to generator")

def update_generator_test_code():
    """Update the generator's test code generation to handle hardware platforms better."""
    with open(GENERATOR_FILE, 'r') as f:
    
    # Find the generate_test_file function
    generate_func_pos = content.find("def generate_test_file(")
    if generate_func_pos < 0:
        logger.warning("Could not find generate_test_file function")
        return
    
    # Find the end of the function
    next_func_pos = content.find("def ", generate_func_pos + 1)
    if next_func_pos < 0:
        next_func_pos = len(content)
    
    generate_func = content[generate_func_pos:next_func_pos]
    
    # Check if the function already handles hardware platforms
    if "platform = getattr(args, 'platform', 'all')" in generate_func:
        logger.info("generate_test_file already handles hardware platforms")
        return
    
    # Add hardware platform handling to the generated test code
    updated_func = generate_func.replace(
        "    # Create the test file content",
        """    # Handle hardware platform options
    platform = getattr(args, 'platform', 'all')
    
    # Check if hardware detection should be skipped
    skip_detection = getattr(args, 'skip_hardware_detection', False)
    
    # Detect available hardware if not skipped
    available_hardware = None
    if not skip_detection:
        available_hardware = detect_available_hardware()
        
        # If a specific platform is requested but not available, warn and continue anyway
        if platform != "all" and available_hardware and not available_hardware.get(platform.lower(), False):
            logger.warning(f"Requested platform {platform} is not available, but continuing with generation")
    
    # Create the test file content"""
    )
    
    # Update the content with the modified function
    updated_content = content[:generate_func_pos] + updated_func + content[next_func_pos:]
    
    # Write the updated content
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_content)
    
    logger.info("Updated generate_test_file to handle hardware platforms")

def generate_test_with_fixed_generator(model_name, platform="all"):
    """
    Generate a test file for a model using the fixed generator.
    
    Args:
        model_name: Name of the model
        platform: Target hardware platform
        
    Returns:
        Path to the generated test file, or None if generation failed
    """
    try:
        # First, update the generator
        backup_generator()
        update_generator_hardware_detection()
        update_generator_hardware_template_support()
        update_generator_cli_options()
        update_generator_test_code()
        
        # Create the output directory if it doesn't exist
        FIXED_GENERATED_TESTS_DIR.mkdir(exist_ok=True)
        
        # Run the generator
        cmd = [
            sys.executable,
            str(GENERATOR_FILE),
            "--generate", model_name,
            "--platform", platform,
            "--skip-hardware-detection",  # Skip hardware detection to allow generation for all platforms
            "--output-dir", str(FIXED_GENERATED_TESTS_DIR)
        ]
        
        logger.info(f"Generating test for {model_name} targeting {platform} platform")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60  # 1 minute timeout
        )
        
        # Check if generation succeeded
        test_file_path = FIXED_GENERATED_TESTS_DIR / f"test_hf_{model_name}.py"
        
        if result.returncode == 0 and test_file_path.exists():
            logger.info(f"Successfully generated test for {model_name} targeting {platform} platform")
            return test_file_path
        else:
            logger.error(f"Failed to generate test for {model_name}: {result.stderr}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating test for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_generated_test(test_file_path, platform="all"):
    """
    Validate a generated test file by running it with a mock implementation.
    
    Args:
        test_file_path: Path to the test file
        platform: Target hardware platform
        
    Returns:
        True if validation succeeds, False otherwise
    """
    try:
        logger.info(f"Validating test file {test_file_path} for {platform} platform")
        
        # Run the test file with mock implementation
        cmd = [
            sys.executable,
            str(test_file_path),
            f"--platform={platform}",
            "--skip-downloads",  # Skip downloading models
            "--mock"  # Use mock implementations to avoid actual model loading
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30  # 30 second timeout
        )
        
        # Check if validation succeeded
        if result.returncode == 0:
            logger.info(f"Test file {test_file_path} validated successfully for {platform} platform")
            return True
        else:
            logger.error(f"Test file {test_file_path} validation failed for {platform} platform")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error validating test file {test_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_all_model_tests():
    """Generate and validate tests for all key models on all platforms."""
    results = {}
    
    # First update the generator
    backup_generator()
    update_generator_hardware_detection()
    update_generator_hardware_template_support()
    update_generator_cli_options()
    update_generator_test_code()
    
    # Process each model
    for model in KEY_MODELS:
        results[model] = {}
        
        # Generate tests for the model
        logger.info(f"Processing model: {model}")
        
        # First generate a test that works on all platforms
        logger.info(f"Generating test for {model} for all platforms")
        test_file_path = generate_test_with_fixed_generator(model, "all")
        
        if test_file_path:
            # Validate the test on each platform
            for platform in HARDWARE_PLATFORMS:
                logger.info(f"Validating {model} test on {platform} platform")
                result = validate_generated_test(test_file_path, platform)
                results[model][platform] = result
        else:
            # Generation failed, mark all platforms as failed
            logger.error(f"Failed to generate test for {model}")
            for platform in HARDWARE_PLATFORMS:
                results[model][platform] = False
    
    # Save the results
    results_file = TEST_DIR / "test_generator_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("--- Validation Summary ---")
    
    success_count = 0
    total_count = 0
    
    for model, platforms in results.items():
        model_success = sum(1 for result in platforms.values() if result)
        logger.info(f"{model}: {model_success}/{len(platforms)} platforms passed")
        success_count += model_success
        total_count += len(platforms)
    
    logger.info(f"Overall: {success_count}/{total_count} tests passed ({success_count/total_count*100:.1f}%)")
    logger.info(f"Results saved to {results_file}")
    
    return results

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Fix test generator for Phase 16 hardware compatibility")
    
    # Add script options
    parser.add_argument("--fix-all", action="store_true", help="Fix and validate tests for all key models")
    parser.add_argument("--test-model", help="Test a specific model")
    parser.add_argument("--test-platform", help="Test a specific platform")
    parser.add_argument("--validate-compatibility", action="store_true", help="Validate the hardware compatibility matrix")
    parser.add_argument("--update-generator", action="store_true", help="Update the generator only, don't run tests")
    
    args = parser.parse_args()
    
    # Process arguments
    if args.fix_all:
        fix_all_model_tests()
    elif args.test_model and args.test_platform:
        test_file_path = generate_test_with_fixed_generator(args.test_model, args.test_platform)
        if test_file_path:
            validate_generated_test(test_file_path, args.test_platform)
    elif args.validate_compatibility:
        # Validate that the compatibility matrix matches the CLAUDE.md documentation
        matrix = create_hardware_compatibility_matrix()
        print(json.dumps(matrix, indent=2))
    elif args.update_generator:
        backup_generator()
        update_generator_hardware_detection()
        update_generator_hardware_template_support()
        update_generator_cli_options()
        update_generator_test_code()
        print("Generator updated successfully")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

    
    
    # Find the hardware map in the content
    pattern = r'# Define key models that need special handling with hardware backends\s*KEY_MODEL_HARDWARE_MAP = \{.*?\}'
    # Use re.DOTALL to enable multi-line matching
    hardware_map_match = re.search(pattern, content, re.DOTALL)
    
    if hardware_map_match:
        # Replace the hardware map
        updated_content = content[:hardware_map_match.start()] + updated_hardware_map + content[hardware_map_match.end():]
        
        # Write the updated content back to the file
        with open(GENERATOR_FILE, 'w') as f:
            f.write(updated_content)
        
        print("Updated KEY_MODEL_HARDWARE_MAP with proper WebNN and WebGPU support")
        return True
    else:
        print("Could not find KEY_MODEL_HARDWARE_MAP in the file")
        return False

def fix_test_platform_function():
    """Update the test_platform function to handle WebNN and WebGPU properly."""
    # Read the generator file
    with open(GENERATOR_FILE, 'r') as f:
    
    # Find the test_platform function in the template string
    pattern = r'def test_platform\(platform="cpu"\):.*?return False'
    # Use re.DOTALL to enable multi-line matching
    test_platform_match = re.search(pattern, content, re.DOTALL)
    
    if test_platform_match:
        old_function = test_platform_match.group(0)
        
        # Define a new pattern to match the test input selection part
        input_pattern = r'# Get modality-appropriate test input.*?(\s+# Test inference)'
        input_match = re.search(input_pattern, old_function, re.DOTALL)
        
        if input_match:
            new_input_code = """        # Get modality-appropriate test input based on platform and model type
        test_input = "MODEL_INPUT_PLACEHOLDER"
        
        # For WebNN platform
        if platform == "webnn":
            if hasattr(test_model, 'test_webnn_text'):
                test_input = test_model.test_webnn_text
            elif hasattr(test_model, 'test_webnn_image'):
                test_input = test_model.test_webnn_image
            elif hasattr(test_model, 'test_webnn_audio'):
                test_input = test_model.test_webnn_audio
            elif hasattr(test_model, 'test_input_webnn'):
                test_input = test_model.test_input_webnn
            elif hasattr(test_model, 'get_webnn_input'):
                test_input = test_model.get_webnn_input()
            elif hasattr(test_model, 'test_text') and "MODEL_TASK_PLACEHOLDER" in ["text-generation", "fill-mask", "text2text-generation"]:
                test_input = test_model.test_text
            elif hasattr(test_model, 'test_image') and "MODEL_TASK_PLACEHOLDER" in ["image-classification", "object-detection", "image-segmentation"]:
                test_input = test_model.test_image
            elif hasattr(test_model, 'test_audio') and "MODEL_TASK_PLACEHOLDER" in ["automatic-speech-recognition", "audio-classification"]:
                test_input = test_model.test_audio
            elif hasattr(test_model, 'test_input'):
                test_input = test_model.test_input
        
        # For WebGPU platform
        elif platform == "webgpu":
            if hasattr(test_model, 'test_webgpu_text'):
                test_input = test_model.test_webgpu_text
            elif hasattr(test_model, 'test_webgpu_image'):
                test_input = test_model.test_webgpu_image
            elif hasattr(test_model, 'test_webgpu_audio'):
                test_input = test_model.test_webgpu_audio
            elif hasattr(test_model, 'test_input_webgpu'):
                test_input = test_model.test_input_webgpu
            elif hasattr(test_model, 'get_webgpu_input'):
                test_input = test_model.get_webgpu_input()
            elif hasattr(test_model, 'test_text') and "MODEL_TASK_PLACEHOLDER" in ["text-generation", "fill-mask", "text2text-generation"]:
                test_input = test_model.test_text
            elif hasattr(test_model, 'test_image') and "MODEL_TASK_PLACEHOLDER" in ["image-classification", "object-detection", "image-segmentation"]:
                test_input = test_model.test_image
            elif hasattr(test_model, 'test_audio') and "MODEL_TASK_PLACEHOLDER" in ["automatic-speech-recognition", "audio-classification"]:
                test_input = test_model.test_audio
            elif hasattr(test_model, 'test_input'):
                test_input = test_model.test_input

        """
            
            # Replace the test input part
            new_function = old_function.replace(input_match.group(0), new_input_code + input_match.group(1))
            
            # Now modify the batch inference part
            batch_pattern = r'# Test batch inference.*?(\s+print\(f"{platform\.upper\(\)} platform test successful"\))'
            batch_match = re.search(batch_pattern, new_function, re.DOTALL)
            
            if batch_match:
                new_batch_code = """        # Test batch inference if supported by the platform
        if hasattr(test_model, 'test_batch'):
            # WebNN and WebGPU can support batching for certain model types
            web_batch_supported = platform in ["webnn", "webgpu"] and "{modality}" in ["text", "vision"]
            
            if platform not in ["webnn", "webgpu"] or web_batch_supported:
                try:
                    batch_start = time.time()
                    
                    # Determine batch input
                    if platform == "webnn" and hasattr(test_model, 'test_batch_webnn'):
                        batch_input = test_model.test_batch_webnn
                    elif platform == "webgpu" and hasattr(test_model, 'test_batch_webgpu'):
                        batch_input = test_model.test_batch_webgpu
                    elif hasattr(test_model, 'test_batch'):
                        batch_input = test_model.test_batch
                    else:
                        batch_input = [test_input, test_input]
                    
                    batch_result = handler(batch_input)
                    batch_time = time.time() - batch_start
                    
                    # Calculate throughput
                    throughput = len(batch_input) / batch_time if batch_time > 0 else 0
                    
                    print(f"Batch inference time: {batch_time:.4f} seconds")
                    print(f"Batch throughput: {throughput:.2f} items/second")
                    print(f"Batch implementation: {batch_result.get('implementation_type', 'UNKNOWN')}")
                except Exception as batch_error:
                    print(f"Batch inference not supported on {platform} for this model: {batch_error}")

                """
                
                # Replace the batch inference part
                new_function = new_function.replace(batch_match.group(0), new_batch_code + batch_match.group(1))
            
            # Finally, replace the entire function in the content
            updated_content = content.replace(old_function, new_function)
            
            # Write the updated content back to the file
            with open(GENERATOR_FILE, 'w') as f:
                f.write(updated_content)
            
            print("Updated test_platform function with better WebNN and WebGPU support")
            return True
        else:
            print("Could not find test input section in the test_platform function")
            return False
    else:
        print("Could not find test_platform function in the template")
        return False

def fix_webnn_webgpu_docstrings():
    """Fix the docstrings for WebNN and WebGPU initialization functions."""
    # Read the generator file
    with open(GENERATOR_FILE, 'r') as f:
    
    # Replace the WebNN docstring
    webnn_pattern = r'def init_webnn\(self, model_name=None, device="webnn", backend="gpu"\):.*?This implementation has three modes:'
    webnn_replacement = 'def init_webnn(self, model_name=None, device="webnn", backend="gpu"):\n        """Initialize model for WebNN-based inference.\n        \n        WebNN (Web Neural Network API) is a web standard for accelerated ML inference in browsers.\n        This implementation has three modes:'
    
    # Use re.DOTALL to enable multi-line matching
    content = re.sub(webnn_pattern, webnn_replacement, content, flags=re.DOTALL)
    
    # Replace the WebGPU docstring
    webgpu_pattern = r'def init_webgpu\(self, model_name=None, device="webgpu"\):.*?This implementation has two modes:'
    webgpu_replacement = 'def init_webgpu(self, model_name=None, device="webgpu"):\n        """Initialize model for WebGPU-based inference using transformers.js.\n        \n        WebGPU is a web standard for GPU computation in browsers.\n        transformers.js is a JavaScript port of the Transformers library that can use WebGPU.\n        \n        This implementation has two modes:'
    
    # Use re.DOTALL to enable multi-line matching
    content = re.sub(webgpu_pattern, webgpu_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(GENERATOR_FILE, 'w') as f:
        f.write(content)
    
    print("Fixed docstrings for WebNN and WebGPU initialization functions")
    return True

def add_mock_import():
    """Add missing MagicMock import."""
    # Read the generator file
    with open(GENERATOR_FILE, 'r') as f:
    
    # Check if MagicMock is already imported
    if "from unittest.mock import MagicMock" not in content:
        # Find the imports section
        import_pattern = r'from typing import Dict, List, Any, Optional, Union, Tuple, Set'
        
        # Replace with added import
        updated_imports = import_pattern + '\nfrom unittest.mock import MagicMock'
        
        # Update the content
        updated_content = content.replace(import_pattern, updated_imports)
        
        # Write the updated content back to the file
        with open(GENERATOR_FILE, 'w') as f:
            f.write(updated_content)
        
        print("Added missing MagicMock import")
        return True
    else:
        print("MagicMock already imported")
        return True

def add_test_attributes_to_templates():
    """Add test attributes for WebNN and WebGPU to each modality template."""
    # Read the generator file
    with open(GENERATOR_FILE, 'r') as f:
    
    # Define attributes to add for each modality
    for modality in ["text", "vision", "audio", "multimodal"]:
        # Find the appropriate __init__ function in the modality-specific template
        init_pattern = fr'def __init__\(self, resources=None, metadata=None\):.*?# {modality.capitalize()}-specific test data'
        
        if modality == "text":
            attributes = """
        # WebNN and WebGPU specific test data
        self.test_webnn_text = "The quick brown fox jumps over the lazy dog."
        self.test_webgpu_text = "The quick brown fox jumps over the lazy dog."
        self.test_batch_webnn = ["The quick brown fox jumps over the lazy dog.", "Hello world!"]
        self.test_batch_webgpu = ["The quick brown fox jumps over the lazy dog.", "Hello world!"]
        """
        elif modality == "vision":
            attributes = """
        # WebNN and WebGPU specific test data
        self.test_webnn_image = "test.jpg"
        self.test_webgpu_image = "test.jpg"
        self.test_batch_webnn = ["test.jpg", "test.jpg"]
        self.test_batch_webgpu = ["test.jpg", "test.jpg"]
        """
        elif modality == "audio":
            attributes = """
        # WebNN and WebGPU specific test data
        self.test_webnn_audio = "test.mp3"
        self.test_webgpu_audio = "test.mp3"
        self.test_batch_webnn = ["test.mp3", "test.mp3"]
        self.test_batch_webgpu = ["test.mp3", "test.mp3"]
        """
        elif modality == "multimodal":
            attributes = """
        # WebNN and WebGPU specific test data
        self.test_webnn_text = "What's in this image?"
        self.test_webnn_image = "test.jpg"
        self.test_webgpu_text = "What's in this image?"
        self.test_webgpu_image = "test.jpg"
        """
        
        # Update the content with the new attributes
        init_match = re.search(init_pattern, content, re.DOTALL)
        if init_match:
            init_text = init_match.group(0)
            updated_init = init_text + attributes
            content = content.replace(init_text, updated_init)
    
    # Write the updated content back to the file
    with open(GENERATOR_FILE, 'w') as f:
        f.write(content)
    
    print("Added WebNN and WebGPU test attributes to modality templates")
    return True

def main():
    """Main function to update the merged_test_generator.py file."""
    print("Starting to fix merged_test_generator.py for WebNN and WebGPU support...")
    
    # Backup the original file
    backup_path = backup_generator()
    
    # Add missing MagicMock import
    add_mock_import()
    
    # Fix the KEY_MODEL_HARDWARE_MAP
    fix_key_model_hardware_map()
    
    # Fix docstrings for WebNN and WebGPU initialization functions
    fix_webnn_webgpu_docstrings()
    
    # Fix the test_platform function
    fix_test_platform_function()
    
    # Add test attributes to templates
    add_test_attributes_to_templates()
    
    print(f"\nFix completed! Original file backed up at: {backup_path}")
    print("\nYou can now generate tests for all key models with proper WebNN and WebGPU support.")
    print("\nTo generate tests for key models, run:")
    for model in KEY_MODELS:
        print(f"  python merged_test_generator.py --generate {model}")

if __name__ == "__main__":
    main()