#!/usr/bin/env python3
"""
Cleanup and prepare the integration environment for the HuggingFace test system.

This script:
1. Creates required directories
2. Sets up templates
3. Ensures the environment is ready for test integration
4. Creates basic templates if needed

Usage:
    python cleanup_integration.py [--templates] [--force]
"""

import os
import sys
import shutil
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
        logging.FileHandler(f"cleanup_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
FIXED_TESTS_DIR = os.path.join(ROOT_DIR, "fixed_tests")
COLLECTED_RESULTS_DIR = os.path.join(FIXED_TESTS_DIR, "collected_results")

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    else:
        logger.info(f"Directory already exists: {directory}")

def create_basic_templates(force=False):
    """Create basic templates for different architecture types."""
    # Define architecture types
    architecture_types = {
        "encoder_only": "Encoder-Only (BERT, RoBERTa, etc.)",
        "decoder_only": "Decoder-Only (GPT-2, LLaMA, etc.)",
        "encoder_decoder": "Encoder-Decoder (T5, BART, etc.)",
        "vision": "Vision (ViT, Swin, etc.)",
        "vision_text": "Vision-Text (CLIP, BLIP, etc.)",
        "speech": "Speech (Whisper, Wav2Vec2, etc.)",
        "multimodal": "Multimodal (LLaVA, etc.)"
    }
    
    # Create template files
    for arch_type, description in architecture_types.items():
        template_path = os.path.join(TEMPLATES_DIR, f"{arch_type}_template.py")
        
        # Skip if file exists and not forcing
        if os.path.exists(template_path) and not force:
            logger.info(f"Template already exists (skipping): {template_path}")
            continue
        
        # Create basic template content
        content = f"""#!/usr/bin/env python3
# Template for {description} models

import os
import sys
import json
import time
import datetime
import logging
import argparse
from unittest.mock import MagicMock
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
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

# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {{
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }}
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
    
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
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

# Main test class for {description} models
class Test{arch_type.title()}Model:
    \"\"\"Test class for {description} models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize test class.\"\"\"
        self.model_id = model_id or "MODEL_DEFAULT_ID"
        self.task = "MODEL_TASK"
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
    
    def test_pipeline(self, device="auto"):
        \"\"\"Test using pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        # Implementation for {description} models
        logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
        
        # Add test implementation here
        pass
    
    def test_from_pretrained(self, device="auto"):
        \"\"\"Test using from_pretrained API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        # Implementation for {description} models
        logger.info(f"Testing {{self.model_id}} with from_pretrained() on {{device}}...")
        
        # Add test implementation here
        pass
    
    def run_tests(self, all_hardware=False):
        \"\"\"Run all tests.\"\"\"
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
        
        # Test on all available hardware if requested
        if all_hardware:
            # Test on CPU if not already tested
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
        
        # Return results
        return {{
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES
        }}

def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {description} models")
    parser.add_argument("--model", type=str, help="Model ID to test")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    args = parser.parse_args()
    
    # Create tester
    tester = Test{arch_type.title()}Model(model_id=args.model)
    
    # Run tests
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Print results
    print(f"\\nResults for {{tester.model_id}}:")
    success = any(r.get("success", False) for r in results["results"].values())
    if success:
        print("✅ Tests passed")
    else:
        print("❌ Tests failed")

if __name__ == "__main__":
    main()
"""
        
        # Write template file
        with open(template_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created template: {template_path}")

def copy_fixed_tests(source_dir=None, force=False):
    """Copy fixed test files to the fixed_tests directory."""
    if source_dir is None:
        # Look for fixed test files in current directory
        source_files = [f for f in os.listdir(ROOT_DIR) if f.startswith("test_hf_") and f.endswith(".py")]
        source_dir = ROOT_DIR
    else:
        # Use specified source directory
        source_files = [f for f in os.listdir(source_dir) if f.startswith("test_hf_") and f.endswith(".py")]
    
    if not source_files:
        logger.warning(f"No test files found in {source_dir}")
        return
    
    # Copy each file
    for filename in source_files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(FIXED_TESTS_DIR, filename)
        
        # Skip if file exists and not forcing
        if os.path.exists(dest_path) and not force:
            logger.info(f"File already exists (skipping): {dest_path}")
            continue
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied test file: {dest_path}")
        
        # If original has a .fixed version, copy that too
        fixed_source = source_path + ".fixed"
        if os.path.exists(fixed_source):
            fixed_dest = dest_path + ".fixed"
            shutil.copy2(fixed_source, fixed_dest)
            logger.info(f"Copied fixed version: {fixed_dest}")

def clean_existing_files(directory, pattern="*"):
    """Clean existing files in a directory matching pattern."""
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        logger.info(f"No files matching '{pattern}' found in {directory}")
        return
    
    # Remove each file
    for file_path in files:
        if os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Cleanup and prepare HuggingFace test integration")
    parser.add_argument("--templates", action="store_true", help="Create basic templates")
    parser.add_argument("--copy-tests", type=str, help="Copy tests from specified directory")
    parser.add_argument("--clean", action="store_true", help="Clean existing files")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    
    args = parser.parse_args()
    
    # Create required directories
    ensure_directory(TEMPLATES_DIR)
    ensure_directory(FIXED_TESTS_DIR)
    ensure_directory(COLLECTED_RESULTS_DIR)
    
    # Clean existing files if requested
    if args.clean:
        if input("Clean all existing files? (y/n): ").lower() == 'y':
            clean_existing_files(TEMPLATES_DIR, "*.py")
            clean_existing_files(FIXED_TESTS_DIR, "*.py")
            clean_existing_files(COLLECTED_RESULTS_DIR, "*.json")
            logger.info("Cleaned existing files")
    
    # Create basic templates if requested
    if args.templates:
        create_basic_templates(force=args.force)
    
    # Copy fixed tests if requested
    if args.copy_tests:
        copy_fixed_tests(source_dir=args.copy_tests, force=args.force)
    elif not args.templates:
        # Default behavior: copy tests from current directory
        copy_fixed_tests(force=args.force)
    
    logger.info("Environment prepared for test integration")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())