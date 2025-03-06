#!/usr/bin/env python3
"""
Update Test Generators for Phase 16

This script updates all test generators to use the centralized hardware detection
module and ensure consistent hardware detection and optimization across all tests.

Usage:
  python update_generators/update_test_generators.py
"""

import os
import re
import sys
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("update_generators")

# Path to centralized hardware detection module
HARDWARE_DETECTION_MODULE_PATH = Path(__file__).parent.parent / "centralized_hardware_detection"

# Path to generators that need updating
GENERATORS_TO_UPDATE = [
    Path(__file__).parent.parent / "fixed_merged_test_generator.py",
    Path(__file__).parent.parent / "merged_test_generator.py",
    Path(__file__).parent.parent / "integrated_skillset_generator.py",
    Path(__file__).parent.parent / "implementation_generator.py",
    Path(__file__).parent.parent / "template_hardware_detection.py",
]

def backup_file(file_path: Path) -> Path:
    """
    Create a backup of a file before modifying it.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".py.bak_{timestamp}")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def detect_duplicate_hardware_detection(file_content: str) -> bool:
    """
    Detect if file has duplicate hardware detection code.
    
    Args:
        file_content: Content of the file to check
        
    Returns:
        True if duplicate hardware detection found
    """
    # Look for patterns that indicate hardware detection
    hardware_detection_patterns = [
        r"def\s+check_hardware\s*\(\s*\)",  # check_hardware function definition
        r"HAS_CUDA\s*=",                   # Hardware capability flags
        r"HAS_ROCM\s*=",
        r"HAS_MPS\s*=",
        r"torch\.cuda\.is_available\(\)",   # Hardware detection calls
    ]
    
    # Count occurrences of hardware detection patterns
    occurrences = 0
    for pattern in hardware_detection_patterns:
        matches = re.findall(pattern, file_content)
        if len(matches) > 0:
            occurrences += len(matches)
    
    # If multiple occurrences, likely has duplicate hardware detection
    return occurrences >= 2

def update_generator_file(file_path: Path) -> bool:
    """
    Update a generator file to use centralized hardware detection.
    
    Args:
        file_path: Path to the generator file to update
        
    Returns:
        True if file was updated
    """
    if not file_path.exists():
        logger.warning(f"File {file_path} not found, skipping")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Detect if it needs updating
    if not detect_duplicate_hardware_detection(content):
        logger.info(f"File {file_path} doesn't have duplicate hardware detection, may not need updating")
    
    # Create backup
    backup_file(file_path)
    
    # Replace direct hardware detection code
    # Pattern 1: Import section for hardware detection
    hardware_import_pattern = r"""
# (?:Hardware Detection|Try to import torch).*?
import os
import importlib\.util

# Try to import torch.*?
HW_CAPABILITIES = check_hardware\(\)
"""
    
    # New import section
    new_hardware_import = """
# Centralized hardware detection
from centralized_hardware_detection import (
    get_hardware_manager,
    get_capabilities,
    get_web_optimizations,
    get_browser_info,
    get_hardware_detection_code
)

# Get hardware capabilities
hw_manager = get_hardware_manager()
HW_CAPABILITIES = get_capabilities()

# Quick access to hardware flags
HAS_CUDA = hw_manager.has_cuda
HAS_ROCM = hw_manager.has_rocm
HAS_MPS = hw_manager.has_mps
HAS_OPENVINO = hw_manager.has_openvino
HAS_WEBNN = hw_manager.has_webnn
HAS_WEBGPU = hw_manager.has_webgpu
"""
    
    # Replace if found
    content_updated = re.sub(hardware_import_pattern, new_hardware_import, content, flags=re.DOTALL)
    
    # Pattern 2: check_hardware function definition
    check_hardware_pattern = r"""
def check_hardware\(\):
    .*?
    return capabilities
"""
    
    # New check_hardware function (for backward compatibility)
    new_check_hardware = """
def check_hardware():
    """Check available hardware and return capabilities (for backward compatibility)."""
    return get_capabilities()
"""
    
    # Replace if found
    content_updated = re.sub(check_hardware_pattern, new_check_hardware, content_updated, flags=re.DOTALL)
    
    # Pattern 3: Web platform optimization functions
    web_optimizations_pattern = r"""
def apply_web_platform_optimizations\(.*?\):
    .*?
    return optimizations
"""
    
    # New web optimizations function
    new_web_optimizations = """
def apply_web_platform_optimizations(model_type, implementation_type=None):
    """Apply web platform optimizations (for backward compatibility)."""
    return get_web_optimizations(model_type, implementation_type)
"""
    
    # Replace if found
    content_updated = re.sub(web_optimizations_pattern, new_web_optimizations, content_updated, flags=re.DOTALL)
    
    # Pattern 4: Browser detection function
    browser_detection_pattern = r"""
def detect_browser_for_optimizations\(\):
    .*?
    return browser_info
"""
    
    # New browser detection function
    new_browser_detection = """
def detect_browser_for_optimizations():
    """Detect browser for optimizations (for backward compatibility)."""
    return get_browser_info()
"""
    
    # Replace if found
    content_updated = re.sub(browser_detection_pattern, new_browser_detection, content_updated, flags=re.DOTALL)
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content_updated)
    
    logger.info(f"Updated {file_path} to use centralized hardware detection")
    return True

def main():
    """Main function to update all test generators."""
    logger.info("Starting update of test generators for Phase 16")
    
    # Check if centralized hardware detection module exists
    if not HARDWARE_DETECTION_MODULE_PATH.exists():
        logger.error(f"Centralized hardware detection module not found at {HARDWARE_DETECTION_MODULE_PATH}")
        return 1
    
    # Update each generator
    updated_files = 0
    for generator_path in GENERATORS_TO_UPDATE:
        if update_generator_file(generator_path):
            updated_files += 1
    
    logger.info(f"Updated {updated_files}/{len(GENERATORS_TO_UPDATE)} test generators")
    
    return 0

if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())