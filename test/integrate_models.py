#!/usr/bin/env python3
"""
Script to integrate successfully generated model tests into the main codebase
and update the HF_MODEL_COVERAGE_ROADMAP.md accordingly.
"""

import os
import sys
import re
import json
import time
import shutil
import logging
import argparse
import importlib.util
from glob import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from enhanced_generator
try:
    from enhanced_generator import (
        MODEL_REGISTRY, 
        get_model_architecture,
        validate_generated_file
    )
except ImportError as e:
    logger.error(f"Failed to import from enhanced_generator: {e}")
    sys.exit(1)

def find_generated_models(directories: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Find all successfully generated model test files in the given directories.
    
    Args:
        directories: List of directories to search
        
    Returns:
        Dictionary of {model_name: details}
    """
    generated_models = {}
    
    for directory in directories:
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        
        # Find all Python test files
        test_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and file.startswith('test_'):
                    test_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(test_files)} test files in {directory}")
        
        for file_path in test_files:
            # Extract model name from filename (test_<model_name>.py)
            file_name = os.path.basename(file_path)
            match = re.match(r'test_(.+)\.py', file_name)
            if not match:
                continue
            
            model_name = match.group(1).replace('-', '_')
            
            # Verify file syntax
            is_valid, validation_msg = validate_generated_file(file_path)
            
            if is_valid:
                # Try to determine architecture
                try:
                    architecture = get_model_architecture(model_name)
                except:
                    architecture = "unknown"
                
                # Build file information
                file_info = {
                    "model": model_name,
                    "file_path": file_path,
                    "file_name": file_name,
                    "architecture": architecture,
                    "valid": True,
                    "directory": directory
                }
                
                generated_models[model_name] = file_info
            else:
                logger.warning(f"Invalid test file: {file_path} - {validation_msg}")
    
    logger.info(f"Found a total of {len(generated_models)} valid model test files")
    return generated_models

def update_roadmap(roadmap_file: str, integrated_models: List[str]) -> bool:
    """
    Update the implementation status in the HF_MODEL_COVERAGE_ROADMAP.md file.
    
    Args:
        roadmap_file: Path to the roadmap markdown file
        integrated_models: List of newly integrated model names
        
    Returns:
        True if the roadmap file was updated, False otherwise
    """
    if not os.path.isfile(roadmap_file):
        logger.error(f"Roadmap file not found: {roadmap_file}")
        return False
    
    # Read the roadmap file
    with open(roadmap_file, 'r') as f:
        content = f.read()
    
    # Create a copy of the original file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_file = f"{roadmap_file}.bak.{timestamp}"
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Normalize model names for comparison
    # Add variations for matching within the roadmap
    model_variations = []
    for model in integrated_models:
        model_name = model.lower().strip()
        model_variations.append(model_name)
        
        # Add common variations
        if '_' in model_name:
            model_variations.append(model_name.replace('_', '-'))
        if '-' in model_name:
            model_variations.append(model_name.replace('-', '_'))
        if model_name.startswith('hf_'):
            model_variations.append(model_name[3:])
        if not model_name.startswith('hf_') and not model_name.startswith('test_'):
            model_variations.append(f"hf_{model_name}")
    
    # Remove duplicates
    model_variations = list(set(model_variations))
    
    # Update implementation status for each model
    updated_content = content
    models_updated = []
    
    # Look for models in checklist items
    for model_name in model_variations:
        pattern = re.compile(f"- \\[ \\]\\s+({re.escape(model_name)}[\\s,.:])", re.IGNORECASE)
        
        # Keep track of how many replacements were made
        updated_content_new, count = pattern.subn(
            f"- [x] \\1 - Implemented on {datetime.now().strftime('%B %d, %Y')}", 
            updated_content
        )
        
        if count > 0:
            updated_content = updated_content_new
            models_updated.append(model_name)
    
    # Update the current status section
    total_models_match = re.search(r'\*\*Total Models Tracked:\*\*\s*(\d+)', updated_content)
    implemented_match = re.search(r'\*\*Implemented Models:\*\*\s*(\d+)\s*\(([0-9.]+)%\)', updated_content)
    missing_match = re.search(r'\*\*Missing Models:\*\*\s*(\d+)\s*\(([0-9.]+)%\)', updated_content)
    
    if total_models_match and implemented_match and missing_match:
        total_models = int(total_models_match.group(1))
        implemented = int(implemented_match.group(1)) + len(set(models_updated))
        missing = total_models - implemented
        
        implemented_pct = round(implemented / total_models * 100, 1)
        missing_pct = round(missing / total_models * 100, 1)
        
        updated_content = re.sub(
            r'\*\*Implemented Models:\*\*\s*\d+\s*\([0-9.]+%\)',
            f"**Implemented Models:** {implemented} ({implemented_pct}%)",
            updated_content
        )
        
        updated_content = re.sub(
            r'\*\*Missing Models:\*\*\s*\d+\s*\([0-9.]+%\)',
            f"**Missing Models:** {missing} ({missing_pct}%)",
            updated_content
        )
    
    # Write the updated content if changes were made
    if updated_content != content:
        with open(roadmap_file, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated roadmap file with {len(set(models_updated))} newly implemented models")
        logger.info(f"New stats: {implemented}/{total_models} implemented ({implemented_pct}%)")
        return True
    else:
        logger.warning("No updates needed in the roadmap file")
        return False

def consolidate_tests(test_dir: str, integrated_models: Dict[str, Dict[str, Any]]) -> Tuple[List[str], int]:
    """
    Consolidate test files into a single directory, organizing by architecture.
    
    Args:
        test_dir: Path to the output test directory
        integrated_models: Dictionary of model details from find_generated_models()
        
    Returns:
        Tuple of (list of copied files, number of files copied)
    """
    # Create output directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Create architecture subdirectories
    architectures = ["encoder-only", "decoder-only", "encoder-decoder", "vision", "vision-text", "speech", "multimodal", "unknown"]
    
    for arch in architectures:
        os.makedirs(os.path.join(test_dir, arch), exist_ok=True)
    
    # Copy files to appropriate directories
    copied_files = []
    for model_name, details in integrated_models.items():
        arch = details["architecture"]
        if arch not in architectures:
            arch = "unknown"
        
        # Determine source and destination paths
        src_path = details["file_path"]
        dst_path = os.path.join(test_dir, arch, details["file_name"])
        
        # Skip if already copied
        if os.path.exists(dst_path):
            logger.info(f"File already exists: {dst_path}")
            continue
        
        # Copy the file
        try:
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_path)
            logger.info(f"Copied {src_path} to {dst_path}")
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
    
    return copied_files, len(copied_files)

def generate_integration_report(integrated_models: Dict[str, Dict[str, Any]], copied_files: List[str], 
                               updated_roadmap: bool, output_file: str) -> None:
    """
    Generate a report of integrated models.
    
    Args:
        integrated_models: Dictionary of model details
        copied_files: List of copied file paths
        updated_roadmap: Whether the roadmap was updated
        output_file: Output file path for the report
    """
    # Group models by architecture
    arch_groups = {}
    for model_name, details in integrated_models.items():
        arch = details["architecture"]
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append((model_name, details))
    
    with open(output_file, 'w') as f:
        f.write(f"# Model Integration Report\n\n")
        f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write(f"## Summary\n\n")
        f.write(f"- **Total models integrated:** {len(integrated_models)}\n")
        f.write(f"- **Files copied:** {len(copied_files)}\n")
        f.write(f"- **Roadmap updated:** {updated_roadmap}\n\n")
        
        # Architecture statistics
        f.write(f"## Models by Architecture\n\n")
        for arch, models in sorted(arch_groups.items()):
            f.write(f"### {arch.capitalize()} ({len(models)})\n\n")
            for model_name, details in sorted(models):
                f.write(f"- **{model_name}**\n")
                f.write(f"  - File: {details['file_name']}\n")
                f.write(f"  - Source: {details['directory']}\n")
            f.write("\n")
        
        # Next steps
        f.write(f"## Next Steps\n\n")
        f.write(f"1. **Verify Tests**: Run the integrated tests to ensure they function correctly\n")
        f.write(f"2. **Add Remaining Models**: Add any remaining models from the roadmap\n")
        f.write(f"3. **Document Coverage**: Update documentation to reflect the improved coverage\n")
        f.write(f"4. **Commit Changes**: Commit the changes to the repository\n")
    
    logger.info(f"Integration report written to {output_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate model tests and update roadmap")
    parser.add_argument("--input-dirs", nargs="+", default=["critical_priority_tests", "high_priority_tests", "medium_priority_tests"],
                        help="Directories containing generated tests")
    parser.add_argument("--output-dir", default="integrated_tests", help="Output directory for consolidated tests")
    parser.add_argument("--roadmap", default="skills/HF_MODEL_COVERAGE_ROADMAP.md", help="Path to roadmap file")
    parser.add_argument("--report", default="model_integration_report.md", help="Output file for integration report")
    parser.add_argument("--no-update-roadmap", action="store_true", help="Don't update the roadmap file")
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Find all generated models
    logger.info(f"Finding generated models in {args.input_dirs}")
    integrated_models = find_generated_models(args.input_dirs)
    
    if not integrated_models:
        logger.error("No valid model test files found")
        return 1
    
    # Consolidate tests into output directory
    logger.info(f"Consolidating tests into {args.output_dir}")
    copied_files, copy_count = consolidate_tests(args.output_dir, integrated_models)
    
    logger.info(f"Copied {copy_count} test files to {args.output_dir}")
    
    # Update roadmap if requested
    updated_roadmap = False
    if not args.no_update_roadmap:
        logger.info(f"Updating roadmap file: {args.roadmap}")
        updated_roadmap = update_roadmap(args.roadmap, list(integrated_models.keys()))
    
    # Generate integration report
    generate_integration_report(integrated_models, copied_files, updated_roadmap, args.report)
    
    logger.info(f"Integration complete!")
    logger.info(f"See the report at: {args.report}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())